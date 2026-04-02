from pathlib import Path

import h5py
import numpy as np
import pycolmap

from mpsfm.baseclass import BaseClass
from mpsfm.sfm.scene.pointcov import PointCovs
from mpsfm.sfm.scene.reconstruction.mixins import ReconstructionMixin


class ColmapReconstructionWrapper:
    def __getattr__(self, name):
        if name == "images":
            return self._images  # Return the alternate variable
        if name == "cameras":
            return self._cameras
        return getattr(self.rec, name)


class MpsfmReconstruction(BaseClass, ColmapReconstructionWrapper, ReconstructionMixin):
    """MP-SfM reconstruction wrapper for COLMAP reconstruction."""

    default_conf = {
        "image": {},
        "colmap_options": "<--->",
        "normscale": 387,  # ETH3D:387.5 T&T: 480
        "depth_type": "monocular",
        "matches_mode": "<--->",
        "verbose": 0,
    }
    best_next_ref_imid = None

    def _propagate_conf(self):
        self.conf.image.depth_type = self.conf.depth_type
        self.conf.image.verbose = self.conf.verbose

    def _init(self):
        self._rec = pycolmap.Reconstruction()
        self._images = {}
        self._cameras = {}

        self.point_covs = PointCovs()
        self.is_primary_2d = None
        self.refrec_dir = None
        self.references = None
        self.images_dir = None
        # 累计的世界系相似变换（来自多次 normalize）：x' = s_total * x + t_total
        # 用于将先验外参从“原世界系”映射到“当前世界系”
        self.world_sim_scale = 1.0
        self.world_sim_translation = np.zeros(3, dtype=float)
        self.world_sim_rotation = np.eye(3, dtype=float)

    @property
    def image_ids(self) -> list[int]:
        """Returns the list of image IDs."""
        return list(self.images.keys())

    @property
    def registered_images(self) -> dict[int : pycolmap.Image]:
        return {imid: self.images[imid] for imid in self.reg_image_ids()}

    @property
    def rec(self):
        return self._rec

    @rec.setter
    def rec(self, value):
        raise AttributeError("Use set_rec() to set 'rec'")

    def set_rec(self, rec):
        self._rec = rec
        self._images = {imid: rec.images[imid] for imid in rec.images}
        self._cameras = {camid: rec.cameras[camid] for camid in rec.cameras}

    def add_camera(self, camera):
        self.rec.add_camera(camera)
        self._cameras[camera.camera_id] = camera

    def add_image(self, image):
        self.rec.add_image(image)
        self._images[image.image_id] = image

    def imid(self, name):
        for imid, image in self.images.items():
            if image.name == name:
                return imid
        return None

    def keypoints(self, imid):
        """keypoints of image"""
        return np.array([kp.xy for kp in self.images[imid].points2D])

    def keypoints_with_p3d(self, imid):
        """keypoints of image"""
        kps = []
        pt2ds = []
        for p2Did, kp in enumerate(self.images[imid].points2D):
            if kp.has_point3D():
                kps.append(kp.xy)
                pt2ds.append(p2Did)
        return np.array(kps), np.array(pt2ds)

    def camera(self, imid):
        """camera model of image"""
        return self.rec.cameras[self.images[imid].camera_id]

    def filtered_image_pairs(self, two_view_geom, two_view_config=2) -> set[frozenset]:
        pairs = set()
        for imid1 in self.image_ids:
            for imid2 in self.image_ids:
                if imid1 == imid2 or {imid1, imid2} in pairs:
                    continue
                two_view_geom_, success = two_view_geom(self.images[imid1].name, self.images[imid2].name)
                if not success:
                    continue
                if two_view_config == two_view_geom_.config:
                    pairs.add(frozenset([imid1, imid2]))
        return pairs

    def normalize(self):
        """Normalizes reconstruction"""
        sim3d = self.rec.normalize(False, 5, 0.2, 0.8, False)
        scale, translation = sim3d.scale, sim3d.translation
        # 若返回含旋转，获取旋转矩阵；否则使用单位阵
        try:
            Rg = sim3d.rotation.matrix() if hasattr(sim3d, "rotation") else np.eye(3, dtype=float)
        except Exception:
            Rg = np.eye(3, dtype=float)
        
        self.world_sim_translation = scale * (Rg @ self.world_sim_translation) + translation
        self.world_sim_rotation = Rg @ self.world_sim_rotation
        self.world_sim_scale *= scale
        # 同步深度尺度到新的世界尺度
        self.normalize_depths(scale)
        return True

    def align_prior_pose_to_current_world(self, pose: pycolmap.Rigid3d) -> pycolmap.Rigid3d:
        """将先验位姿（定义在原世界系）映射到当前世界系。

        通用 Sim3（s_tot, R_tot, t_tot）对应关系：
        R'_cw = R_cw * R_tot^T
        t'_cw = s_tot * t_cw - R'_cw * t_tot
        """
        try:
            s = float(self.world_sim_scale)
            t = np.asarray(self.world_sim_translation, dtype=float)
            R_tot = np.asarray(self.world_sim_rotation, dtype=float)
            R_cw = pose.rotation
            t_cw = pose.translation
            # R'_cw = R_cw * R_tot^T
            R_new_mat = R_cw.matrix() @ R_tot.T
            R_new = pycolmap.Rotation3d(R_new_mat)
            # t'_cw = s * t_cw - R'_cw * t_tot
            t_new = s * t_cw - (R_new * t)
            
            return pycolmap.Rigid3d(R_new, t_new)
        except Exception:
            return pose

    def cache_depths(self, priors_dir: Path):
        with h5py.File(priors_dir, "w") as f:
            for imid, image in self.registered_images.items():
                if image.depth is None:
                    continue
                if image.depth.data is not None:
                    f.create_dataset(f"depths/{imid}", data=image.depth.data, compression="gzip")
                if image.depth.continuity_mask is not None:
                    f.create_dataset(f"continuity/{imid}", data=image.depth.continuity_mask, compression="gzip")
                if image.depth.scale is not None:
                    f.create_dataset(f"scales/{imid}", data=image.depth.scale)
                if image.depth.data_prior is not None:
                    f.create_dataset(f"pdepth/{imid}", data=image.depth.data_prior, compression="gzip")

    def write(self, output_path: Path):
        """Writes reconstruction"""
        rec_dir = output_path / "rec"
        rec_dir.mkdir(parents=True, exist_ok=True)
        self.rec.write(rec_dir)
        priors_dir = output_path / "depths.h5"
        self.cache_depths(priors_dir)

    def read(self, input_path):
        """Reads reconstruction"""
        self.rec.read(input_path)

    def find_local_bundle_ids(self, refimid, num_images=None):
        """Finds local bundle ids"""
        impl = pycolmap.IncrementalMapperImpl()
        confs = {
            k: v for k, v in self.conf.colmap_options.items() if k in {"local_ba_min_tri_angle", "local_ba_num_images"}
        }
        if num_images is not None:
            confs["local_ba_num_images"] = num_images
        options = pycolmap.IncrementalMapperOptions(**confs)
        return impl.find_local_bundle(options, refimid, self.rec)
