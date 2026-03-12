import numpy as np
import pycolmap
from pycolmap import IncrementalTriangulator, IncrementalTriangulatorOptions

from mpsfm.baseclass import BaseClass
from mpsfm.utils.geometry import has_point_positive_depth


class ColmapTriangulatorWrapper:
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self._triangulator, name)


class MpsfmTriangulator(BaseClass, ColmapTriangulatorWrapper):
    """MP-SfM triangulator wrapper for COLMAP triangulator."""

    default_conf = {
        "hard_angle": 1.5,  # colmap default
        "colmap_options": "<--->",
        "new_retry_nbatch": 5,
        "re_ignore_two_view_tracks": False,
        "retri_min_angle": 1.5,
        "lift_low_parallax": True,
        "nsafe_threshold": 60,
        "verbose": 0,
    }

    def _init(self, mpsfm_rec, correspondences_graph, **kwargs):
        self.mpsfm_rec = mpsfm_rec
        self._triangulator = IncrementalTriangulator(correspondences_graph, self.mpsfm_rec.rec, self.mpsfm_rec.obs)

        self.options = IncrementalTriangulatorOptions(
            **{
                k: v
                for k, v in self.conf.colmap_options.items()
                if k in set(IncrementalTriangulatorOptions().todict().keys())
            }
        )

    def triangulate_image(self, *args, **kwargs) -> bool:
        """Triangulate image with the given image id."""
        return self._triangulate_image(*args, **kwargs)

    def _triangulate_image(self, imid, **kwargs) -> bool:
        in3D = set(self.mpsfm_rec.points3D.keys())
        self._triangulator.triangulate_image(self.options, imid)
        if self.conf.lift_low_parallax:
            diff3D = np.array(list(set(self.mpsfm_rec.points3D.keys()) - in3D))
            if len(diff3D) == 0:
                return True
            risky_mask = self.mpsfm_rec.find_points3D_with_small_triangulation_angle(
                min_angle=self.conf.hard_angle, point3D_ids=diff3D
            )
            diff3D = diff3D[risky_mask]
            for point3D_id in diff3D:
                point3D = self.mpsfm_rec.points3D[point3D_id]
                imids = [el.image_id for el in point3D.track.elements]
                ptids = [el.point2D_idx for el in point3D.track.elements]
                cams_from_world = [self.mpsfm_rec.images[imid].cam_from_world for imid in imids]

                self.mpsfm_rec.obs.delete_point3D(point3D_id)
                for liftid, limid in enumerate(imids):
                    if self.mpsfm_rec.images[limid].depth.activated:
                        lift_image = self.mpsfm_rec.images[limid]
                        xy = np.array([self.mpsfm_rec.images[limid].points2D[ptids[liftid]].xy])
                        valid = self.mpsfm_rec.images[limid].depth.valid_at_kps(xy)
                        if not valid[0]:
                            continue
                        d = self.mpsfm_rec.images[limid].depth.data_at_kps(xy)[:, None]

                        lift_camera = self.mpsfm_rec.rec.cameras[lift_image.camera_id]
                        xyz = lift_image.cam_from_world.inverse() * (
                            np.concatenate([lift_camera.cam_from_img(xy), np.ones((xy.shape[0], 1))], -1) * d
                        )

                        track = pycolmap.Track()
                        for imid_, ptid, cam_from_world in zip(imids, ptids, cams_from_world):
                            if has_point_positive_depth(cam_from_world.matrix(), xyz):
                                track.add_element(imid_, ptid)
                        self.mpsfm_rec.obs.add_point3D(xyz[0], track)
                        break

        return True

    def complete_image(self, imid):
        return self._triangulator.complete_image(self.options, imid)

    def complete_all_tracks(self):
        return self._triangulator.complete_all_tracks(self.options)

    def complete_tracks(self, points3D):
        return self._triangulator.complete_tracks(self.options, points3D)

    def merge_tracks(self, points3D):
        return self._triangulator.merge_tracks(self.options, points3D)

    def merge_all_tracks(self):
        return self._triangulator.merge_all_tracks(self.options)

    def retriangulate(self):
        colmap_options = dict(self.conf.colmap_options)  # copy to avoid modifying the original
        colmap_options["ignore_two_view_tracks"] = self.conf.re_ignore_two_view_tracks

        if self.conf.new_retry_nbatch is not None:
            risky_imids = []
            for imid in list(self.mpsfm_rec.registered_images):
                p3d = set(self.mpsfm_rec.images[imid].point3D_ids()) - {18446744073709551615}
                nsafe = (np.array([len(self.mpsfm_rec.points3D[p].track.elements) for p in p3d]) > 2).sum()
                self.log(f"Image {imid} has {nsafe} points with track length > 2", level=3)
                if nsafe < self.conf.nsafe_threshold:
                    risky_imids.append(imid)
            expanded_risky_imids = []
            for imid in risky_imids:
                expanded_risky_imids.append(self.mpsfm_rec.find_local_bundle_ids(imid, self.conf.new_retry_nbatch))
            expanded_risky_imids = sum(expanded_risky_imids, [])
            risky_imids = risky_imids + expanded_risky_imids
        else:
            risky_imids = []
        self.log("safe iamges:", set(self.mpsfm_rec.registered_images) - set(risky_imids), level=3)
        self.log("risky images:", risky_imids, level=3)
        out = self._triangulator.retriangulate(self.options, set(risky_imids))  # Pass the set of image IDs to ignore

        p3ds = np.array(list(self.mpsfm_rec.points3D.keys()))
        risky_mask = self.mpsfm_rec.find_points3D_with_small_triangulation_angle(
            min_angle=self.conf.retri_min_angle, point3D_ids=p3ds
        )
        risky_p3ds = p3ds[risky_mask]

        ids = []
        count = 0
        for point3D_id in risky_p3ds:
            point3D = self.mpsfm_rec.points3D[point3D_id]
            imids = [el.image_id for el in point3D.track.elements]
            ptids = [el.point2D_idx for el in point3D.track.elements]
            cams_from_world = [self.mpsfm_rec.images[imid].cam_from_world for imid in imids]

            self.mpsfm_rec.obs.delete_point3D(point3D_id)
            for liftid, imid in enumerate(imids):
                if self.mpsfm_rec.images[imid].depth.activated:
                    lift_image = self.mpsfm_rec.images[imid]
                    xy = np.array([self.mpsfm_rec.images[imid].points2D[ptids[liftid]].xy])
                    valid = self.mpsfm_rec.images[imid].depth.valid_at_kps(xy)
                    if not valid[0]:
                        continue
                    d = self.mpsfm_rec.images[imid].depth.data_at_kps(xy)[:, None]

                    lift_camera = self.mpsfm_rec.rec.cameras[lift_image.camera_id]
                    xyz = lift_image.cam_from_world.inverse() * (
                        np.concatenate([lift_camera.cam_from_img(xy), np.ones((xy.shape[0], 1))], -1) * d
                    )

                    track = pycolmap.Track()
                    for imid_, ptid, cam_from_world in zip(imids, ptids, cams_from_world):
                        if has_point_positive_depth(cam_from_world.matrix(), xyz):
                            track.add_element(imid_, ptid)
                    pts3didx = self.mpsfm_rec.obs.add_point3D(xyz[0], track)
                    ids.append(pts3didx)
                    count += 1
                    break

        return out

    def complete_and_merge_all_tracks(self) -> int:
        """Completes and merges all tracks in recosntruction"""
        num_completed = self.complete_all_tracks()
        num_merged = self.merge_all_tracks()
        return num_completed + num_merged

    def complete_and_merge_tracks(self, points3D) -> int:
        """Completes and merges tracks of sepcific 3d points"""
        num_completed = self.complete_tracks(points3D)
        num_merged = self.merge_tracks(points3D)
        return num_completed + num_merged
