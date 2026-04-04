from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from mpsfm.baseclass import BaseClass
from mpsfm.sfm.estimators import AbsolutePose, RelativePose
from mpsfm.utils.geometry import calculate_triangulation_angle, has_point_positive_depth
from mpsfm.utils.prior_pose import load_prior_pose_arrays, pose_name_variants


class MpsfmRegistration(BaseClass):
    """MP-SfM Registration class. This class is used to register images and triangulate points."""

    default_conf = {
        "lifted_registration": True,
        "absolute_pose": {},
        "relative_pose": {},
        "reduce_min_inliers_at_failure": 6,
        "parallax_thresh": 1.5,
        "combined_triangle_thresh": 1.5,
        "robust_triangles": 1.0,
        "resample_bunlde": False,
        "colmap_options": "<--->",
        "verbose": 0,
        "use_prior_poses": False,
        "pose_config_path": None,
        "use_prior_pose_pairing": True,
        "ba_refine_prior_pose": True,
        "refine_init_prior_pose": True,
        "epipolar_refine_min_matches": 12,
        "epipolar_refine_max_iters": 300,
        "epipolar_refine_loss": "soft_l1",
        "epipolar_refine_f_scale": 1e-2,
        "init_soft_filter_min_weight": 0.05,
        "init_refine_irls_iters": 3,
        "init_refine_stage_a_max_rot_deg": 3.0,
        "init_refine_stage_b_max_rot_deg": 5.0,
        "init_refine_stage_b_max_trans_deg": 10.0,
        "init_refine_rotation_reg_weight": 5e-3,
        "init_refine_translation_reg_weight": 2e-3,
        "init_core_inlier_residual_mult": 2.5,
        "init_core_min_weight": 0.15,
        "init_core_min_matches": 12,
        "init_core_min_tri_angle": 0.1,
        "init_support_inlier_residual_mult": 4.0,
        "init_support_min_weight": 0.05,
        "init_support_min_matches": 24,
        "init_support_min_tri_angle": 0.03,
        "init_spatial_reweight": True,
        "init_spatial_grid_rows": 4,
        "init_spatial_grid_cols": 4,
        "init_spatial_reweight_power": 0.5,
        "init_spatial_reweight_min": 0.35,
        "init_spatial_reweight_max": 2.5,
        "init_pseudo_inlier_residual_mult": 2.5,
        "init_pseudo_inlier_min_weight": 0.15,
        "init_pseudo_inlier_min_matches": 12,
        "init_pseudo_min_tri_angle": 0.1,
        "init_reproj_refine_min_points": 8,
        "init_reproj_refine_max_iters": 60,
        "init_reproj_refine_thresh": 4.0,
        "init_reproj_refine_max_rot_deg": 1.5,
        "init_reproj_refine_max_trans_deg": 3.0,
        "init_reproj_refine_rot_reg_weight": 1e-3,
        "init_reproj_refine_trans_reg_weight": 1e-3,
        "init_joint_refine_reproj_thresh": 12.0,
        "init_joint_refine_min_depth_matches": 8,
        "init_joint_refine_reproj_weight": 1.0,
        "init_joint_refine_rot_reg_weight": 1e-3,
        "init_joint_refine_trans_reg_weight": 1e-3,
        "init_joint_refine_scale_reg_weight": 1e-2,
        "init_joint_refine_max_depth_scale": 5.0,
        "init_joint_refine_max_rot_change_deg": 4.0,
        "init_joint_refine_max_trans_change_deg": 6.0,
        "init_joint_refine_max_scale_change_ratio": 1.5,
        "init_joint_refine_require_improvement": True,
        "init_stageb_min_support_count": 32,
        "init_stageb_min_support_ratio": 0.6,
        "init_stageb_min_support_tri_median_deg": 0.8,
        "init_stageb_min_support_tri_p25_deg": 0.3,
        "init_lookahead_topk": 5,
        "init_lookahead_min_inlier_gain": 6,
        "init_lookahead_min_reproj_gain_px": 0.25,
        "init_lookahead_min_score_gain": 2.0,
        "refine_remaining_prior_pose": True,
        "prior_pose_refine_min_corrs": 24,
        "prior_pose_refine_max_iters": 150,
        "prior_pose_refine_reproj_thresh": 12.0,
    }

    def _init(self, mpsfm_rec, correspondences, triangulator, **kwargs):
        self.mpsfm_rec = mpsfm_rec
        self.correspondences = correspondences
        self.triangulator = triangulator
        self.relative_pose_estimator = RelativePose(self.conf.relative_pose)
        self.absolute_pose_estimator = AbsolutePose(self.conf.absolute_pose)

        self.half_ap_min_inliers = 0
        self.registration_cache = defaultdict(dict)
        self.prior_poses = {}
        if self.conf.use_prior_poses and self.conf.pose_config_path:
            self._load_prior_poses()

    def _load_prior_poses(self):
        """Load prior camera poses from a YAML config."""
        try:
            pose_entries = load_prior_pose_arrays(Path(self.conf.pose_config_path))
            self.prior_poses = {
                name: pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), translation)
                for name, (rotation, translation, _) in pose_entries.items()
            }
            print(f"Loaded {len(self.prior_poses)} prior pose entries from {self.conf.pose_config_path}")
        except Exception as e:
            print(f"Warning: Failed to load prior poses: {e}")

    def has_prior_pose(self, image_name: str) -> bool:
        return any(name in self.prior_poses for name in pose_name_variants(image_name))

    def get_prior_pose(self, image_name: str):
        for name in pose_name_variants(image_name):
            pose = self.prior_poses.get(name)
            if pose is not None:
                return pose
        return None

    @staticmethod
    def _skew(vec):
        x, y, z = vec
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)

    @staticmethod
    def _normalize_translation(t):
        norm = np.linalg.norm(t)
        if norm < 1e-9:
            return t
        return t / norm

    @staticmethod
    def _camera_center_from_pose(pose: pycolmap.Rigid3d) -> np.ndarray:
        R = pose.rotation.matrix()
        t = np.asarray(pose.translation, dtype=np.float64)
        return -R.T @ t

    @staticmethod
    def _estimate_sim3(src_pts: np.ndarray, dst_pts: np.ndarray):
        if src_pts.shape[0] < 3 or dst_pts.shape[0] < 3:
            return None
        src_mean = src_pts.mean(axis=0)
        dst_mean = dst_pts.mean(axis=0)
        src_centered = src_pts - src_mean
        dst_centered = dst_pts - dst_mean
        cov = (dst_centered.T @ src_centered) / float(src_pts.shape[0])
        U, S, Vt = np.linalg.svd(cov)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        var_src = np.mean(np.sum(src_centered**2, axis=1))
        if var_src < 1e-12:
            return None
        scale = float(np.sum(S) / var_src)
        t = dst_mean - scale * (R @ src_mean)
        return scale, R, t

    def _update_world_sim_from_priors(self, min_shared: int = 3) -> bool:
        if not self.prior_poses:
            return False
        src_centers = [] # src_centers：先验世界坐标系中的相机中心
        dst_centers = [] # dst_centers：当前世界坐标系中的相机中心
        for imid, image in self.mpsfm_rec.registered_images.items():
            prior_pose = self.get_prior_pose(image.name)
            if prior_pose is None:
                continue
            if not image.has_pose:
                continue
            src_centers.append(self._camera_center_from_pose(prior_pose))
            dst_centers.append(self._camera_center_from_pose(image.cam_from_world))
        if len(src_centers) < min_shared:
            return False
        src_pts = np.stack(src_centers, axis=0) 
        dst_pts = np.stack(dst_centers, axis=0) 
        sim3 = self._estimate_sim3(src_pts, dst_pts)
        if sim3 is None:
            return False
        scale, R, t = sim3
        self.mpsfm_rec.world_sim_scale = scale
        self.mpsfm_rec.world_sim_rotation = R
        self.mpsfm_rec.world_sim_translation = t
        return True

    def _normalized_keypoints(self, camera, keypoints):
        if len(keypoints) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        pts = np.asarray(keypoints, dtype=np.float64)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])
        K = np.asarray(camera.calibration_matrix(), dtype=np.float64)
        K_inv = np.linalg.inv(K)
        rays = (K_inv @ pts_h.T).T
        return rays

    @staticmethod
    def _relative_pose_components(T_c1w: pycolmap.Rigid3d, T_c2w: pycolmap.Rigid3d):
        R1 = T_c1w.rotation.matrix()
        t1 = np.array(T_c1w.translation, dtype=np.float64)
        R2 = T_c2w.rotation.matrix()
        t2 = np.array(T_c2w.translation, dtype=np.float64)
        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1
        return R_rel, t_rel

    @staticmethod
    def _compose_left_se3_delta(R_rel, t_rel, delta):
        rot_update = Rotation.from_rotvec(delta[:3]).as_matrix()
        trans_update = delta[3:]
        R_new = rot_update @ R_rel
        t_new = rot_update @ t_rel + trans_update
        return R_new, t_new

    def _essential_from_rt(self, R_rel, t_rel):
        return self._skew(t_rel) @ R_rel

    def _epipolar_residuals(self, R_rel, t_rel, pts1, pts2):
        E = self._essential_from_rt(R_rel, t_rel)
        Ex1 = (E @ pts1.T).T
        Etx2 = (E.T @ pts2.T).T
        numerators = np.einsum("ij,ij->i", pts2, Ex1)
        denom = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2 + 1e-12
        return numerators / np.sqrt(denom)

    @staticmethod
    def _robust_residual_scale(residuals, floor):
        residuals = np.abs(np.asarray(residuals, dtype=np.float64))
        if residuals.size == 0:
            return max(float(floor), 1e-6)
        med = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - med)))
        return max(float(floor), 1.4826 * mad, 1e-6)

    @staticmethod
    def _cauchy_soft_weights(residuals, scale, floor=0.0):
        scale = max(float(scale), 1e-6)
        normalized = np.asarray(residuals, dtype=np.float64) / scale
        weights = 1.0 / (1.0 + normalized**2)
        if floor > 0:
            weights = np.clip(weights, floor, 1.0)
        return weights

    @staticmethod
    def _tangent_basis(unit_vec):
        unit_vec = np.asarray(unit_vec, dtype=np.float64)
        unit_vec = unit_vec / max(np.linalg.norm(unit_vec), 1e-12)
        anchor = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(unit_vec, anchor)) > 0.9:
            anchor = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        basis1 = np.cross(unit_vec, anchor)
        basis1 /= max(np.linalg.norm(basis1), 1e-12)
        basis2 = np.cross(unit_vec, basis1)
        basis2 /= max(np.linalg.norm(basis2), 1e-12)
        return basis1, basis2

    @staticmethod
    def _grid_cell_weights(points, all_points, rows, cols, power):
        if len(points) == 0:
            return np.zeros(0, dtype=np.float64)
        all_points = np.asarray(all_points, dtype=np.float64)
        points = np.asarray(points, dtype=np.float64)
        min_xy = np.min(all_points, axis=0)
        max_xy = np.max(all_points, axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)
        x_ids = np.clip(((points[:, 0] - min_xy[0]) / span[0] * cols).astype(int), 0, cols - 1)
        y_ids = np.clip(((points[:, 1] - min_xy[1]) / span[1] * rows).astype(int), 0, rows - 1)
        cell_ids = y_ids * cols + x_ids
        counts = np.bincount(cell_ids, minlength=rows * cols).astype(np.float64)
        return 1.0 / np.maximum(counts[cell_ids], 1.0) ** power

    def _refine_init_pair_pose_bootstrap(
        self, ref_imid, qry_imid, T_c1w, T_c2w, matches, kps1, kps2, camera1, camera2
    ):
        empty_out = {
            "rotation": None,
            "full": None,
            "stage_b_enabled": False,
        }
        if not self.conf.refine_init_prior_pose:
            return empty_out
        matches = np.asarray(matches)
        zero_mask = np.zeros(len(matches), dtype=bool)
        min_matches = self.conf.epipolar_refine_min_matches
        if len(matches) < min_matches:
            return empty_out

        pts1 = self._normalized_keypoints(camera1, kps1[matches[:, 0]])
        pts2 = self._normalized_keypoints(camera2, kps2[matches[:, 1]])
        if len(pts1) < min_matches:
            return empty_out

        R_rel, t_rel = self._relative_pose_components(T_c1w, T_c2w)
        baseline_norm = float(np.linalg.norm(t_rel))
        if baseline_norm < 1e-9:
            return empty_out
        t_rel_unit = self._normalize_translation(t_rel)

        epipolar_scale = max(float(getattr(self.conf, "epipolar_refine_f_scale", 1e-2)), 1e-6)
        soft_floor = max(float(getattr(self.conf, "init_soft_filter_min_weight", 0.05)), 0.0)
        irls_iters = max(int(getattr(self.conf, "init_refine_irls_iters", 3)), 1)
        rot_reg_weight = max(float(getattr(self.conf, "init_refine_rotation_reg_weight", 5e-3)), 0.0)
        trans_reg_weight = max(float(getattr(self.conf, "init_refine_translation_reg_weight", 2e-3)), 0.0)
        stage_a_max_rot = np.deg2rad(max(float(getattr(self.conf, "init_refine_stage_a_max_rot_deg", 3.0)), 0.1))
        stage_b_max_trans = np.deg2rad(max(float(getattr(self.conf, "init_refine_stage_b_max_trans_deg", 10.0)), 0.1))
        core_weight_min = max(float(getattr(self.conf, "init_core_min_weight", 0.15)), 0.0)
        core_res_mult = max(float(getattr(self.conf, "init_core_inlier_residual_mult", 2.5)), 1.0)
        core_min_matches = max(int(getattr(self.conf, "init_core_min_matches", 12)), 3)
        core_min_tri_angle = float(getattr(self.conf, "init_core_min_tri_angle", 0.1))
        support_weight_min = max(float(getattr(self.conf, "init_support_min_weight", 0.05)), 0.0)
        support_res_mult = max(float(getattr(self.conf, "init_support_inlier_residual_mult", 4.0)), core_res_mult)
        support_min_matches = max(int(getattr(self.conf, "init_support_min_matches", 24)), core_min_matches)
        support_min_tri_angle = float(getattr(self.conf, "init_support_min_tri_angle", 0.03))
        reproj_min_points = max(int(getattr(self.conf, "init_reproj_refine_min_points", 8)), 3)
        reproj_thresh = max(float(getattr(self.conf, "init_reproj_refine_thresh", 4.0)), 1e-6)
        reproj_iters = max(int(getattr(self.conf, "init_reproj_refine_max_iters", 60)), 1)
        reproj_rot_reg = max(float(getattr(self.conf, "init_reproj_refine_rot_reg_weight", 1e-3)), 0.0)
        reproj_trans_reg = max(float(getattr(self.conf, "init_reproj_refine_trans_reg_weight", 1e-3)), 0.0)
        reproj_max_rot = np.deg2rad(max(float(getattr(self.conf, "init_reproj_refine_max_rot_deg", 1.5)), 0.1))
        reproj_max_trans = np.deg2rad(max(float(getattr(self.conf, "init_reproj_refine_max_trans_deg", 3.0)), 0.1))
        joint_max_rot_change_deg = max(float(getattr(self.conf, "init_joint_refine_max_rot_change_deg", 4.0)), 0.0)
        joint_max_trans_change_deg = max(
            float(getattr(self.conf, "init_joint_refine_max_trans_change_deg", 6.0)), 0.0
        )
        joint_max_scale_change_ratio = max(
            float(getattr(self.conf, "init_joint_refine_max_scale_change_ratio", 1.5)), 1.0
        )
        require_improvement = bool(getattr(self.conf, "init_joint_refine_require_improvement", True))
        stageb_min_support_count = max(
            int(getattr(self.conf, "init_stageb_min_support_count", 32)),
            support_min_matches,
        )
        stageb_min_support_ratio = max(float(getattr(self.conf, "init_stageb_min_support_ratio", 0.6)), 0.0)
        stageb_min_support_tri_median = max(
            float(getattr(self.conf, "init_stageb_min_support_tri_median_deg", 0.8)),
            support_min_tri_angle,
        )
        stageb_min_support_tri_p25 = max(
            float(getattr(self.conf, "init_stageb_min_support_tri_p25_deg", 0.3)),
            0.0,
        )
        use_spatial_reweight = bool(getattr(self.conf, "init_spatial_reweight", True))
        spatial_rows = max(int(getattr(self.conf, "init_spatial_grid_rows", 4)), 1)
        spatial_cols = max(int(getattr(self.conf, "init_spatial_grid_cols", 4)), 1)
        spatial_power = max(float(getattr(self.conf, "init_spatial_reweight_power", 0.5)), 0.0)
        spatial_min = max(float(getattr(self.conf, "init_spatial_reweight_min", 0.35)), 0.0)
        spatial_max = max(float(getattr(self.conf, "init_spatial_reweight_max", 2.5)), 1.0)

        def _compose_relative(R_base, t_base_unit, rot_state, dir_state=None):
            R_curr = Rotation.from_rotvec(rot_state).as_matrix() @ R_base
            if dir_state is None:
                return R_curr, t_base_unit.copy()
            basis1, basis2 = self._tangent_basis(t_base_unit)
            t_curr = t_base_unit + dir_state[0] * basis1 + dir_state[1] * basis2
            t_curr = self._normalize_translation(t_curr)
            if np.linalg.norm(t_curr) < 1e-9:
                t_curr = t_base_unit.copy()
            return R_curr, t_curr

        def _relative_to_absolute(R_curr, t_curr_unit):
            refined_translation = t_curr_unit * baseline_norm
            refined_relative = pycolmap.Rigid3d(pycolmap.Rotation3d(R_curr), refined_translation)
            return refined_relative * T_c1w

        def _estimate_scale_for_pose(T_qry_cw, match_mask):
            if match_mask is None:
                return 1.0, 0
            match_mask = np.asarray(match_mask, dtype=bool)
            if match_mask.size != len(matches) or int(np.count_nonzero(match_mask)) < 3:
                return 1.0, 0
            candidate_points = self._candidate_points3D_for_init(
                T_c1w,
                T_qry_cw,
                matches[match_mask],
                self.mpsfm_rec.images[ref_imid],
                self.mpsfm_rec.images[qry_imid],
                camera1,
                camera2,
            )
            scale_value = self._estimate_init_rescale(ref_imid, kps1, T_c1w, candidate_points)
            return float(scale_value), int(len(candidate_points.get("xyz", [])))

        def _run_stage_a(weights):
            sqrt_w = np.sqrt(np.asarray(weights, dtype=np.float64))

            def residual(rot_state):
                R_curr, t_curr_unit = _compose_relative(R_rel, t_rel_unit, rot_state)
                epi_raw = self._epipolar_residuals(R_curr, t_curr_unit, pts1, pts2)
                blocks = [sqrt_w * (epi_raw / epipolar_scale)]
                if rot_reg_weight > 0:
                    blocks.append(np.sqrt(rot_reg_weight) * rot_state)
                return np.concatenate(blocks)

            bounds = (
                np.full(3, -stage_a_max_rot, dtype=np.float64),
                np.full(3, stage_a_max_rot, dtype=np.float64),
            )
            return least_squares(
                residual,
                np.zeros(3, dtype=np.float64),
                method="trf",
                loss="linear",
                bounds=bounds,
                max_nfev=self.conf.epipolar_refine_max_iters,
            )

        def _build_core_support_masks(R_curr, t_curr_unit, weights):
            epi_curr = self._epipolar_residuals(R_curr, t_curr_unit, pts1, pts2)
            robust_scale = self._robust_residual_scale(epi_curr, epipolar_scale)
            core_thresh = core_res_mult * robust_scale
            support_thresh = support_res_mult * robust_scale
            prelim_support = (np.abs(epi_curr) <= support_thresh) & (weights >= support_weight_min)
            if np.count_nonzero(prelim_support) < support_min_matches:
                stats = {
                    "core_count": 0,
                    "support_count": 0,
                    "support_tri_mean": 0.0,
                    "support_tri_median": 0.0,
                    "support_tri_p25": 0.0,
                    "support_ratio": 0.0,
                    "epi_rms": float(np.sqrt(np.mean(epi_curr**2))) if len(epi_curr) else 0.0,
                    "robust_scale": robust_scale,
                }
                return (
                    np.zeros(len(matches), dtype=bool),
                    np.zeros(len(matches), dtype=bool),
                    epi_curr,
                    robust_scale,
                    stats,
                )

            prelim_matches = matches[prelim_support]
            candidate_points = self._candidate_points3D_for_init(
                T_c1w,
                _relative_to_absolute(R_curr, t_curr_unit),
                prelim_matches,
                self.mpsfm_rec.images[ref_imid],
                self.mpsfm_rec.images[qry_imid],
                camera1,
                camera2,
            )
            pair_geom = {}
            tri_angles = candidate_points.get("tri_angle", [])
            posdepth1 = candidate_points.get("posdepth1", [])
            posdepth2 = candidate_points.get("posdepth2", [])
            pair_ids = zip(candidate_points.get("pt2d_id_1", []), candidate_points.get("pt2d_id_2", []))
            for pair, tri_angle, pd1, pd2 in zip(pair_ids, tri_angles, posdepth1, posdepth2):
                pair_key = (int(pair[0]), int(pair[1]))
                if pd1 and pd2:
                    prev = pair_geom.get(pair_key, None)
                    if (prev is None) or (tri_angle > prev):
                        pair_geom[pair_key] = float(tri_angle)

            core_mask = np.zeros(len(matches), dtype=bool)
            support_mask = np.zeros(len(matches), dtype=bool)
            support_tri = []
            if not pair_geom:
                stats = {
                    "core_count": 0,
                    "support_count": 0,
                    "support_tri_mean": 0.0,
                    "support_tri_median": 0.0,
                    "support_tri_p25": 0.0,
                    "support_ratio": 0.0,
                    "epi_rms": float(np.sqrt(np.mean(epi_curr**2))) if len(epi_curr) else 0.0,
                    "robust_scale": robust_scale,
                }
                return core_mask, support_mask, epi_curr, robust_scale, stats
            for idx, pair in enumerate(matches):
                pair_key = (int(pair[0]), int(pair[1]))
                tri_angle = pair_geom.get(pair_key)
                if tri_angle is None:
                    continue
                if prelim_support[idx] and tri_angle >= support_min_tri_angle:
                    support_mask[idx] = True
                    support_tri.append(tri_angle)
                    if (
                        abs(epi_curr[idx]) <= core_thresh
                        and weights[idx] >= core_weight_min
                        and tri_angle >= core_min_tri_angle
                    ):
                        core_mask[idx] = True

            support_count = int(np.count_nonzero(support_mask))
            core_count = int(np.count_nonzero(core_mask))
            support_tri = np.asarray(support_tri, dtype=np.float64)
            stats = {
                "core_count": core_count,
                "support_count": support_count,
                "support_tri_mean": float(np.mean(support_tri)) if len(support_tri) else 0.0,
                "support_tri_median": float(np.median(support_tri)) if len(support_tri) else 0.0,
                "support_tri_p25": float(np.percentile(support_tri, 25)) if len(support_tri) else 0.0,
                "support_ratio": float(support_count / max(int(np.count_nonzero(prelim_support)), 1)),
                "epi_rms": float(np.sqrt(np.mean(epi_curr**2))) if len(epi_curr) else 0.0,
                "robust_scale": robust_scale,
            }
            return core_mask, support_mask, epi_curr, robust_scale, stats

        def _score_direction(mask_stats, state_vec, bound_t):
            score = (
                4.0 * mask_stats["support_count"]
                + 2.0 * mask_stats["core_count"]
                + 6.0 * mask_stats["support_ratio"]
                + 0.25 * mask_stats["support_tri_median"]
                + 0.1 * mask_stats["support_tri_mean"]
                - 0.5 * (mask_stats["epi_rms"] / max(mask_stats["robust_scale"], 1e-6))
            )
            if trans_reg_weight > 0 and bound_t > 1e-9:
                normalized_state = float(np.linalg.norm(state_vec) / bound_t)
                score -= np.sqrt(trans_reg_weight) * float(len(matches)) * normalized_state
            return score

        def _run_stage_b(R_stage):
            bound_t = np.tan(stage_b_max_trans)
            best_state = np.zeros(2, dtype=np.float64)
            current_weights = prior_weights.copy()
            best_pack = None

            for _ in range(irls_iters):
                search_state = best_state.copy()
                search_radii = [bound_t, 0.5 * bound_t, 0.25 * bound_t]
                for radius in search_radii:
                    candidates = []
                    for dx in (-radius, 0.0, radius):
                        for dy in (-radius, 0.0, radius):
                            cand = np.clip(search_state + np.array([dx, dy], dtype=np.float64), -bound_t, bound_t)
                            candidates.append(cand)
                    # remove duplicates while keeping order
                    unique_candidates = []
                    seen = set()
                    for cand in candidates:
                        key = tuple(np.round(cand, decimals=8))
                        if key in seen:
                            continue
                        seen.add(key)
                        unique_candidates.append(cand)

                    local_best = None
                    local_best_score = -np.inf
                    for cand in unique_candidates:
                        R_curr, t_curr_unit = _compose_relative(R_stage, t_rel_unit, np.zeros(3), cand)
                        core_mask, support_mask, epi_curr, robust_scale, mask_stats = _build_core_support_masks(
                            R_curr, t_curr_unit, current_weights
                        )
                        score = _score_direction(mask_stats, cand, bound_t)
                        if score > local_best_score:
                            local_best_score = score
                            local_best = {
                                "state": cand,
                                "R": R_curr,
                                "t_unit": t_curr_unit,
                                "core_mask": core_mask,
                                "support_mask": support_mask,
                                "epi": epi_curr,
                                "scale": robust_scale,
                                "stats": mask_stats,
                                "score": float(score),
                            }
                    if local_best is not None:
                        search_state = local_best["state"]
                        best_pack = local_best

                if best_pack is None:
                    break
                best_state = best_pack["state"]
                current_weights = prior_weights * self._cauchy_soft_weights(
                    best_pack["epi"], best_pack["scale"], floor=soft_floor
                )

            return best_pack, current_weights

        def _small_reprojection_refine(R_base, t_base_unit, core_mask, weights):
            core_matches = matches[core_mask]
            if len(core_matches) < reproj_min_points:
                return R_base, t_base_unit, core_mask, {
                    "num_seed_points": 0,
                    "reproj_before_rms": 0.0,
                    "reproj_after_rms": 0.0,
                    "reproj_inliers_before": 0,
                    "reproj_inliers_after": 0,
                }
            candidate_points = self._candidate_points3D_for_init(
                T_c1w,
                _relative_to_absolute(R_base, t_base_unit),
                core_matches,
                self.mpsfm_rec.images[ref_imid],
                self.mpsfm_rec.images[qry_imid],
                camera1,
                camera2,
            )
            if len(candidate_points.get("xyz", [])) < reproj_min_points:
                return R_base, t_base_unit, {
                    "num_seed_points": int(len(candidate_points.get("xyz", []))),
                    "reproj_before_rms": 0.0,
                    "reproj_after_rms": 0.0,
                    "reproj_inliers_before": 0,
                    "reproj_inliers_after": 0,
                }

            points3D = np.asarray(candidate_points["xyz"], dtype=np.float64)
            pts2d = kps2[np.asarray(candidate_points["pt2d_id_2"], dtype=int)]
            pair_weights = []
            pair_to_weight = {
                (int(m[0]), int(m[1])): float(w)
                for m, w in zip(matches[core_mask], np.asarray(weights[core_mask], dtype=np.float64))
            }
            for pair in zip(candidate_points["pt2d_id_1"], candidate_points["pt2d_id_2"]):
                pair_weights.append(pair_to_weight.get((int(pair[0]), int(pair[1])), 1.0))
            pair_weights = np.sqrt(np.asarray(pair_weights, dtype=np.float64))
            bound_t = np.tan(reproj_max_trans)
            lower = np.array(
                [-reproj_max_rot, -reproj_max_rot, -reproj_max_rot, -bound_t, -bound_t],
                dtype=np.float64,
            )
            upper = -lower

            def reproj_errors(R_curr, t_curr_unit):
                T_curr = _relative_to_absolute(R_curr, t_curr_unit)
                xyz_cam = T_curr * points3D
                residual = np.full((len(points3D), 2), reproj_thresh * 5.0, dtype=np.float64)
                valid = xyz_cam[:, 2] > 1e-9
                if np.any(valid):
                    proj = camera2.img_from_cam(xyz_cam[valid])
                    residual[valid] = proj - pts2d[valid]
                return residual, valid

            e_before, valid_before = reproj_errors(R_base, t_base_unit)
            norm_before = np.linalg.norm(e_before, axis=1)

            def residual(state_vec):
                rot_state = state_vec[:3]
                dir_state = state_vec[3:]
                R_curr, t_curr_unit = _compose_relative(R_base, t_base_unit, rot_state, dir_state)
                e2d, _ = reproj_errors(R_curr, t_curr_unit)
                blocks = [pair_weights.repeat(2) * (e2d.reshape(-1) / reproj_thresh)]
                if reproj_rot_reg > 0:
                    blocks.append(np.sqrt(reproj_rot_reg) * rot_state)
                if reproj_trans_reg > 0:
                    blocks.append(np.sqrt(reproj_trans_reg) * dir_state)
                return np.concatenate(blocks)

            result = least_squares(
                residual,
                np.zeros(5, dtype=np.float64),
                method="trf",
                loss="soft_l1",
                f_scale=1.0,
                bounds=(lower, upper),
                max_nfev=reproj_iters,
            )
            if not result.success:
                return R_base, t_base_unit, {
                    "num_seed_points": int(len(points3D)),
                    "reproj_before_rms": float(np.sqrt(np.mean(norm_before**2))) if len(norm_before) else 0.0,
                    "reproj_after_rms": float(np.sqrt(np.mean(norm_before**2))) if len(norm_before) else 0.0,
                    "reproj_inliers_before": int(np.count_nonzero(valid_before & (norm_before <= reproj_thresh))),
                    "reproj_inliers_after": int(np.count_nonzero(valid_before & (norm_before <= reproj_thresh))),
                }

            R_final, t_final_unit = _compose_relative(R_base, t_base_unit, result.x[:3], result.x[3:])
            e_after, valid_after = reproj_errors(R_final, t_final_unit)
            norm_after = np.linalg.norm(e_after, axis=1)
            stats = {
                "num_seed_points": int(len(points3D)),
                "reproj_before_rms": float(np.sqrt(np.mean(norm_before**2))) if len(norm_before) else 0.0,
                "reproj_after_rms": float(np.sqrt(np.mean(norm_after**2))) if len(norm_after) else 0.0,
                "reproj_inliers_before": int(np.count_nonzero(valid_before & (norm_before <= reproj_thresh))),
                "reproj_inliers_after": int(np.count_nonzero(valid_after & (norm_after <= reproj_thresh))),
            }
            return R_final, t_final_unit, core_mask, stats

        prior_epi = self._epipolar_residuals(R_rel, t_rel_unit, pts1, pts2)
        prior_scale = self._robust_residual_scale(prior_epi, epipolar_scale)
        prior_weights = self._cauchy_soft_weights(prior_epi, prior_scale, floor=soft_floor)
        if use_spatial_reweight:
            ref_match_pts = kps1[matches[:, 0]]
            qry_match_pts = kps2[matches[:, 1]]
            w_ref = self._grid_cell_weights(ref_match_pts, kps1, spatial_rows, spatial_cols, spatial_power)
            w_qry = self._grid_cell_weights(qry_match_pts, kps2, spatial_rows, spatial_cols, spatial_power)
            spatial_weights = np.sqrt(w_ref * w_qry)
            if spatial_weights.size > 0:
                spatial_weights /= max(float(np.mean(spatial_weights)), 1e-6)
                spatial_weights = np.clip(spatial_weights, spatial_min, spatial_max)
                prior_weights = prior_weights * spatial_weights
        prior_weights = np.clip(prior_weights, soft_floor, None)

        def _trans_angle_deg(base_t, cand_t):
            return float(
                np.rad2deg(
                    np.arccos(
                        np.clip(
                            np.dot(base_t, cand_t) / (np.linalg.norm(base_t) * np.linalg.norm(cand_t)),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )

        def _make_hypothesis(stage_name, pose, R_curr, t_curr_unit, core_mask, support_mask, epi_curr, mask_stats, extra):
            rot_delta = R_curr @ R_rel.T
            stats = {
                "stage": stage_name,
                "num_matches": int(len(matches)),
                "baseline_norm": baseline_norm,
                "soft_filter_scale": prior_scale,
                "soft_filter_mean_weight": float(np.mean(prior_weights)),
                "irls_iters": irls_iters,
                "rot_diff_deg": float(np.rad2deg(Rotation.from_matrix(rot_delta).magnitude())),
                "trans_angle_deg": _trans_angle_deg(t_rel_unit, t_curr_unit),
                "residual_before_rms": float(np.sqrt(np.mean(prior_epi**2))),
                "residual_after_rms": float(np.sqrt(np.mean(epi_curr**2))) if len(epi_curr) else 0.0,
                "core_inliers": int(np.count_nonzero(core_mask)),
                "support_inliers": int(np.count_nonzero(support_mask)),
                "support_tri_mean": float(mask_stats["support_tri_mean"]),
                "support_tri_median": float(mask_stats["support_tri_median"]),
                "support_tri_p25": float(mask_stats["support_tri_p25"]),
                "support_ratio": float(mask_stats["support_ratio"]),
                "support_residual_scale": float(mask_stats["robust_scale"]),
                "spatial_reweight_enabled": bool(use_spatial_reweight),
                "used_depth": False,
            }
            stats.update(extra)
            return {
                "pose": pose,
                "core_mask": core_mask,
                "support_mask": support_mask,
                "stats": stats,
            }

        try:
            stage_a_result = _run_stage_a(prior_weights)
            if not stage_a_result.success:
                return empty_out
            R_stage_a, t_stage_a_unit = _compose_relative(R_rel, t_rel_unit, stage_a_result.x)
            stage_a_pose = _relative_to_absolute(R_stage_a, t_stage_a_unit)
            stage_a_core_mask, stage_a_support_mask, epi_stage_a, stage_a_scale, stage_a_mask_stats = (
                _build_core_support_masks(R_stage_a, t_stage_a_unit, prior_weights)
            )
        except Exception as exc:
            self.log(f"[InitPair] Bootstrap refine failed: {exc}", level=1)
            return empty_out

        rotation_hyp = _make_hypothesis(
            "rotation_only",
            stage_a_pose,
            R_stage_a,
            t_stage_a_unit,
            stage_a_core_mask,
            stage_a_support_mask,
            epi_stage_a,
            stage_a_mask_stats,
            {},
        )

        strong_geometry = (
            int(np.count_nonzero(stage_a_support_mask)) >= stageb_min_support_count
            and int(np.count_nonzero(stage_a_core_mask)) >= core_min_matches
            and float(stage_a_mask_stats["support_ratio"]) >= stageb_min_support_ratio
            and float(stage_a_mask_stats["support_tri_median"]) >= stageb_min_support_tri_median
            and float(stage_a_mask_stats["support_tri_p25"]) >= stageb_min_support_tri_p25
        )
        rotation_hyp["stats"]["stage_b_enabled"] = bool(strong_geometry)

        if not strong_geometry:
            return {
                "rotation": rotation_hyp,
                "full": None,
                "stage_b_enabled": False,
            }

        try:
            stage_b_pack, final_weights = _run_stage_b(R_stage_a)
            if stage_b_pack is None:
                return {
                    "rotation": rotation_hyp,
                    "full": None,
                    "stage_b_enabled": True,
                }
        except Exception as exc:
            self.log(f"[InitPair] Stage B refine failed: {exc}", level=1)
            return {
                "rotation": rotation_hyp,
                "full": None,
                "stage_b_enabled": True,
            }

        R_stage_b = stage_b_pack["R"]
        t_stage_b_unit = stage_b_pack["t_unit"]
        core_mask = stage_b_pack["core_mask"]
        support_mask = stage_b_pack["support_mask"]
        if int(np.count_nonzero(support_mask)) < support_min_matches or int(np.count_nonzero(core_mask)) < core_min_matches:
            return {
                "rotation": rotation_hyp,
                "full": None,
                "stage_b_enabled": True,
            }

        R_final, t_final_unit, _, reproj_stats = _small_reprojection_refine(R_stage_b, t_stage_b_unit, core_mask, final_weights)
        core_mask, support_mask, epi_final, residual_scale_final, final_mask_stats = _build_core_support_masks(
            R_final, t_final_unit, final_weights
        )
        if int(np.count_nonzero(support_mask)) < support_min_matches or int(np.count_nonzero(core_mask)) < core_min_matches:
            return {
                "rotation": rotation_hyp,
                "full": None,
                "stage_b_enabled": True,
            }

        refined_pose = _relative_to_absolute(R_final, t_final_unit)
        prior_scale_value, prior_scale_points = _estimate_scale_for_pose(T_c2w, support_mask)
        refined_scale_value, refined_scale_points = _estimate_scale_for_pose(refined_pose, support_mask)
        if (
            np.isfinite(prior_scale_value)
            and np.isfinite(refined_scale_value)
            and prior_scale_value > 1e-9
            and refined_scale_value > 1e-9
        ):
            scale_change_ratio = max(
                float(refined_scale_value / prior_scale_value),
                float(prior_scale_value / refined_scale_value),
            )
        else:
            scale_change_ratio = np.inf
        rot_diff_deg = float(np.rad2deg(Rotation.from_matrix(R_final @ R_rel.T).magnitude()))
        trans_angle = _trans_angle_deg(t_rel_unit, t_final_unit)
        rot_gate_ok = rot_diff_deg <= joint_max_rot_change_deg
        trans_gate_ok = trans_angle <= joint_max_trans_change_deg
        scale_gate_ok = scale_change_ratio <= joint_max_scale_change_ratio
        epi_gate_ok = True
        if require_improvement:
            epi_gate_ok = float(np.sqrt(np.mean(epi_final**2))) <= float(np.sqrt(np.mean(prior_epi**2))) + 1e-8

        full_hyp = None
        if rot_gate_ok and trans_gate_ok and scale_gate_ok and epi_gate_ok:
            full_hyp = _make_hypothesis(
                "full_refine",
                refined_pose,
                R_final,
                t_final_unit,
                core_mask,
                support_mask,
                epi_final,
                final_mask_stats,
                {
                    "stage_a_rot_diff_deg": float(np.rad2deg(Rotation.from_matrix(R_stage_a @ R_rel.T).magnitude())),
                    "stage_b_rot_diff_deg": float(np.rad2deg(Rotation.from_matrix(R_stage_b @ R_rel.T).magnitude())),
                    "support_score": float(stage_b_pack["score"]),
                    "prior_scale_value": float(prior_scale_value),
                    "refined_scale_value": float(refined_scale_value),
                    "scale_change_ratio": float(scale_change_ratio),
                    "scale_points_prior": int(prior_scale_points),
                    "scale_points_refined": int(refined_scale_points),
                    "rot_gate_ok": bool(rot_gate_ok),
                    "trans_gate_ok": bool(trans_gate_ok),
                    "scale_gate_ok": bool(scale_gate_ok),
                    "epi_gate_ok": bool(epi_gate_ok),
                    "reproj_num_seed_points": int(reproj_stats.get("num_seed_points", 0)),
                    "reproj_before_rms": float(reproj_stats.get("reproj_before_rms", 0.0)),
                    "reproj_after_rms": float(reproj_stats.get("reproj_after_rms", 0.0)),
                    "reproj_inliers_before": int(reproj_stats.get("reproj_inliers_before", 0)),
                    "reproj_inliers_after": int(reproj_stats.get("reproj_inliers_after", 0)),
                    "support_residual_scale": float(residual_scale_final),
                },
            )

        return {
            "rotation": rotation_hyp,
            "full": full_hyp,
            "stage_b_enabled": True,
        }

    @staticmethod
    def _candidate_points3D_for_init(
        cam_from_world1, cam_from_world2, matches, image1, image2, camera1, camera2, inliers=None
    ):
        candidate_points = defaultdict(list)
        if inliers is None:
            inliers = slice(None)
        for match in matches[inliers]:
            pt2d_id_1, pt2d_id_2 = match
            kp1_i = image1.points2D[pt2d_id_1].xy
            kp2_i = image2.points2D[pt2d_id_2].xy
            pointdata = np.array([kp1_i, kp2_i], dtype=np.float64)
            out = pycolmap.estimate_triangulation(pointdata, [cam_from_world1, cam_from_world2], [camera1, camera2])
            if out is None:
                continue

            projection_center1 = cam_from_world1.rotation.inverse() * -cam_from_world1.translation
            projection_center2 = cam_from_world2.rotation.inverse() * -cam_from_world2.translation
            tri_angle = calculate_triangulation_angle(projection_center1, projection_center2, out["xyz"])
            posdepth1 = has_point_positive_depth(cam_from_world1.matrix(), out["xyz"])
            posdepth2 = has_point_positive_depth(cam_from_world2.matrix(), out["xyz"])

            candidate_points["pt2d_id_1"].append(pt2d_id_1)
            candidate_points["pt2d_id_2"].append(pt2d_id_2)
            candidate_points["tri_angle"].append(np.rad2deg(tri_angle))
            candidate_points["posdepth1"].append(posdepth1)
            candidate_points["posdepth2"].append(posdepth2)
            candidate_points["xyz"].append(out["xyz"])
        return candidate_points

    def _estimate_init_rescale(self, ref_imid, kps_ref, T_ref_cw, candidate_points):
        if len(candidate_points.get("xyz", [])) == 0:
            return 1.0
        image_ref = self.mpsfm_rec.images[ref_imid]
        if image_ref.depth is None:
            return 1.0
        tri_world = np.vstack(candidate_points["xyz"])
        tri_cam = T_ref_cw * tri_world
        z = tri_cam[:, -1]
        mask = np.asarray(candidate_points["pt2d_id_1"], dtype=int)
        d = self.mpsfm_rec.images[ref_imid].depth.data_prior_at_kps(kps_ref[mask])
        valid = np.isfinite(d) & (d > 0) & np.isfinite(z) & (z > 0)
        if not np.any(valid):
            return 1.0
        return float(np.median(z[valid] / d[valid]))

    def _select_init_seed_mask(self, prior_mask, core_mask=None, support_mask=None, min_matches=3):
        prior_mask = np.asarray(prior_mask, dtype=bool)
        candidates = [("prior", prior_mask)]
        if support_mask is not None:
            support_mask = np.asarray(support_mask, dtype=bool)
            candidates.insert(0, ("prior_support", prior_mask & support_mask))
        if core_mask is not None:
            core_mask = np.asarray(core_mask, dtype=bool)
            candidates.insert(0, ("prior_core", prior_mask & core_mask))

        seen = set()
        for label, mask in candidates:
            key = mask.tobytes()
            if key in seen:
                continue
            seen.add(key)
            if int(np.count_nonzero(mask)) >= min_matches:
                return mask, label
        return None, "invalid"

    def _init_pair_prior_inlier_mask_from_pose(
        self, ref_imid, qry_imid, T_c1w, T_c2w, matches, kps1, kps2, camera1, camera2
    ):
        matches = np.asarray(matches)
        if len(matches) == 0:
            return np.zeros(0, dtype=bool)

        pts1 = self._normalized_keypoints(camera1, kps1[matches[:, 0]])
        pts2 = self._normalized_keypoints(camera2, kps2[matches[:, 1]])
        if len(pts1) == 0:
            return np.zeros(len(matches), dtype=bool)

        R_rel, t_rel = self._relative_pose_components(T_c1w, T_c2w)
        if float(np.linalg.norm(t_rel)) < 1e-9:
            return np.zeros(len(matches), dtype=bool)
        t_rel_unit = self._normalize_translation(t_rel)

        epipolar_scale = max(float(getattr(self.conf, "epipolar_refine_f_scale", 1e-2)), 1e-6)
        support_res_mult = max(float(getattr(self.conf, "init_support_inlier_residual_mult", 4.0)), 1.0)
        support_min_tri_angle = float(getattr(self.conf, "init_support_min_tri_angle", 0.03))

        epi = self._epipolar_residuals(R_rel, t_rel_unit, pts1, pts2)
        robust_scale = self._robust_residual_scale(epi, epipolar_scale)
        support_thresh = support_res_mult * robust_scale

        candidate_points = self._candidate_points3D_for_init(
            T_c1w,
            T_c2w,
            matches,
            self.mpsfm_rec.images[ref_imid],
            self.mpsfm_rec.images[qry_imid],
            camera1,
            camera2,
        )
        valid_pairs = set()
        for pt1, pt2, tri_angle, posdepth1, posdepth2 in zip(
            candidate_points.get("pt2d_id_1", []),
            candidate_points.get("pt2d_id_2", []),
            candidate_points.get("tri_angle", []),
            candidate_points.get("posdepth1", []),
            candidate_points.get("posdepth2", []),
        ):
            if tri_angle >= support_min_tri_angle and posdepth1 and posdepth2:
                valid_pairs.add((int(pt1), int(pt2)))

        mask = np.zeros(len(matches), dtype=bool)
        for idx, pair in enumerate(matches):
            pair_key = (int(pair[0]), int(pair[1]))
            if pair_key in valid_pairs and abs(epi[idx]) <= support_thresh:
                mask[idx] = True
        return mask

    def _build_init_seed_data(self, imid1, imid2, T_c1w, T_c2w, matches_used, camera1, camera2):
        candidate_points = self._candidate_points3D_for_init(
            T_c1w,
            T_c2w,
            matches_used,
            self.mpsfm_rec.images[imid1],
            self.mpsfm_rec.images[imid2],
            camera1,
            camera2,
        )
        min_tri_angle = float(self.conf.colmap_options.init_min_tri_angle)
        seed_entries = []
        ref_maps = {
            imid1: defaultdict(list),
            imid2: defaultdict(list),
        }

        for pt1, pt2, tri_angle, posdepth1, posdepth2, xyz in zip(
            candidate_points.get("pt2d_id_1", []),
            candidate_points.get("pt2d_id_2", []),
            candidate_points.get("tri_angle", []),
            candidate_points.get("posdepth1", []),
            candidate_points.get("posdepth2", []),
            candidate_points.get("xyz", []),
        ):
            if tri_angle < min_tri_angle or (not posdepth1) or (not posdepth2):
                continue
            seed_idx = len(seed_entries)
            seed_entries.append(
                {
                    "pt2d_id_1": int(pt1),
                    "pt2d_id_2": int(pt2),
                    "tri_angle": float(tri_angle),
                    "xyz": np.asarray(xyz, dtype=np.float64),
                }
            )
            ref_maps[imid1][int(pt1)].append(seed_idx)
            ref_maps[imid2][int(pt2)].append(seed_idx)

        return {
            "entries": seed_entries,
            "ref_maps": ref_maps,
            "num_seed_points": int(len(seed_entries)),
        }

    def _select_init_lookahead_images(self, imid1, imid2):
        topk = max(int(getattr(self.conf, "init_lookahead_topk", 5)), 1)
        name1 = self.mpsfm_rec.images[imid1].name
        name2 = self.mpsfm_rec.images[imid2].name
        ranked = []

        for cand_imid, image in self.mpsfm_rec.images.items():
            if cand_imid in {imid1, imid2}:
                continue
            corr_total = int(self.correspondences.num_correspondences_between_images(imid1, cand_imid)) + int(
                self.correspondences.num_correspondences_between_images(imid2, cand_imid)
            )
            if corr_total == 0:
                continue
            pair_score = float(
                self.correspondences.inlier_match_scores.get(
                    frozenset([name1, image.name]),
                    0.0,
                )
                + self.correspondences.inlier_match_scores.get(
                    frozenset([name2, image.name]),
                    0.0,
                )
            )
            ranked.append((pair_score, corr_total, cand_imid))

        ranked.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
        return [cand_imid for _, _, cand_imid in ranked[:topk]]

    def _evaluate_init_hypothesis_lookahead(self, imid1, imid2, seed_data, lookahead_imids):
        if seed_data["num_seed_points"] < 3 or len(lookahead_imids) == 0:
            return {
                "num_evaluated": int(len(lookahead_imids)),
                "num_registerable": 0,
                "median_inliers": 0.0,
                "median_inlier_ratio": 0.0,
                "mean_reproj_rmse": float("inf"),
                "score": -float(seed_data["num_seed_points"]),
            }

        seed_xyz = np.asarray([entry["xyz"] for entry in seed_data["entries"]], dtype=np.float64)
        ap_min_num_inliers = int(self.conf.colmap_options.abs_pose_min_num_inliers)
        eval_stats = []

        for qry_imid in lookahead_imids:
            votes = defaultdict(lambda: defaultdict(int))
            for ref_imid in (imid1, imid2):
                corr = self.correspondences.matches(ref_imid, qry_imid)
                if len(corr) == 0:
                    continue
                for ref_pt, qry_pt in corr:
                    for seed_idx in seed_data["ref_maps"][ref_imid].get(int(ref_pt), []):
                        votes[int(qry_pt)][seed_idx] += 1

            if not votes:
                eval_stats.append({"registerable": False, "inliers": 0, "inlier_ratio": 0.0, "reproj_rmse": float("inf")})
                continue

            kps_qry = self.mpsfm_rec.keypoints(qry_imid)
            camera_qry = self.mpsfm_rec.camera(qry_imid)
            points2D = []
            points3D = []
            for qry_pt, seed_votes in votes.items():
                best_seed_idx = max(seed_votes.items(), key=lambda item: (item[1], -item[0]))[0]
                points2D.append(kps_qry[qry_pt])
                points3D.append(seed_xyz[best_seed_idx])

            points2D = np.asarray(points2D, dtype=np.float64)
            points3D = np.asarray(points3D, dtype=np.float64)
            if len(points2D) < 4:
                eval_stats.append({"registerable": False, "inliers": 0, "inlier_ratio": 0.0, "reproj_rmse": float("inf")})
                continue

            ap_info = self.absolute_pose_estimator(points2D, points3D, camera_qry)
            if ap_info is None:
                eval_stats.append({"registerable": False, "inliers": 0, "inlier_ratio": 0.0, "reproj_rmse": float("inf")})
                continue

            inlier_mask = np.asarray(ap_info["inlier_mask"], dtype=bool)
            num_inliers = int(ap_info["num_inliers"])
            registerable = num_inliers >= ap_min_num_inliers
            reproj_rmse = float("inf")
            if np.any(inlier_mask):
                xyz_cam = ap_info["cam_from_world"] * points3D[inlier_mask]
                valid = xyz_cam[:, 2] > 1e-9
                if np.any(valid):
                    proj = camera_qry.img_from_cam(xyz_cam[valid])
                    xy_target = points2D[inlier_mask][valid]
                    reproj_rmse = float(np.sqrt(np.mean(np.sum((proj - xy_target) ** 2, axis=1))))

            eval_stats.append(
                {
                    "registerable": bool(registerable),
                    "inliers": num_inliers,
                    "inlier_ratio": float(num_inliers / max(len(points2D), 1)),
                    "reproj_rmse": reproj_rmse,
                }
            )

        reg_stats = [stat for stat in eval_stats if stat["registerable"]]
        if reg_stats:
            median_inliers = float(np.median([stat["inliers"] for stat in reg_stats]))
            median_inlier_ratio = float(np.median([stat["inlier_ratio"] for stat in reg_stats]))
            mean_reproj_rmse = float(np.mean([stat["reproj_rmse"] for stat in reg_stats]))
        else:
            median_inliers = 0.0
            median_inlier_ratio = 0.0
            mean_reproj_rmse = float("inf")

        num_registerable = int(sum(stat["registerable"] for stat in eval_stats))
        score = (
            100.0 * num_registerable
            + 1.0 * median_inliers
            + 20.0 * median_inlier_ratio
            - 0.5 * (mean_reproj_rmse if np.isfinite(mean_reproj_rmse) else 1e3)
            + 0.05 * float(seed_data["num_seed_points"])
        )
        return {
            "num_evaluated": int(len(lookahead_imids)),
            "num_registerable": num_registerable,
            "median_inliers": median_inliers,
            "median_inlier_ratio": median_inlier_ratio,
            "mean_reproj_rmse": mean_reproj_rmse,
            "score": float(score),
        }

    def _is_refined_init_hypothesis_better(self, refined_stats, prior_stats):
        min_inlier_gain = float(getattr(self.conf, "init_lookahead_min_inlier_gain", 6))
        min_reproj_gain = float(getattr(self.conf, "init_lookahead_min_reproj_gain_px", 0.25))
        min_score_gain = float(getattr(self.conf, "init_lookahead_min_score_gain", 2.0))

        if refined_stats["num_registerable"] > prior_stats["num_registerable"]:
            return True
        if refined_stats["num_registerable"] < prior_stats["num_registerable"]:
            return False
        if refined_stats["median_inliers"] >= prior_stats["median_inliers"] + min_inlier_gain:
            return True
        if (
            np.isfinite(refined_stats["mean_reproj_rmse"])
            and np.isfinite(prior_stats["mean_reproj_rmse"])
            and refined_stats["mean_reproj_rmse"] <= prior_stats["mean_reproj_rmse"] - min_reproj_gain
            and refined_stats["median_inliers"] >= prior_stats["median_inliers"]
        ):
            return True
        return refined_stats["score"] >= prior_stats["score"] + min_score_gain

    def _make_init_hypothesis(
        self,
        name,
        pose,
        prior_mask,
        matches,
        imid1,
        imid2,
        T_c1w,
        kps1,
        camera1,
        camera2,
        core_mask=None,
        support_mask=None,
        extra_stats=None,
    ):
        seed_mask, seed_source = self._select_init_seed_mask(prior_mask, core_mask=core_mask, support_mask=support_mask)
        if seed_mask is None:
            return None
        matches_used = matches[seed_mask]
        seed_data = self._build_init_seed_data(imid1, imid2, T_c1w, pose, matches_used, camera1, camera2)
        if seed_data["num_seed_points"] < 3:
            return None
        seed_tri_points = self._candidate_points3D_for_init(
            T_c1w,
            pose,
            matches_used,
            self.mpsfm_rec.images[imid1],
            self.mpsfm_rec.images[imid2],
            camera1,
            camera2,
        )
        return {
            "name": name,
            "pose": pose,
            "seed_mask": seed_mask,
            "seed_source": seed_source,
            "matches_used": matches_used,
            "rescale": self._estimate_init_rescale(imid1, kps1, T_c1w, seed_tri_points),
            "seed_data": seed_data,
            "stats": extra_stats or {},
        }

    def _merge_candidate_points(self, points_lifted, points_triangulated):
        candidate_points = {}
        ids_lift_1 = points_lifted.get("pt2d_id_1", [])
        ids_lift_2 = points_lifted.get("pt2d_id_2", [])
        ids_tri_1 = points_triangulated.get("pt2d_id_1", [])
        ids_tri_2 = points_triangulated.get("pt2d_id_2", [])

        keys = list(points_lifted.keys()) if len(points_lifted) else list(points_triangulated.keys())
        if not keys:
            keys = ["pt2d_id_1", "pt2d_id_2", "tri_angle", "posdepth1", "posdepth2", "xyz"]
        for key in keys:
            candidate_points[key] = []

        tri_map = defaultdict(list)
        for idx, pair in enumerate(zip(ids_tri_1, ids_tri_2)):
            tri_map[pair].append(idx)

        for idx_lift, pair in enumerate(zip(ids_lift_1, ids_lift_2)):
            if pair in tri_map and tri_map[pair]:
                idx_tri = tri_map[pair].pop(0)
                use_lift = points_triangulated["tri_angle"][idx_tri] < self.conf.combined_triangle_thresh
                src = points_lifted if use_lift else points_triangulated
                src_idx = idx_lift if use_lift else idx_tri
                for key in keys:
                    candidate_points[key].append(src[key][src_idx])
                continue

            if points_lifted["tri_angle"][idx_lift] < self.conf.combined_triangle_thresh:
                for key in keys:
                    candidate_points[key].append(points_lifted[key][idx_lift])

        for tri_indices in tri_map.values():
            for idx_tri in tri_indices:
                if points_triangulated["tri_angle"][idx_tri] >= self.conf.combined_triangle_thresh:
                    for key in keys:
                        candidate_points[key].append(points_triangulated[key][idx_tri])

        return candidate_points

    def _collect_prior_inlier_masks(self, imid, ref_imids, T_qry_cw, camera_qry, kps_qry, log_stage):
        inlier_masks = {}
        for ref_id in ref_imids:
            image_ref = self.mpsfm_rec.images[ref_id]
            camera_ref = self.mpsfm_rec.rec.cameras[image_ref.camera_id]
            matches_ref_qry = self.correspondences.matches(ref_id, imid)
            if len(matches_ref_qry) == 0:
                inlier_masks[ref_id] = np.zeros(0, dtype=bool)
                continue
            kps_ref = self.mpsfm_rec.keypoints(ref_id)
            inlier_masks[ref_id] = self._inlier_mask_for_pair_under_pose(
                ref_id,
                imid,
                image_ref.cam_from_world,
                T_qry_cw,
                camera_ref,
                camera_qry,
                matches_ref_qry,
                kps_ref,
                kps_qry,
                log_stage=log_stage,
            )
        return inlier_masks

    def _sanitize_ref_imids(self, ref_imids):
        """Keep only currently registered reference images."""
        if ref_imids is None:
            ref_imids = self.mpsfm_rec.registered_images.keys()

        reg_ids = set(self.mpsfm_rec.registered_images.keys())
        valid_ref_ids = []
        seen = set()
        for ref_id in ref_imids:
            ref_id = int(ref_id)
            if ref_id in seen or ref_id not in reg_ids:
                continue
            seen.add(ref_id)
            valid_ref_ids.append(ref_id)
        return valid_ref_ids

    def _count_ref_correspondences(self, imid, ref_imids):
        total_corrs = 0
        nonempty_refs = 0
        for ref_id in ref_imids:
            num_corrs = int(self.correspondences.num_correspondences_between_images(ref_id, imid))
            total_corrs += num_corrs
            if num_corrs > 0:
                nonempty_refs += 1
        return total_corrs, nonempty_refs

    def _filter_existing_point3d_refs(self, image_ref, pts2d_ids_ref, use_3d):
        """Drop stale Point3D references left behind after filtering."""
        use_3d = np.asarray(use_3d, dtype=bool).copy()
        point3D_ids = []
        for idx, (pt2d_id, has_3d) in enumerate(zip(pts2d_ids_ref, use_3d)):
            if not has_3d:
                continue
            point3D_id = int(image_ref.points2D[pt2d_id].point3D_id)
            if point3D_id not in self.mpsfm_rec.points3D:
                use_3d[idx] = False
                continue
            point3D_ids.append(point3D_id)
        return use_3d, np.array(point3D_ids, dtype=int)

    def _find_2D3D_pairs_with_inlier_mask(self, im_ref_id, imid, image_ref, image, inlier_mask, pair2D3D):
        """Collect ref-to-query 2D-3D pairs under a fixed inlier mask."""
        corr = self.correspondences.matches(im_ref_id, imid)
        if len(corr) == 0:
            pair2D3D["2d"] = np.zeros((0, 2))
            pair2D3D["3d"] = np.zeros((0, 3))
            pair2D3D["lifted"] = np.zeros(0, dtype=bool)
            pair2D3D["3dids"] = np.zeros(0, dtype=int)
            return

        if inlier_mask is not None:
            inlier_mask = np.asarray(inlier_mask, dtype=bool)
            if len(inlier_mask) != len(corr):
                min_len = min(len(inlier_mask), len(corr))
                corr = corr[:min_len]
                inlier_mask = inlier_mask[:min_len]
            corr = corr[inlier_mask]

        if len(corr) == 0:
            pair2D3D["2d"] = np.zeros((0, 2))
            pair2D3D["3d"] = np.zeros((0, 3))
            pair2D3D["lifted"] = np.zeros(0, dtype=bool)
            pair2D3D["3dids"] = np.zeros(0, dtype=int)
            return

        pts2d_ids_ref, pts2d_ids_qry = corr.T
        use_3d = np.array([image_ref.points2D[pt].has_point3D() for pt in pts2d_ids_ref], dtype=bool)
        use_3d, point3D_ids = self._filter_existing_point3d_refs(image_ref, pts2d_ids_ref, use_3d)
        self._collect_pairs(
            im_ref_id,
            image_ref,
            image,
            pts2d_ids_ref,
            pts2d_ids_qry,
            use_3d,
            point3D_ids,
            pair2D3D,
        )

    @staticmethod
    def _compose_left_pose_delta(T_cw: pycolmap.Rigid3d, delta):
        """左乘 se(3) 增量更新绝对位姿."""
        R0 = T_cw.rotation.matrix()
        t0 = np.asarray(T_cw.translation, dtype=np.float64)
        R_upd = Rotation.from_rotvec(delta[:3]).as_matrix()
        t_upd = delta[3:]
        R_new = R_upd @ R0
        t_new = R_upd @ t0 + t_upd
        return R_new, t_new

    def _refine_prior_pose_with_inliers(self, imid, ref_imids, inlier_masks_map, initial_pose):
        """使用 inliers 构造 2D-3D 约束，对 prior pose 做 pose-only refine."""
        if not self.conf.refine_remaining_prior_pose:
            return initial_pose, False, {}

        image = self.mpsfm_rec.images[imid]
        camera = self.mpsfm_rec.rec.cameras[image.camera_id]

        pair2D3D = defaultdict(dict)
        nonempty_refs = 0
        for ref_id in ref_imids:
            image_ref = self.mpsfm_rec.images[ref_id]
            inlier_mask = inlier_masks_map.get(ref_id, None)
            self._find_2D3D_pairs_with_inlier_mask(ref_id, imid, image_ref, image, inlier_mask, pair2D3D[ref_id])
            if pair2D3D[ref_id].get("2d", np.zeros((0, 2))).shape[0] > 0:
                nonempty_refs += 1

        if len(pair2D3D) == 0:
            return initial_pose, False, {}

        points2D, points3D, _, _, _ = self._process_2D3D_pairs(pair2D3D)
        min_corrs = int(self.conf.prior_pose_refine_min_corrs)
        if len(points2D) < min_corrs:
            return initial_pose, False, {"num_corrs": int(len(points2D)), "required_corrs": min_corrs}

        reproj_thresh = float(self.conf.prior_pose_refine_reproj_thresh)

        def reproj_errors_from_rt(R_mat, t_vec):
            xyz_cam = (R_mat @ points3D.T).T + t_vec[None]
            residual = np.full((len(points3D), 2), reproj_thresh * 10.0, dtype=np.float64)
            valid = xyz_cam[:, 2] > 1e-9
            if np.any(valid):
                proj = camera.img_from_cam(xyz_cam[valid])
                residual[valid] = proj - points2D[valid]
            return residual, valid

        def residual(delta):
            R_mat, t_vec = self._compose_left_pose_delta(initial_pose, delta)
            e2d, _ = reproj_errors_from_rt(R_mat, t_vec)
            return e2d.reshape(-1)

        e_before, valid_before = reproj_errors_from_rt(initial_pose.rotation.matrix(), np.asarray(initial_pose.translation))
        err_before = np.linalg.norm(e_before, axis=1)
        inliers_before = int(np.count_nonzero(valid_before & (err_before <= reproj_thresh)))

        try:
            result = least_squares(
                residual,
                np.zeros(6, dtype=np.float64),
                method="trf",
                loss="soft_l1",
                f_scale=reproj_thresh,
                max_nfev=int(self.conf.prior_pose_refine_max_iters),
            )
        except Exception as exc:
            self.log(f"[PriorPoseRefine] least_squares failed for imid={imid}: {exc}", level=1)
            return initial_pose, False, {"error": str(exc), "num_corrs": int(len(points2D))}

        if not result.success:
            return initial_pose, False, {
                "status": int(getattr(result, "status", -1)),
                "message": str(getattr(result, "message", "")),
                "num_corrs": int(len(points2D)),
            }

        R_opt, t_opt = self._compose_left_pose_delta(initial_pose, result.x)
        refined_pose = pycolmap.Rigid3d(pycolmap.Rotation3d(R_opt), t_opt)

        e_after, valid_after = reproj_errors_from_rt(R_opt, t_opt)
        err_after = np.linalg.norm(e_after, axis=1)
        inliers_after = int(np.count_nonzero(valid_after & (err_after <= reproj_thresh)))

        rot_delta = R_opt @ initial_pose.rotation.matrix().T
        rot_diff_deg = float(np.rad2deg(Rotation.from_matrix(rot_delta).magnitude()))
        trans_diff = float(np.linalg.norm(t_opt - np.asarray(initial_pose.translation)))

        stats = {
            "stage": "register_next_prior",
            "imid": int(imid),
            "num_refs": int(nonempty_refs),
            "num_corrs": int(len(points2D)),
            "inliers_before": inliers_before,
            "inliers_after": inliers_after,
            "rms_before": float(np.sqrt(np.mean(err_before**2))) if len(err_before) else 0.0,
            "rms_after": float(np.sqrt(np.mean(err_after**2))) if len(err_after) else 0.0,
            "rot_diff_deg": rot_diff_deg,
            "trans_diff": trans_diff,
            "nfev": int(result.nfev),
        }
        return refined_pose, True, stats

    def _inlier_mask_for_pair_under_pose(
        self,
        ref_imid,
        qry_imid,
        T_ref_cw,
        T_qry_cw,
        camera_ref,
        camera_qry,
        matches,
        kps_ref,
        kps_qry,
        log_stage=None,
    ):
        """在固定先验位姿下，对 (ref -> qry) 的匹配进行基于重投影误差的内点筛选。

        先将参考帧可用的三角化点与深度 lift 点合并，再统一进行重投影误差筛选。
        """
        px_thresh = 12.0  # 默认重投影阈值
        if matches is None or len(matches) == 0:
            return np.zeros(0, dtype=bool)

        ref_idx = matches[:, 0].astype(int)
        qry_idx = matches[:, 1].astype(int)
        xy_ref = kps_ref[ref_idx]
        xy_qry = kps_qry[qry_idx]
        mask_out = np.zeros(len(matches), dtype=bool)

        pair_to_indices = defaultdict(list)
        for midx, pair in enumerate(zip(ref_idx, qry_idx)):
            pair_to_indices[(int(pair[0]), int(pair[1]))].append(midx)

        image_ref = self.mpsfm_rec.images[ref_imid]

        has3d = np.array([image_ref.points2D[int(pid)].has_point3D() for pid in ref_idx], dtype=bool)
        tri_candidates = []
        if np.any(has3d):
            p3d_ids = np.array([image_ref.points2D[int(pid)].point3D_id for pid in ref_idx[has3d]], dtype=int)
            xyz_world_p3d = np.array([self.mpsfm_rec.points3D[int(pid)].xyz for pid in p3d_ids])
            for rid, qid, xyz in zip(ref_idx[has3d], qry_idx[has3d], xyz_world_p3d):
                tri_candidates.append(
                    {
                        "pt2d_id_1": int(rid),
                        "pt2d_id_2": int(qid),
                        "xyz": np.asarray(xyz, dtype=np.float64),
                        "source": "tri",
                    }
                )

        lift_candidates = []
        unproj_cam, valid_lifted = self._lift_points_for_init(ref_imid, xy_ref, camera_ref)
        lift_candidate_total = int(len(valid_lifted))
        lift_depth_valid = int(np.count_nonzero(valid_lifted))
        if unproj_cam.shape[0] > 0:
            xyz_world_l = T_ref_cw.inverse() * unproj_cam
            for rid, qid, xyz, valid_flag in zip(ref_idx, qry_idx, xyz_world_l, valid_lifted):
                if not valid_flag:
                    continue
                lift_candidates.append(
                    {
                        "pt2d_id_1": int(rid),
                        "pt2d_id_2": int(qid),
                        "xyz": np.asarray(xyz, dtype=np.float64),
                        "source": "lift",
                    }
                )

        def _combine_candidates(tri_list, lift_list):
            combined = []
            tri_map = {cand["pt2d_id_1"]: cand for cand in tri_list}
            lift_map = {cand["pt2d_id_1"]: cand for cand in lift_list}
            all_refs = set(tri_map.keys()) | set(lift_map.keys())
            for ref_id in all_refs:
                tri_c = tri_map.get(ref_id)
                lift_c = lift_map.get(ref_id)
                if tri_c:
                    combined.append(tri_c)
                    continue
                if lift_c:
                    combined.append(lift_c)
            return combined

        combined_candidates = _combine_candidates(tri_candidates, lift_candidates)
        combined_total = len(combined_candidates)
        if combined_total == 0:
            stats = {
                    "combined": 0,
                    "depth_reject": 0,
                    "depth_valid": 0,
                    "lift_candidates": lift_candidate_total,
                    "lift_depth_valid": 0,
                    "lift_inliers": 0,
                    "lift_kept": 0,
                    "matches": len(matches),
                    "reproj_inliers": 0,
                    "reproj_reject": len(matches),
                    "tri_candidates": len(tri_candidates),
                    "tri_depth_valid": 0,
                    "tri_inliers": 0,
                    "tri_kept": len(tri_candidates),
                }
            return mask_out

        xyz_world = np.array([cand["xyz"] for cand in combined_candidates])
        xy_qry_target = np.array([kps_qry[cand["pt2d_id_2"]] for cand in combined_candidates])
        sources = np.array([cand.get("source", "tri") for cand in combined_candidates])
        is_tri = sources == "tri"
        is_lift = sources == "lift"
        xyz_qry_cam = T_qry_cw * xyz_world
        valid_depth = xyz_qry_cam[:, 2] > 0
        reproj_errs = np.full(len(combined_candidates), np.inf, dtype=np.float64)
        if np.any(valid_depth):
            proj_qry = camera_qry.img_from_cam(xyz_qry_cam[valid_depth])
            reproj_errs[valid_depth] = np.linalg.norm(proj_qry - xy_qry_target[valid_depth], axis=1)
        inlier_flags = reproj_errs <= px_thresh
        for cand, is_inlier in zip(combined_candidates, inlier_flags):
            if not is_inlier:
                continue
            pair = (cand["pt2d_id_1"], cand["pt2d_id_2"])
            for match_idx in pair_to_indices.get(pair, []):
                mask_out[match_idx] = True

        depth_valid = int(np.count_nonzero(valid_depth))
        depth_reject = combined_total - depth_valid
        lift_candidates_kept = int(np.count_nonzero(is_lift))
        tri_candidates_kept = int(np.count_nonzero(is_tri))
        lift_depth_valid_final = int(np.count_nonzero(is_lift & valid_depth))
        tri_depth_valid_final = int(np.count_nonzero(is_tri & valid_depth))
        lift_inliers = int(np.count_nonzero(is_lift & inlier_flags))
        tri_inliers = int(np.count_nonzero(is_tri & inlier_flags))
        reproj_inliers = int(np.count_nonzero(mask_out))
        reproj_reject = len(matches) - reproj_inliers

        stats = {
            "combined": combined_total,
            "depth_reject": depth_reject,
            "depth_valid": depth_valid,
            "lift_candidates": lift_candidate_total,
            "lift_depth_valid": lift_depth_valid_final,
            "lift_inliers": lift_inliers,
            "lift_kept": lift_candidates_kept,
            "matches": len(matches),
            "reproj_inliers": reproj_inliers,
            "reproj_reject": reproj_reject,
            "tri_candidates": len(tri_candidates),
            "tri_depth_valid": tri_depth_valid_final,
            "tri_inliers": tri_inliers,
            "tri_kept": tri_candidates_kept,
        }
        return mask_out

    def _find_2D3D_pairs(self, im_ref_id, imid, image_ref, image, pair2D3D):
        corr = self.correspondences.matches(im_ref_id, imid)
        if im_ref_id in self.mpsfm_rec.images[image.imid].ignore_matches_AP:
            keep_matches = ~self.mpsfm_rec.images[image.imid].ignore_matches_AP[im_ref_id]
            corr = corr[keep_matches]
        if len(corr) == 0:
            pair2D3D["2d"] = np.zeros((0, 2))
            pair2D3D["3d"] = np.zeros((0, 3))
            pair2D3D["lifted"] = np.zeros(0, dtype=bool)
            return

        pts2d_ids_ref, pts2d_ids_qry = corr.T

        use_3d = np.array([image_ref.points2D[pt].has_point3D() for pt in pts2d_ids_ref], dtype=bool)
        use_3d, point3D_ids = self._filter_existing_point3d_refs(image_ref, pts2d_ids_ref, use_3d)
        self._collect_pairs(
            im_ref_id,
            image_ref,
            image,
            pts2d_ids_ref,
            pts2d_ids_qry,
            use_3d,
            point3D_ids,
            pair2D3D,
        )
    
    def register_and_triangulate_init_pair(self, imid1, imid2):
        """Register initial image pair and triangulate it's points."""
        matches = self.correspondences.matches(imid1, imid2)
        kps1 = self.mpsfm_rec.keypoints(imid1)
        kps2 = self.mpsfm_rec.keypoints(imid2)
        camera1 = self.mpsfm_rec.camera(imid1)
        camera2 = self.mpsfm_rec.camera(imid2)

        # 先验优先：T2 的优先级为 prior2 > AP > E
        name1 = self.mpsfm_rec.images[imid1].name
        name2 = self.mpsfm_rec.images[imid2].name
        prior1 = self.get_prior_pose(name1)
        prior2 = self.get_prior_pose(name2)
        

        if prior1 is not None:
            T_c1w = self.mpsfm_rec.align_prior_pose_to_current_world(prior1)
        else:
            T_c1w = pycolmap.Rigid3d()

        matches_used = None
        T_c2w = None
        rescale = 1
        unproj_cam_cached = None
        valid_lifted_cached = None

        if prior2 is not None:
            T_c2w = self.mpsfm_rec.align_prior_pose_to_current_world(prior2)
            prior_inlier_mask = self._init_pair_prior_inlier_mask_from_pose(
                imid1,
                imid2,
                T_c1w,
                T_c2w,
                matches,
                kps1,
                kps2,
                camera1,
                camera2,
            )
            self.mpsfm_rec.last_ap_inlier_masks = {imid1: prior_inlier_mask}
            lookahead_imids = self._select_init_lookahead_images(imid1, imid2)

            prior_hyp = self._make_init_hypothesis(
                "prior",
                T_c2w,
                prior_inlier_mask,
                matches,
                imid1,
                imid2,
                T_c1w,
                kps1,
                camera1,
                camera2,
                extra_stats={
                    "stage": "prior",
                    "prior_inliers": int(np.count_nonzero(prior_inlier_mask)),
                },
            )
            refined_hyps = []

            if self.conf.refine_init_prior_pose:
                refine_out = self._refine_init_pair_pose_bootstrap(
                    imid1, imid2, T_c1w, T_c2w, matches, kps1, kps2, camera1, camera2
                )
                for hyp_name, hyp_key in (("rotation", "rotation"), ("refined", "full")):
                    hyp_out = refine_out.get(hyp_key)
                    if hyp_out is None:
                        continue
                    hyp = self._make_init_hypothesis(
                        hyp_name,
                        hyp_out["pose"],
                        prior_inlier_mask,
                        matches,
                        imid1,
                        imid2,
                        T_c1w,
                        kps1,
                        camera1,
                        camera2,
                        core_mask=hyp_out["core_mask"],
                        support_mask=hyp_out["support_mask"],
                        extra_stats=hyp_out["stats"],
                    )
                    if hyp is not None:
                        refined_hyps.append(hyp)

            if prior_hyp is None and len(refined_hyps) == 0:
                print(f"Init pair {imid1} and {imid2} has no valid high-confidence seed hypothesis")
                return False

            if prior_hyp is not None:
                prior_hyp["lookahead"] = self._evaluate_init_hypothesis_lookahead(
                    imid1,
                    imid2,
                    prior_hyp["seed_data"],
                    lookahead_imids,
                )
            for hyp in refined_hyps:
                hyp["lookahead"] = self._evaluate_init_hypothesis_lookahead(
                    imid1,
                    imid2,
                    hyp["seed_data"],
                    lookahead_imids,
                )

            def _hyp_rank_key(hyp):
                look = hyp["lookahead"]
                return (
                    look["num_registerable"],
                    look["median_inliers"],
                    look["median_inlier_ratio"],
                    look["score"],
                    hyp["seed_data"]["num_seed_points"],
                )

            selected_hyp = prior_hyp
            if refined_hyps:
                best_refined_hyp = max(refined_hyps, key=_hyp_rank_key)
                if selected_hyp is None or self._is_refined_init_hypothesis_better(
                    best_refined_hyp["lookahead"],
                    selected_hyp["lookahead"],
                ):
                    selected_hyp = best_refined_hyp

            self.mpsfm_rec.last_init_hypothesis_selection = {
                "lookahead_imids": [int(imid) for imid in lookahead_imids],
                "selected_name": selected_hyp["name"],
                "prior": None
                if prior_hyp is None
                else {
                    "seed_source": prior_hyp["seed_source"],
                    "seed_points": int(prior_hyp["seed_data"]["num_seed_points"]),
                    "lookahead": prior_hyp["lookahead"],
                    "stats": prior_hyp["stats"],
                },
                "refined": [
                    {
                        "name": hyp["name"],
                        "seed_source": hyp["seed_source"],
                        "seed_points": int(hyp["seed_data"]["num_seed_points"]),
                        "lookahead": hyp["lookahead"],
                        "stats": hyp["stats"],
                    }
                    for hyp in refined_hyps
                ],
            }
            T_c2w = selected_hyp["pose"]
            matches_used = selected_hyp["matches_used"]
            rescale = selected_hyp["rescale"]
            self.mpsfm_rec.last_ap_inlier_masks = {imid1: selected_hyp["seed_mask"]}
        else:
            unproj_cam0, valid_lifted0 = self._lift_points_for_init(imid1, kps1, camera1)
            valid_matches0 = matches[valid_lifted0[matches[:, 0]]]
            unproj_world0 = T_c1w.inverse() * unproj_cam0
            AP_info = self.absolute_pose_estimator(
                kps2[valid_matches0[:, 1]], unproj_world0[valid_matches0[:, 0]], camera2
            )
            ap_min_num_inliers = self.conf.colmap_options.abs_pose_min_num_inliers
            ap_sufficient = (AP_info is not None) and (AP_info["num_inliers"] >= ap_min_num_inliers)

            if ap_sufficient:
                E_info = self.relative_pose_estimator(
                    kps1[matches[:, 0]], kps2[matches[:, 1]], camera1, camera2
                )
                inlier_matches_e = matches[E_info["inlier_mask"]]
                T_c2w_e = E_info["cam2_from_cam1"] * T_c1w
                pts_tri_e = self._candidate_points3D_for_init(
                    T_c1w, T_c2w_e, inlier_matches_e, self.mpsfm_rec.images[imid1], self.mpsfm_rec.images[imid2], camera1, camera2
                )
                triangles = np.array(pts_tri_e["tri_angle"]) if len(pts_tri_e["tri_angle"]) > 0 else np.array([])
                high_parallax = (triangles > self.conf.parallax_thresh).sum() > AP_info["num_inliers"]

                if high_parallax:
                    T_c2w = T_c2w_e
                    matches_used = inlier_matches_e
                    rescale = self._estimate_init_rescale(imid1, kps1, T_c1w, pts_tri_e)
                else:
                    T_c2w = AP_info["cam_from_world"]
                    matches_used = valid_matches0[AP_info["inlier_mask"]]
                    unproj_cam_cached = unproj_cam0
                    valid_lifted_cached = valid_lifted0
            else:
                E_info = self.relative_pose_estimator(
                    kps1[matches[:, 0]], kps2[matches[:, 1]], camera1, camera2
                )
                inlier_matches = matches[E_info["inlier_mask"]]
                T_c2w = E_info["cam2_from_cam1"] * T_c1w
                matches_used = inlier_matches
                pts_tri_e = self._candidate_points3D_for_init(
                    T_c1w, T_c2w, inlier_matches, self.mpsfm_rec.images[imid1], self.mpsfm_rec.images[imid2], camera1, camera2
                )
                rescale = self._estimate_init_rescale(imid1, kps1, T_c1w, pts_tri_e)

        if rescale == 1 and unproj_cam_cached is not None:
            unproj_cam, valid_lifted = unproj_cam_cached, valid_lifted_cached
        else:
            unproj_cam, valid_lifted = self._lift_points_for_init(imid1, kps1, camera1, rescale=rescale)
        unproj_world = T_c1w.inverse() * unproj_cam
        pts_lift = self._candidate_lift_for_init(
            T_c1w, T_c2w, matches_used[valid_lifted[matches_used[:, 0]]], unproj_world
        )
        pts_tri = self._candidate_points3D_for_init(
            T_c1w, T_c2w, matches_used, self.mpsfm_rec.images[imid1], self.mpsfm_rec.images[imid2], camera1, camera2
        )
        cand = self._merge_candidate_points(pts_lift, pts_tri)

        self.mpsfm_rec.images[imid1].cam_from_world = T_c1w
        self.mpsfm_rec.images[imid2].cam_from_world = T_c2w

        self.mpsfm_rec.register_image(imid1)
        self.mpsfm_rec.register_image(imid2)
        if len(cand.get("xyz", [])) < 3:
            print(f"Init pair {imid1} and {imid2} has less than 3 points to triangulate. Not registered")
            return False
        for i, xyz in enumerate(cand["xyz"]):
            track = pycolmap.Track()
            track.add_element(imid1, cand["pt2d_id_1"][i])
            track.add_element(imid2, cand["pt2d_id_2"][i])
            if (
                self.mpsfm_rec.images[imid1].points2D[cand["pt2d_id_1"][i]].has_point3D()
                or self.mpsfm_rec.images[imid2].points2D[cand["pt2d_id_2"][i]].has_point3D()
            ):
                continue
            if (
                self.conf.colmap_options.init_min_tri_angle < cand["tri_angle"][i]
                and cand["posdepth1"][i]
                and cand["posdepth2"][i]
            ):
                self.mpsfm_rec.obs.add_point3D(xyz, track)
        return not len(self.mpsfm_rec.points3D) < 3

    def register_next_image(self, imid, ref_imids=None, **kwargs):
        """Register next image and triangulate points."""
        image = self.mpsfm_rec.images[imid]
        camera = self.mpsfm_rec.rec.cameras[image.camera_id]

        image_name = image.name
        prior_pose = self.get_prior_pose(image_name)
        if prior_pose is not None:
            self._update_world_sim_from_priors()
            image.cam_from_world = self.mpsfm_rec.align_prior_pose_to_current_world(prior_pose)

            ref_imids = self._sanitize_ref_imids(ref_imids)
            if len(ref_imids) == 0:
                print(f"\nImage {imid} has no valid registered reference images")
                return False
            total_corrs, nonempty_refs = self._count_ref_correspondences(imid, ref_imids)
            if total_corrs == 0:
                print(f"\nImage {imid} has no correspondences to registered reference images")
                return False

            kps_qry = self.mpsfm_rec.keypoints(imid)
            camera_qry = camera
            inlier_masks_map = self._collect_prior_inlier_masks(
                imid,
                ref_imids,
                image.cam_from_world,
                camera_qry,
                kps_qry,
                log_stage="register_next",
            )

            refined_pose, refined_ok, refine_stats = self._refine_prior_pose_with_inliers(
                imid, ref_imids, inlier_masks_map, image.cam_from_world
            )
            if refined_ok:
                image.cam_from_world = refined_pose
                inlier_masks_map = self._collect_prior_inlier_masks(
                    imid,
                    ref_imids,
                    image.cam_from_world,
                    camera_qry,
                    kps_qry,
                    log_stage="register_next_refined",
                )

            self.mpsfm_rec.last_ap_inlier_masks = inlier_masks_map
            if nonempty_refs == 0:
                print(f"\nImage {imid} has no usable registered reference images")
                return False
            self.mpsfm_rec.register_image(imid)
            return True

        ref_imids = self._sanitize_ref_imids(ref_imids)
        if len(ref_imids) == 0:
            print(f"\nImage {imid} has no valid registered reference images")
            return False
        total_corrs, _ = self._count_ref_correspondences(imid, ref_imids)
        if total_corrs == 0:
            print(f"\nImage {imid} has no correspondences to registered reference images")
            return False

        self.registration_cache[imid]["store_matches"] = {}
        ap_min_num_inliers = self.conf.colmap_options.abs_pose_min_num_inliers
        if self.half_ap_min_inliers:
            ap_min_num_inliers = int(ap_min_num_inliers / (1.2**self.half_ap_min_inliers))
        force_registration = self.half_ap_min_inliers >= self.conf.reduce_min_inliers_at_failure

        while True:
            pair2D3D = defaultdict(dict)
            for im_ref_id in ref_imids:
                image_ref = self.mpsfm_rec.images[im_ref_id]
                self._find_2D3D_pairs(im_ref_id, imid, image_ref, image, pair2D3D[im_ref_id])

            points2D, points3D, stack_order, lifted_mask, ids3d = self._process_2D3D_pairs(pair2D3D)

            unique_ids3d, unique_indices, el_to_unique_index = np.unique(ids3d, return_index=True, return_inverse=True)
            triangpts3D = points3D[~lifted_mask][unique_indices]
            triangpts2D = points2D[~lifted_mask][unique_indices]
            if self.conf.lifted_registration:
                liftedpts2D, liftedpts3D = points2D[lifted_mask], points3D[lifted_mask]
            else:
                liftedpts2D, liftedpts3D = np.zeros((0, 2)), np.zeros((0, 3))
            points2D = np.concatenate([triangpts2D, liftedpts2D])
            points3D = np.concatenate([triangpts3D, liftedpts3D])

            if len(points2D) < 3:
                print(f"\nImage {imid} has less than 3 points to triangulate. Not registered")
                return False

            AP_info = self.absolute_pose_estimator(points2D, points3D, camera)
            if AP_info is None:
                print("\nAP estim No inliers found")
                return False

            if AP_info["num_inliers"] < ap_min_num_inliers and not force_registration:
                print(f"\nAP estim Not enough inliers: {ap_min_num_inliers}")
                return False

            inlier_mask = AP_info["inlier_mask"]
            ref_match_sizes = [len(pair2D3D[im_ref_id]["2d"]) for im_ref_id in stack_order]
            split_indices = np.cumsum(ref_match_sizes)[:-1]
            # mapping back masks to correspondences
            t_mask = inlier_mask[: len(triangpts3D)]
            l_mask = inlier_mask[len(triangpts3D) :]
            remapped_inl_mask = np.ones(len(lifted_mask), dtype=bool)
            remapped_inl_mask[lifted_mask] = l_mask
            remapped_inl_mask[~lifted_mask] = t_mask[el_to_unique_index]
            assert (
                set(np.unique(ids3d[t_mask[el_to_unique_index]])) - set(unique_ids3d[inlier_mask[: len(triangpts3D)]])
                == set()
            )

            split_mask = dict(zip(stack_order, np.split(remapped_inl_mask, split_indices)))
            best_id = self.mpsfm_rec.best_next_ref_imid
            self.mpsfm_rec.last_ap_inlier_masks = split_mask

            if self.conf.resample_bunlde:
                compare_ids = set(stack_order)
                compare_ids.remove(best_id)
                compare_ids = list(compare_ids)
                print()
                print("Best:", best_id, "other:", compare_ids)
                print("_+_+_+_+_+_+" * 7)
                print(
                    "TOTS:",
                    split_mask[best_id].sum(),
                    "vs",
                    [split_mask[im_ref_id].sum() for im_ref_id in compare_ids],
                )
                print(
                    "RATIOS:",
                    split_mask[best_id].sum() / len(split_mask[best_id]),
                    "vs",
                    [split_mask[im_ref_id].sum() / len(split_mask[im_ref_id]) for im_ref_id in compare_ids],
                )
                print("_+_+_+_+_+_+" * 7)
                if (
                    split_mask[best_id].sum() / len(split_mask[best_id]) < 0.1
                    and np.nanmax(
                        [split_mask[im_ref_id].sum() / len(split_mask[im_ref_id]) for im_ref_id in compare_ids]
                    )
                    > 0.2
                ):
                    for ref_id in split_mask:
                        if len(split_mask[ref_id]) > 0:

                            if ref_id in self.mpsfm_rec.images[imid].ignore_matches_AP:
                                used = ~self.mpsfm_rec.images[imid].ignore_matches_AP[ref_id]
                                self.mpsfm_rec.images[imid].ignore_matches_AP[ref_id][used] |= split_mask[ref_id]
                            else:
                                self.mpsfm_rec.images[imid].ignore_matches_AP[ref_id] = split_mask[ref_id]

                    continue

            image.cam_from_world = AP_info["cam_from_world"]
            self.mpsfm_rec.register_image(imid)
            break

        return True

    def register_and_triangulate_next_image(self, imid, ref_imids=None):
        """Register next image and triangulate points."""
        if not self.register_next_image(imid, ref_imids=ref_imids):
            return False
        if int(self.correspondences.num_correspondences_for_image(imid)) == 0:
            print(f"\nImage {imid} has no graph correspondences after registration")
            return False

        return self.triangulate_image(imid)

    def _collect_pairs(
        self, im_ref_id, image_ref, image, pts2d_ids_ref, pts2d_ids_qry, use_3d, point3D_ids, pair2D3D, **kwrags
    ):
        pair2D3D["2d"] = np.array([image.points2D[pt].xy for pt in pts2d_ids_qry])
        pair2D3D["3d"] = np.ones((pts2d_ids_ref.shape[0], 3)) * -1
        if sum(use_3d) > 0:
            if self.conf.robust_triangles is not None and self.conf.lifted_registration:
                risky_mask = self.mpsfm_rec.find_points3D_with_small_triangulation_angle(
                    min_angle=self.conf.robust_triangles, point3D_ids=point3D_ids
                )
                use_3d[use_3d] &= ~risky_mask
                point3D_ids = point3D_ids[~risky_mask]
            if sum(use_3d) > 0:
                pair2D3D["3d"][use_3d] = np.array([self.mpsfm_rec.points3D[pt].xyz for pt in point3D_ids])
        if self.conf.lifted_registration:
            if (~use_3d).sum() > 0:
                pts2Dids_ref_lifted = pts2d_ids_ref[~use_3d]
                pair2D3D["3d"][~use_3d] = self._lift_points_to_3d(
                    im_ref_id,
                    image_ref,
                    [self.mpsfm_rec.images[im_ref_id].points2D[pt].xy for pt in pts2Dids_ref_lifted],
                )

            pair2D3D["lifted"] = ~use_3d
        if not self.conf.lifted_registration:
            pair2D3D["3d"] = pair2D3D["3d"][use_3d]
            pair2D3D["2d"] = pair2D3D["2d"][use_3d]
        pair2D3D["3dids"] = point3D_ids
        return pair2D3D

    def _lift_points_to_3d(self, im_ref_id, image_ref, liftref_2d):
        """Lift 2D points to 3D space using depth maps and camera transformations."""
        xy = np.array(liftref_2d)
        image_depth = self.mpsfm_rec.images[im_ref_id].depth
        if (not self.conf.lifted_registration) or image_depth is None or xy.size == 0:
            return np.zeros((len(xy), 3), dtype=np.float64)
        d = self.mpsfm_rec.images[im_ref_id].depth.data_at_kps(xy)[:, None]
        camera_ref = self.mpsfm_rec.rec.cameras[image_ref.camera_id]
        return image_ref.cam_from_world.inverse() * (
            np.concatenate([camera_ref.cam_from_img(xy), np.ones((xy.shape[0], 1))], -1) * d
        )

    def _lift_points_for_init(self, im_ref_id, liftref_2d, camera_ref, rescale=1):
        """Lift 2D points to 3D space using depth maps and camera transformations."""
        xy = np.array(liftref_2d)
        image_depth = self.mpsfm_rec.images[im_ref_id].depth
        if (not self.conf.lifted_registration) or image_depth is None or xy.size == 0:
            return np.zeros((len(xy), 3), dtype=np.float64), np.zeros(len(xy), dtype=bool)
        d = image_depth.data_prior_at_kps(xy)[:, None]
        if rescale != 1:
            d *= rescale
        valid = image_depth.valid_at_kps(xy)
        return (np.concatenate([camera_ref.cam_from_img(xy), np.ones((xy.shape[0], 1))], -1) * d), valid

    def _process_2D3D_pairs(self, pair2D3D):
        sorted_ref_ids = sorted(pair2D3D)
        pts_2D = np.concatenate([pair2D3D[ref_id]["2d"] for ref_id in sorted_ref_ids])
        pts_3D = np.concatenate([pair2D3D[ref_id]["3d"] for ref_id in sorted_ref_ids])
        if self.conf.lifted_registration:
            lifted = np.concatenate(
                [pair2D3D[ref_id]["lifted"] for ref_id in sorted_ref_ids if "lifted" in pair2D3D[ref_id]]
            )
        if not self.conf.lifted_registration or (~lifted).sum() > 0:
            if all(len(pair2D3D[ref_id]["3dids"]) == 0 for ref_id in sorted_ref_ids if "3dids" in pair2D3D[ref_id]):
                ids3d = np.zeros(0, dtype=int)
            else:
                ids3d = np.concatenate(
                    [pair2D3D[ref_id]["3dids"] for ref_id in sorted_ref_ids if "3dids" in pair2D3D[ref_id]]
                )
            if not self.conf.lifted_registration:
                lifted = np.zeros(len(ids3d), dtype=bool)
        else:
            ids3d = np.zeros(0, dtype=int)
        assert (~lifted).sum() == len(ids3d)
        return pts_2D, pts_3D, sorted_ref_ids, lifted, ids3d

    def triangulate_image(self, imid, **kwargs):
        """Triangulate points for the given image id."""
        return self.triangulator.triangulate_image(imid, **kwargs)

    def _candidate_lift_for_init(self, cam_from_world1, cam_from_world2, matches, lifted3D, inliers=None):
        """Collect candidate points for lifted 3D points."""
        if inliers is None:
            inliers = slice(None)

        candidate_points = defaultdict(list)

        for match in matches[inliers]:
            pt2d_id_1, pt2d_id_2 = match
            xyz = lifted3D[pt2d_id_1]

            projection_center1 = cam_from_world1.rotation.inverse() * -cam_from_world1.translation
            projection_center2 = cam_from_world2.rotation.inverse() * -cam_from_world2.translation
            tri_angle = calculate_triangulation_angle(projection_center1, projection_center2, xyz)
            posdepth1 = has_point_positive_depth(cam_from_world1.matrix(), xyz)
            posdepth2 = has_point_positive_depth(cam_from_world2.matrix(), xyz)
            candidate_points["pt2d_id_1"].append(pt2d_id_1)
            candidate_points["pt2d_id_2"].append(pt2d_id_2)
            candidate_points["tri_angle"].append(np.rad2deg(tri_angle))
            candidate_points["posdepth1"].append(posdepth1)
            candidate_points["posdepth2"].append(posdepth2)
            candidate_points["xyz"].append(xyz)
        return candidate_points
