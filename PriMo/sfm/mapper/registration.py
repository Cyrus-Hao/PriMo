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
        "ba_refine_prior_pose": True,
        "refine_init_prior_pose": True,
        "epipolar_refine_min_matches": 12,
        "epipolar_refine_max_iters": 300,
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

    def _refine_init_pair_pose_epipolar(
        self, ref_imid, qry_imid, T_c1w, T_c2w, matches, kps1, kps2, camera1, camera2
    ):
        if not self.conf.refine_init_prior_pose:
            return T_c2w, False, {}
        matches = np.asarray(matches)
        min_matches = self.conf.epipolar_refine_min_matches
        if len(matches) < min_matches:
            return T_c2w, False, {}

        pts1 = self._normalized_keypoints(camera1, kps1[matches[:, 0]])
        pts2 = self._normalized_keypoints(camera2, kps2[matches[:, 1]])
        if len(pts1) < min_matches:
            return T_c2w, False, {}

        R_rel, t_rel = self._relative_pose_components(T_c1w, T_c2w)
        baseline_norm = float(np.linalg.norm(t_rel))
        if baseline_norm < 1e-9:
            return T_c2w, False, {}
        t_rel = self._normalize_translation(t_rel)

        def residual(delta):
            R_curr, t_curr = self._compose_left_se3_delta(R_rel, t_rel, delta)
            t_curr = self._normalize_translation(t_curr)
            return self._epipolar_residuals(R_curr, t_curr, pts1, pts2)

        def residual_stats(R_mat, t_vec):
            res = self._epipolar_residuals(R_mat, t_vec, pts1, pts2)
            if len(res) == 0:
                return {
                    "count": 0,
                    "rms": 0.0,
                    "abs_mean": 0.0,
                    "abs_median": 0.0,
                    "abs_p95": 0.0,
                    "abs_max": 0.0,
                }
            abs_res = np.abs(res)
            return {
                "count": int(len(res)),
                "rms": float(np.sqrt(np.mean(res**2))),
                "abs_mean": float(np.mean(abs_res)),
                "abs_median": float(np.median(abs_res)),
                "abs_p95": float(np.percentile(abs_res, 95)),
                "abs_max": float(np.max(abs_res)),
            }

        stats_before = residual_stats(R_rel, t_rel)

        try:
            result = least_squares(
                residual,
                np.zeros(6, dtype=np.float64),
                method="lm",
                max_nfev=self.conf.epipolar_refine_max_iters,
            )
        except Exception as exc:
            self.log(f"[InitPair] Epipolar refine failed: {exc}", level=1)
            return T_c2w, False, {}

        if not result.success:
            return T_c2w, False, {}

        R_opt, t_opt = self._compose_left_se3_delta(R_rel, t_rel, result.x)
        t_opt_unit = self._normalize_translation(t_opt)
        stats_after = residual_stats(R_opt, t_opt_unit)

        rot_delta = R_opt @ R_rel.T
        rot_diff = Rotation.from_matrix(rot_delta)
        rot_diff_deg = float(np.rad2deg(rot_diff.magnitude()))
        trans_angle = float(
            np.rad2deg(
                np.arccos(
                    np.clip(
                        np.dot(t_rel, t_opt_unit) / (np.linalg.norm(t_rel) * np.linalg.norm(t_opt_unit)), -1.0, 1.0
                    )
                )
            )
        )

        refine_stats = {
            "num_matches": stats_before["count"],
            "baseline_norm": baseline_norm,
            "residual_before_rms": stats_before["rms"],
            "residual_before_p95": stats_before["abs_p95"],
            "residual_before_max": stats_before["abs_max"],
            "residual_after_rms": stats_after["rms"],
            "residual_after_p95": stats_after["abs_p95"],
            "residual_after_max": stats_after["abs_max"],
            "rot_diff_deg": rot_diff_deg,
            "trans_angle_deg": trans_angle,
            "nfev": int(result.nfev),
        }

        refined_translation = t_opt_unit * baseline_norm
        refined_relative = pycolmap.Rigid3d(pycolmap.Rotation3d(R_opt), refined_translation)
        refined_pose = refined_relative * T_c1w
        return refined_pose, True, refine_stats

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
        tri_world = np.vstack(candidate_points["xyz"])
        tri_cam = T_ref_cw * tri_world
        z = tri_cam[:, -1]
        mask = np.asarray(candidate_points["pt2d_id_1"], dtype=int)
        d = self.mpsfm_rec.images[ref_imid].depth.data_prior_at_kps(kps_ref[mask])
        return np.median(z / d)

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
        point3D_ids = np.array(
            [image_ref.points2D[pt].point3D_id for pt, has_3d in zip(pts2d_ids_ref, use_3d) if has_3d], dtype=int
        )
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

        use_3d = np.array([image_ref.points2D[pt].has_point3D() for pt in pts2d_ids_ref])
        point3D_ids = np.array(
            [image_ref.points2D[pt].point3D_id for pt, has_3d in zip(pts2d_ids_ref, use_3d) if has_3d]
        )
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
            pts_tri_prior = self._candidate_points3D_for_init(
                T_c1w, T_c2w, matches, self.mpsfm_rec.images[imid1], self.mpsfm_rec.images[imid2], camera1, camera2
            )
            rescale = self._estimate_init_rescale(imid1, kps1, T_c1w, pts_tri_prior)
            prior_inlier_mask = self._inlier_mask_for_pair_under_pose(
                imid1,
                imid2,
                T_c1w,
                T_c2w,
                camera1,
                camera2,
                matches,
                kps1,
                kps2,
                log_stage="pre_refine",
            )
            self.mpsfm_rec.last_ap_inlier_masks = {imid1: prior_inlier_mask}
            ap_min_num_inliers = self.conf.colmap_options.abs_pose_min_num_inliers
            prior_inliers = int(prior_inlier_mask.sum())
            fallback_thresh = max(3, ap_min_num_inliers // 2)
            use_prior_only = prior_inliers >= fallback_thresh
            matches_used = matches[prior_inlier_mask] if use_prior_only else matches

            if self.conf.refine_init_prior_pose:
                refine_matches = matches_used if len(matches_used) > 0 else matches
                T_c2w_refined, refined_ok, refine_stats = self._refine_init_pair_pose_epipolar(
                    imid1, imid2, T_c1w, T_c2w, refine_matches, kps1, kps2, camera1, camera2
                )
                if refined_ok:
                    T_c2w = T_c2w_refined
                    refined_mask = self._inlier_mask_for_pair_under_pose(
                        imid1,
                        imid2,
                        T_c1w,
                        T_c2w,
                        camera1,
                        camera2,
                        matches,
                        kps1,
                        kps2,
                        log_stage="post_refine",
                    )
                    self.mpsfm_rec.last_ap_inlier_masks = {imid1: refined_mask}
                    refined_inliers = int(refined_mask.sum())
                    if refined_inliers >= fallback_thresh:
                        matches_used = matches[refined_mask]
                    else:
                        matches_used = matches
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

            if ref_imids is None:
                ref_imids = self.mpsfm_rec.registered_images.keys()
            ref_imids = list(ref_imids)

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
            self.mpsfm_rec.register_image(imid)
            return True

        if ref_imids is None:
            ref_imids = self.mpsfm_rec.registered_images.keys()

        self.registration_cache[imid]["store_matches"] = {}
        ref_imids = list(ref_imids)
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
        d = self.mpsfm_rec.images[im_ref_id].depth.data_at_kps(xy)[:, None]
        camera_ref = self.mpsfm_rec.rec.cameras[image_ref.camera_id]
        return image_ref.cam_from_world.inverse() * (
            np.concatenate([camera_ref.cam_from_img(xy), np.ones((xy.shape[0], 1))], -1) * d
        )

    def _lift_points_for_init(self, im_ref_id, liftref_2d, camera_ref, rescale=1):
        """Lift 2D points to 3D space using depth maps and camera transformations."""
        xy = np.array(liftref_2d)
        d = self.mpsfm_rec.images[im_ref_id].depth.data_prior_at_kps(xy)[:, None]
        if rescale != 1:
            d *= rescale
        valid = self.mpsfm_rec.images[im_ref_id].depth.valid_at_kps(xy)
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
