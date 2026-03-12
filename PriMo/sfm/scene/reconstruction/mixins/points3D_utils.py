import numpy as np

from mpsfm.utils.geometry import project3D_colmap


class Points3DUtils:
    """Points3D utils mixin for reconstruction."""

    def project_image_3d_points(self, imid, pts3dids=None):
        """Projects 3D points into image"""
        image = self.images[imid]

        if pts3dids is None:
            pts2dids = image.get_observation_point2D_idxs()
            pts3dids = image.point3D_ids(pts2dids)

            if len(pts3dids) == 0:
                return None, None, None, None, False
            points3D = self.point3D_coordinates(pts3dids)
        else:
            points3D = self.point3D_coordinates(pts3dids)
            pts2dids = None
        camera = self.rec.cameras[image.camera_id]
        kps, depth = project3D_colmap(image, camera, points3D)
        return pts2dids, pts3dids, kps, depth, True

    def lifted_pointcovs_cam(self, dd, camera, keypoints, var_d, sigma_q=1):
        """Computes the covariance of lifted points in camera coordinates."""
        cc = np.array([camera.principal_point_x, camera.principal_point_y])
        ff = np.array([camera.focal_length_x, camera.focal_length_y])
        ff_inv = 1.0 / ff

        dpdd = np.hstack([(keypoints - cc[None, :]) * ff_inv[None, :], np.ones((keypoints.shape[0], 1))])[:, :, None]

        dpdq = np.zeros((keypoints.shape[0], 2, 3))
        dpdq[:, 0, 0] = dd * ff_inv[0]
        dpdq[:, 1, 1] = dd * ff_inv[1]
        dpdq = np.clip(dpdq, -1e6, 1e6)

        Cov_d = var_d[:, None, None] * np.einsum("nij,nkj->nik", dpdd, dpdd)

        Cov_q_small = sigma_q**2 * np.einsum("nij,nkj->nik", dpdq, dpdq)
        Cov_q = np.zeros((dpdq.shape[0], 3, 3))
        Cov_q[:, :2, :2] = Cov_q_small

        Cov_plifted_cam = Cov_d + Cov_q

        return Cov_plifted_cam

    def rotate_covs(self, Covs, R):
        """Rotates covariances by rotation matrix R"""
        return R[None] @ Covs @ R.T[None]

    def rotate_covs_to_world(self, Covs, imid):
        """Rotates covariances to world coordinates"""
        R = self.images[imid].cam_from_world.rotation.matrix()
        return self.rotate_covs(Covs, R)

    def rotate_covs_to_cam(self, Covs_world, imid):
        """Rotates covariances to camera coordinates from world coordinates"""
        R = self.images[imid].cam_from_world.rotation.matrix()
        return self.rotate_covs(Covs_world, R.T)

    def find_points3D_with_small_triangulation_angle(self, min_angle, point3D_ids):
        """Finds points3D with small triangulation angle"""
        return np.array(
            self.obs.find_small_angle_points_mask(
                float(min_angle),
                point3D_ids,
            )
        )
