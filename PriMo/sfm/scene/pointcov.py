import numpy as np


class PointCovs:
    """Class for 3D point covariances."""

    data = {}
    data_ap = None

    def points_zvars(self, image, p3d_ids=None):
        """Get the z-variance of 3D points in the image."""
        if p3d_ids is None:
            p3d_ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
        zvars = []
        R = image.cam_from_world.rotation.matrix()
        data = np.array([self.data[pt3D_id] for pt3D_id in p3d_ids])  # (N, 3, 3)
        intermediate = np.einsum("ij,njk->nik", R.T, data)  # (N, 3, 3)
        result = np.einsum("nij,jk->nik", intermediate, R)  # (N, 3, 3)
        zvars = result[:, 2, 2]
        return p3d_ids, zvars
