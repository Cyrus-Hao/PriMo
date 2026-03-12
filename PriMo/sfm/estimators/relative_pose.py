import pycolmap

from mpsfm.baseclass import BaseClass


class RelativePose(BaseClass):
    """Relative pose estimation using COLMAP."""

    default_conf = {
        "colmap_options": pycolmap.RANSACOptions().todict(),
        "verbose": 0,
    }

    def __call__(self, points1, points2, camera1, camera2):
        return pycolmap.essential_matrix_estimation(
            points1, points2, camera1, camera2, estimation_options=pycolmap.RANSACOptions(**self.conf.colmap_options)
        )
