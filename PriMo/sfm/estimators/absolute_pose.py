import pycolmap

from mpsfm.baseclass import BaseClass


class AbsolutePose(BaseClass):
    """Absolute pose estimation using COLMAP."""

    default_conf = {
        "colmap_estimation_options": pycolmap.AbsolutePoseEstimationOptions().todict(),
        "colmap_refinement_options": pycolmap.AbsolutePoseRefinementOptions().todict(),
        "verbose": 0,
    }
    default_conf["colmap_estimation_options"]["ransac"]["min_inlier_ratio"] = 0.25

    def __call__(self, points2D, points3D, camera):
        estim_options = dict(self.conf.colmap_estimation_options)
        estim_options["ransac"] = pycolmap.RANSACOptions(**estim_options["ransac"])
        return pycolmap.estimate_and_refine_absolute_pose(
            points2D,
            points3D,
            camera,
            estimation_options=pycolmap.AbsolutePoseEstimationOptions(**estim_options),
            refinement_options=pycolmap.AbsolutePoseRefinementOptions(**self.conf.colmap_refinement_options),
        )
