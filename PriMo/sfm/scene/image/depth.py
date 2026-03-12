"""Moudle for depth class"""

import cv2
import numpy as np

from mpsfm.baseclass import BaseClass
from mpsfm.sfm.scene.image.mixins.priorutils import PriorUtils
from mpsfm.sfm.scene.image.utils import get_continuity_mask


class Depth(BaseClass, PriorUtils):
    """Class for image depth."""

    default_conf = {
        "inherent_noise": 0.02,
        "std_multiplier": 1,
        "lc_std_multiplier": 10,
        "prior_std_multiplier": 3.33,
        "max_std": None,
        "use_continuity": True,
        "depth_lim": None,
        "fixed_uncertainty_val": 0.03,
        "fixed_uncertainty": False,
        "fill": True,
        "prior_uncertainty": True,  # uncertainties predicted by the model
        "flip_consistency": False,  # uncertainties via computing difference between the depth estimates of
        # the image and the flipped image
        "depth_uncertainty": 0.0263,  # uncertainty created by multiplying the depth by a constant
        "verbose": 0,
    }

    uncertainty_update_ap = None

    def _init(self, depth_dict, camera, kps, mask=None, **kwargs):
        self.kps = kps
        self.camera = camera

        self.scale = 1
        self.shift = 0
        self.activated = False

        variances = []
        mews = []
        if self.conf.flip_consistency and not self.conf.prior_uncertainty:  # just loop consistency
            assert "depth2" in depth_dict, "Second depth must be provided for loop consistency"
            mews.append((depth_dict["depth2"] + depth_dict["depth"]) / 2)
            variances.append((depth_dict["depth"] - depth_dict["depth2"]) ** 2)
        elif self.conf.flip_consistency:  # both loop consistency and prior uncertainty
            assert (
                "depth_variance" in depth_dict
            ), "Variance must be provided for prior uncertainty. Force reextraction of mono-prior"
            mews += [depth_dict["depth"], depth_dict["depth2"]]
            variances += [depth_dict["depth_variance"], depth_dict["depth_variance2"]]
        elif self.conf.prior_uncertainty:
            assert "depth_variance" in depth_dict, "Variance must be provided for prior uncertainty"
            mews.append(depth_dict["depth"])
            variances.append(depth_dict["depth_variance"])
        else:
            mews.append(depth_dict["depth"])
        valid_mask = depth_dict["depth"] > 0
        if "valid" in depth_dict:
            valid_mask *= depth_dict["valid"]
        if "valid2" in depth_dict:
            valid_mask *= depth_dict["valid2"]
        if self.conf.use_continuity:
            continuity_mask = get_continuity_mask(depth_dict["depth"])

            if "depth2" in depth_dict:
                continuity_mask *= get_continuity_mask(depth_dict["depth2"])
        if len(mews) > 1:
            self.data_prior = np.sum(mewi / (vari + 1e-6) for mewi, vari in zip(mews, variances)) / (
                np.sum(1 / (vari + 1e-6) for vari in variances) + 1e-6
            )
        else:
            self.data_prior = mews[0]

        if self.conf.depth_uncertainty is not None:
            new_var = []
            if self.conf.prior_uncertainty:
                for mewi, vari in zip(mews, variances):
                    new_var.append(
                        np.maximum(vari * self.conf.prior_std_multiplier**2, (mewi * self.conf.depth_uncertainty) ** 2)
                    )
                if len(new_var) > 1:
                    self.uncertainty = 1 / (np.sum(1 / (vari + 1e-6) for vari in new_var) + 1e-6)
                else:
                    self.uncertainty = new_var[0]
            else:
                self.uncertainty = (self.data_prior * self.conf.depth_uncertainty) ** 2
        elif self.conf.flip_consistency:
            self.uncertainty = (
                1 / (np.sum(1 / (vari + 1e-6) for vari in variances) + 1e-6)
            ) * self.conf.prior_std_multiplier**2
        elif self.conf.fixed_uncertainty:
            self.uncertainty = np.ones_like(mews[0]) * self.conf.fixed_uncertainty_val * self.conf.std_multiplier**2
        else:
            assert len(variances) == 1, "Only one variance is supported for now"
            self.uncertainty = variances[0]

        self.uncertainty = self.uncertainty.clip(
            self.conf.inherent_noise**2, self.conf.max_std if self.conf.max_std is None else self.conf.max_std**2
        )
        self.uncertainty = self.uncertainty * (self.conf.std_multiplier**2)

        if camera.int_height != self.data_prior.shape[0] or camera.int_width != self.data_prior.shape[1]:
            self.data_prior = cv2.resize(self.data_prior, (camera.int_width, camera.int_height))
            self.uncertainty = cv2.resize(self.uncertainty, (camera.int_width, camera.int_height))
            valid_mask = cv2.resize(valid_mask.astype(float), (camera.int_width, camera.int_height)) == 1
            if self.conf.use_continuity:
                self.continuity_mask = (
                    cv2.resize(continuity_mask.astype(float), (camera.int_width, camera.int_height)) == 1
                )

        if mask is not None and self.uncertainty is not None:
            if mask.shape != self.uncertainty.shape:
                mask = (
                    cv2.resize(mask.astype(np.float32), self.uncertainty.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    > 0.5
                )
            valid_mask *= mask
        self.uncertainty[~valid_mask] = 1e6
        self.valid = valid_mask
        zero_depth_mask = self.data_prior == 0
        self.data_prior[zero_depth_mask] = 0.1
        self.valid[zero_depth_mask] = False

        if self.conf.depth_lim is not None:
            self.valid[self.data_prior > self.conf.depth_lim] = False

        self.uncertainty_update = self.uncertainty_at_kps(kps)

    def reset(self):
        """Reset the depth class to its initial state."""
        self.data_prior /= self.scale
        self.uncertainty /= self.scale**2
        self.uncertainty_update = self.uncertainty_at_kps(self.kps)
        self.scale = 1
        self.shift = 0
        self.activated = False
        self.data = None
