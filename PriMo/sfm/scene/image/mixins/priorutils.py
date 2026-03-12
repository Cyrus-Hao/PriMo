import numpy as np
import torch
from torch.nn import functional as F


class PriorUtils:
    data = None
    data_prior = None
    uncertainty = None
    image_size = None

    camera: any
    valid: np.ndarray

    @classmethod
    def init_empty(cls):
        """Initialize empty instance of PriorUtils"""
        instance = cls.__new__(cls)
        return instance

    def data_prior_at_kps(self, kps):
        """
        Get data_prior at keypoints
        kps: keypoints in the form of (N, 2) numpy array with original scale
        """
        return self._data_at_kps(kps, self.data_prior)

    def data_at_kps(self, kps):
        """
        Get optimized data at keypoints
        kps: keypoints in the form of (N, 2) numpy array with original scale
        """
        return self._data_at_kps(kps, self.data)

    def uncertainty_at_kps(self, kps):
        """
        Get prior uncertainty at keypoints
        kps: keypoints in the form of (N, 2) numpy array with original scale
        """
        return self._data_at_kps(kps, self.uncertainty)

    def valid_at_kps(self, kps):
        """
        Get valid mask at keypoints
        kps: keypoints in the form of (N, 2) numpy array with original scale
        """
        return self._data_at_kps(kps, self.valid) == 1

    def _data_at_kps(self, kps, data, mode="bilinear"):
        kp_t = torch.tensor(kps * np.array([self.camera.sx, self.camera.sy]), dtype=torch.float64)
        if len(kp_t.shape) == 1:
            kp_t = kp_t[None]
        H, W = data.shape[:2]
        data_map_t = torch.tensor(data, dtype=torch.float64)[None, None]
        kp_t[:, 0] = (kp_t[:, 0] / (W - 1)) * 2 - 1
        kp_t[:, 1] = (kp_t[:, 1] / (H - 1)) * 2 - 1
        kp_t = kp_t[None, None].permute(0, 1, 2, 3)
        sampled_data = F.grid_sample(data_map_t, kp_t, mode=mode, padding_mode="zeros", align_corners=True)[
            0, 0, 0
        ].numpy()

        return sampled_data
