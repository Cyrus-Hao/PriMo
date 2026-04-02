import numpy as np

from mpsfm.utils.geometry import project3D, unproject_depth_map_to_world


class DepthUtils:
    """Depth map utils mixin for reconstruction."""

    def _get_image_depth(self, imid):
        image = self.images.get(imid)
        if image is None:
            return None
        return getattr(image, "depth", None)

    def reproject_depth(self, imid1, imid2, cfw1=None, cfw2=None):
        """Reprojects depth from imid1 to imid2"""
        image1 = self.images[imid1]
        image2 = self.images[imid2]
        depth_obj1 = self._get_image_depth(imid1)
        depth_obj2 = self._get_image_depth(imid2)
        if depth_obj1 is None or depth_obj2 is None or depth_obj1.data is None or depth_obj2.data_prior is None:
            return None
        depth1 = depth_obj1.data.copy()
        depth2_shape = depth_obj2.data_prior.shape

        cam1 = self.cameras[self.images[imid1].camera_id]
        cam2 = self.cameras[self.images[imid2].camera_id]
        valid1_mask = np.ones(depth1.shape, dtype=bool)
        depth1[depth1 <= 0] = 0.1
        if cfw1 is None:
            cfw1 = image1.cam_from_world
        if cfw2 is None:
            cfw2 = image2.cam_from_world
        H1 = np.concatenate([cfw1.matrix(), np.array([[0, 0, 0, 1]])], axis=0)
        H2 = np.concatenate([cfw2.matrix(), np.array([[0, 0, 0, 1]])], axis=0)
        K1 = cam1.calibration_matrix()
        K1[0, :] *= cam1.sx
        K1[1, :] *= cam1.sy
        D1W = unproject_depth_map_to_world(depth1, K1, np.linalg.inv(H1), mask=valid1_mask)
        K2 = cam2.calibration_matrix()
        K2[0, :] *= cam2.sx
        K2[1, :] *= cam2.sy
        p2D12, depth12 = project3D(D1W, H2, K2)

        mask12 = (
            (p2D12[:, 0] >= 0)
            & ((p2D12[:, 0] + 0.5) < depth2_shape[1])
            & (p2D12[:, 1] >= 0)
            & ((p2D12[:, 1] + 0.5) < depth2_shape[0])
        )
        mask12 *= depth12 > 0
        out = {
            "depth1": depth1,
            "p2D12": p2D12.reshape(*(depth1.shape), 2),
            "depth12": depth12.reshape(depth1.shape),
            "mask12": mask12.reshape(depth1.shape),
            "valid1_mask": valid1_mask,
        }

        return out

    def activate_depths(self, imids):
        """Activates depths for images"""
        for imid in imids:
            depth = self._get_image_depth(imid)
            if depth is None:
                continue
            if not depth.activated:
                depth.activated = True
                depth.data = depth.data_prior.copy()

    def _rescale_prior(self, imid, shift, scale):
        """Rescales depth prior"""
        depth = self._get_image_depth(imid)
        if depth is None:
            return
        depth.data_prior = depth.data_prior * scale + shift
        depth.scale *= scale
        depth.shift = depth.shift * scale + shift
        depth.uncertainty *= scale**2

    def __rescale_update(self, imid, shift, scale, rescale_depth=False):
        """Rescales optimized depth"""
        depth = self._get_image_depth(imid)
        if depth is None:
            return
        if rescale_depth and depth.activated:
            depth.data = depth.data * scale + shift

        depth.uncertainty_update *= scale**2
        # 裁剪不确定性，避免多次重标定导致过度放大/缩小，引起深度项权重退化
        uu = depth.uncertainty_update
        # 在合理范围内裁剪（数值可根据数据集再调）
        depth.uncertainty_update = np.clip(uu, 1e-8, 1e4)

    def rescale_all(self, shift_scales):
        """Rescales all depths i.e. priors and optimized depths"""
        self.rescale_priors(shift_scales)
        self.rescale_depths(shift_scales)

    def rescale_depths(self, shift_scales):
        """Rescales optimized depths"""
        for imid, (shift, scale) in shift_scales.items():
            self.__rescale_update(imid, shift, scale)

    def rescale_priors(self, shift_scales):
        """Rescales priors"""
        for imid, (shift, scale) in shift_scales.items():
            self._rescale_prior(imid, shift, scale)

    def normalize_depths(self, scale):
        """Normalizes depths"""
        for imid in self.images:
            self._rescale_prior(imid, 0, scale)
            self.__rescale_update(imid, 0, scale, rescale_depth=True)
