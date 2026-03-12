from dataclasses import dataclass, field

import torch

from mpsfm.utils.integration import (
    move_bottom,
    move_left,
    move_right,
    move_top,
    setup_matrix_library,
)

device_g = "cuda" if torch.cuda.is_available() else "cpu"
cp, csr_matrix, cg, identity, diags, sp = setup_matrix_library(device=device_g)


class ColmapCameraWrapper:
    def __getattr__(self, name):
        """Delegate attribute access to the original camera, except for special cases."""
        return getattr(self._camera, name)

    def __setattr__(self, name, value):
        """Set attributes on the original camera if they exist, otherwise on the wrapper."""
        if name == "_camera":
            super().__setattr__(name, value)  # Avoid recursion during initialization
        elif hasattr(self._camera, name):  # Delegate to the original camera
            setattr(self._camera, name, value)
        else:  # Set on the wrapper itself
            super().__setattr__(name, value)

    def as_colmap(self):
        """Return the underlying pycolmap camera."""
        return self._camera


@dataclass
class CameraIntData:
    """Camera data for MP-SfM integration."""

    int_height: int
    int_width: int
    nshape: tuple[int, int] = field(init=False)
    num_normals: int = field(init=False)
    normal_mask: cp.ndarray = field(init=False, repr=False)

    has_left_mask: cp.ndarray = field(init=False, repr=False)
    has_left_mask_left: cp.ndarray = field(init=False, repr=False)
    has_right_mask: cp.ndarray = field(init=False, repr=False)
    has_right_mask_right: cp.ndarray = field(init=False, repr=False)
    has_bottom_mask: cp.ndarray = field(init=False, repr=False)
    has_bottom_mask_bottom: cp.ndarray = field(init=False, repr=False)
    has_top_mask: cp.ndarray = field(init=False, repr=False)
    has_top_mask_top: cp.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.nshape = (self.int_height, self.int_width)
        self.num_normals = self.int_height * self.int_width
        normal_mask = cp.ones((self.int_height, self.int_width), bool)

        self.has_left_mask = cp.logical_and(move_right(normal_mask), normal_mask)
        self.has_left_mask_left = move_left(self.has_left_mask)
        self.has_right_mask = cp.logical_and(move_left(normal_mask), normal_mask)
        self.has_right_mask_right = move_right(self.has_right_mask)
        self.has_bottom_mask = cp.logical_and(move_top(normal_mask), normal_mask)
        self.has_bottom_mask_bottom = move_bottom(self.has_bottom_mask)
        self.has_top_mask = cp.logical_and(move_bottom(normal_mask), normal_mask)
        self.has_top_mask_top = move_top(self.has_top_mask)

        self.update_flat_masks(normal_mask)
        self.calculate_pixel_indices()

    def update_flat_masks(self, normal_mask):
        self.has_left_mask_flat = self.has_left_mask[normal_mask]
        self.has_right_mask_flat = self.has_right_mask[normal_mask]
        self.has_bottom_mask_flat = self.has_bottom_mask[normal_mask]
        self.has_top_mask_flat = self.has_top_mask[normal_mask]

        self.has_left_mask_left_flat = self.has_left_mask_left[normal_mask]
        self.has_right_mask_right_flat = self.has_right_mask_right[normal_mask]
        self.has_bottom_mask_bottom_flat = self.has_bottom_mask_bottom[normal_mask]
        self.has_top_mask_top_flat = self.has_top_mask_top[normal_mask]

    def calculate_pixel_indices(self):
        self.pixel_idx = cp.arange(self.num_normals).reshape((self.int_height, self.int_width))
        self.pixel_idx_flat = cp.arange(self.num_normals)
        self.pixel_idx_flat_indptr = cp.arange(self.num_normals + 1)

        self.pixel_idx_left_center = self.pixel_idx[self.has_left_mask]
        self.pixel_idx_right_right = self.pixel_idx[self.has_right_mask_right]
        self.pixel_idx_top_center = self.pixel_idx[self.has_top_mask]
        self.pixel_idx_bottom_bottom = self.pixel_idx[self.has_bottom_mask_bottom]

        self.pixel_idx_left_left_indptr = cp.concatenate([cp.array([0]), cp.cumsum(self.has_left_mask_left_flat)])
        self.pixel_idx_right_center_indptr = cp.concatenate([cp.array([0]), cp.cumsum(self.has_right_mask_flat)])
        self.pixel_idx_top_top_indptr = cp.concatenate([cp.array([0]), cp.cumsum(self.has_top_mask_top_flat)])
        self.pixel_idx_bottom_center_indptr = cp.concatenate([cp.array([0]), cp.cumsum(self.has_bottom_mask_flat)])


class Camera(ColmapCameraWrapper, CameraIntData):
    """MP-SfM camera wrapper for COLMAP camera."""

    def __init__(self, camera):
        self._camera = camera

    def init_int_data(self, H, W):
        CameraIntData.__init__(self, H, W)

    def __repr__(self):
        base = CameraIntData.__repr__(self)
        colmap = repr(self._camera).replace("\n", "\n  ")
        return f"{base}\n  [pycolmap] {colmap}"
