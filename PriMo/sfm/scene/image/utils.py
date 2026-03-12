"""Utility functions for depth maps."""

import numpy as np


def invert_depth(depth: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Invert depth to get inverse depth."""
    inverse_depth = 1.0 / depth.clip(min=eps)
    return inverse_depth


def fgbg_depth(d: np.ndarray, t: float):
    """Compute the foreground-background depth mask."""
    right_is_big_enough = (d[..., :, 1:] / d[..., :, :-1]) > t
    left_is_big_enough = (d[..., :, :-1] / d[..., :, 1:]) > t
    bottom_is_big_enough = (d[..., 1:, :] / d[..., :-1, :]) > t
    top_is_big_enough = (d[..., :-1, :] / d[..., 1:, :]) > t
    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )


def get_continuity_mask(depth: np.ndarray) -> np.ndarray:
    """Compute the continuity mask for depth."""
    continuity = np.ones_like(depth, dtype=bool)
    inv_depth1 = invert_depth(depth)
    l1, t1, r1, b1 = [~el for el in fgbg_depth(inv_depth1, 1.015)]

    continuity[:, 1:] &= l1 & r1
    continuity[:, :-1] &= l1 & r1
    continuity[1:, :] &= t1 & b1
    continuity[:-1, :] &= t1 & b1
    return continuity
