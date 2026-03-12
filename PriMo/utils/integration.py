"""This module contains utils for integrating in MPSFM. Mostly copied from BNI."""

import torch


def setup_matrix_library(device="cuda"):
    """Set up the matrix library based on the device. Created this way because cupy
    does not have flexible device management."""
    if device == "cuda":
        import cupy as cp
        import scipy.sparse as sp
        from cupyx.scipy.sparse import csr_matrix, diags, identity
        from cupyx.scipy.sparse.linalg import cg

        pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        cp.cuda.set_allocator(pool.malloc)
    else:
        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse import csr_matrix, diags, identity
        from scipy.sparse.linalg import cg

        cp = np
        cp.asnumpy = np.array

    return cp, csr_matrix, cg, identity, diags, sp


device_g = "cuda" if torch.cuda.is_available() else "cpu"
cp, csr_matrix, *_ = setup_matrix_library(device_g)


def move_left(mask):
    """Shift the input mask array to the left by 1, filling the right edge with zeros."""
    return cp.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]


def move_right(mask):
    """Shift the input mask array to the right by 1, filling the left edge with zeros."""
    return cp.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]


def move_top(mask):
    """Shift the input mask array up by 1, filling the bottom edge with zeros."""
    return cp.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]


def move_bottom(mask):
    """Shift the input mask array down by 1, filling the top edge with zeros."""
    return cp.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]


def sigmoid(x, k=1):
    """Sigmoid function with a scaling factor k."""
    cc = cp.clip(-k * x, -709, 709)
    return 1 / (1 + cp.exp(cc))
