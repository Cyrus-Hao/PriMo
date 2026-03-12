"""Utility functions for 3D geometry operations."""

import numpy as np


def project3D_colmap(image, camera, points3D):
    """Project 3D points into image using colmap camera."""
    H = np.concatenate([image.cam_from_world.matrix(), np.array([[0, 0, 0, 1]])], axis=0)
    K = camera.calibration_matrix()
    return project3D(points3D, H, K)


def project3D(points3D, H, K):
    """Project 3D points into image using homography and camera matrix."""
    points3D_h = np.hstack([points3D, np.ones((points3D.shape[0], 1))])
    points_cam = (H @ points3D_h.T)[:3, :].T
    depth = points_cam[:, 2].copy()
    pts = ((K @ (points_cam / depth[:, None]).T).T)[:, :2]
    return pts, depth


def unproject_depth_map_to_world(depth, K, H, mask=None):
    """Unproject depth map to 3D points in world coordinates."""
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()
    if mask is not None:
        mask = mask.flatten()
        x = x[mask]
        y = y[mask]
        depth = depth[mask]

    xy_depth = np.vstack((x * depth, y * depth, depth))
    points_3d = unproject_to_world(xy_depth, K, H)
    return points_3d


def unproject_to_world(xy_depth, K, H):
    """Unproject 2D points with depth to 3D points in world coordinates."""
    points_3d = unproject_to_cam(xy_depth, K)
    points_3d = (H @ points_3d.T).T[:, :3]
    return points_3d


def unproject_to_cam(xy_depth, K):
    """Unproject 2D points with depth to 3D points in camera coordinates."""
    points_3d = np.linalg.inv(K) @ xy_depth
    points_3d = np.vstack((points_3d, np.ones((1, points_3d.shape[1]))))
    return points_3d.T


def calculate_triangulation_angle(proj_center1, proj_center2, point3D):
    """Calculate the angle between two rays from the projection centers to a 3D point."""
    # REIMPLEMENTING: CalculateTriangulationAngle
    baseline_length_squared = np.linalg.norm(proj_center1 - proj_center2)
    ray_length_squared1 = np.linalg.norm(point3D - proj_center1)
    ray_length_squared2 = np.linalg.norm(point3D - proj_center2)
    denominator = 2.0 * np.sqrt(ray_length_squared1 * ray_length_squared2)
    if denominator == 0.0:  # FIXME: why did i put this here? but i guess it should work
        return 0.0
    nominator = ray_length_squared1 + ray_length_squared2 - baseline_length_squared
    angle = np.abs(np.arccos(nominator / denominator))
    return min(angle, np.pi - angle)


def has_point_positive_depth(cam_from_world, point3D, return_depth=False):
    """Check if a 3D point has positive depth in the camera coordinate system."""
    point3D_homogeneous = np.append(point3D, 1)
    third_row = cam_from_world[2, :]
    depth = np.dot(third_row, point3D_homogeneous)
    if return_depth:
        return depth >= np.finfo(float).eps, depth
    return depth >= np.finfo(float).eps
