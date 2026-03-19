from pathlib import Path

import numpy as np
import yaml


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def canonical_pose_name(image_name: str) -> str:
    stem = Path(image_name).name
    suffix = Path(stem).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return Path(stem).stem
    return stem


def pose_name_variants(image_name: str) -> list[str]:
    return [canonical_pose_name(image_name)]


def orthogonalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def camera_center(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return -rotation.T @ translation


def load_prior_pose_arrays(pose_config_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    pose_config_path = Path(pose_config_path)
    if not pose_config_path.exists():
        raise FileNotFoundError(f"Pose config file not found: {pose_config_path}")

    with pose_config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    camera_poses = config.get("camera_poses", {}) or {}
    if not camera_poses:
        raise ValueError(f"No camera_poses found in {pose_config_path}")

    poses = {}
    for image_name, pose_data in camera_poses.items():
        if "transform_matrix" not in pose_data:
            raise ValueError(f"Missing transform_matrix for {image_name} in {pose_config_path}")

        transform_matrix = np.asarray(pose_data["transform_matrix"], dtype=np.float64)
        if transform_matrix.shape != (4, 4):
            raise ValueError(f"transform_matrix for {image_name} is not 4x4")

        rotation = orthogonalize_rotation(transform_matrix[:3, :3])
        translation = transform_matrix[:3, 3]
        center = camera_center(rotation, translation)
        poses.setdefault(canonical_pose_name(image_name), (rotation, translation, center))

    return poses


def lookup_prior_pose_entry(
    poses: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], image_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for variant in pose_name_variants(image_name):
        entry = poses.get(variant)
        if entry is not None:
            return entry
    return None
