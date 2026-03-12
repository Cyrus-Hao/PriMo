from pathlib import Path

import numpy as np
import yaml

from mpsfm.utils.parsers import parse_image_lists

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _pose_name_variants(image_name: str) -> list[str]:
    stem = Path(image_name).name
    suffix = Path(stem).suffix.lower()
    variants = [stem]
    if suffix in _IMAGE_SUFFIXES:
        variants.append(Path(stem).stem)
    else:
        variants.append(f"{stem}.png")
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(variants))


def _orthogonalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def _camera_center(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return -rot.T @ trans


def _coerce_names(names):
    if names is None:
        return None
    if isinstance(names, (list, tuple)):
        return list(names)
    if isinstance(names, (str, Path)):
        return parse_image_lists(Path(names))
    raise ValueError(f"Unknown type for image list: {type(names)}")


def _load_prior_poses(pose_config_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    pose_config_path = Path(pose_config_path)
    if not pose_config_path.exists():
        raise FileNotFoundError(f"Pose config file not found: {pose_config_path}")

    with open(pose_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    camera_poses = config.get("camera_poses", {}) or {}
    if len(camera_poses) == 0:
        raise ValueError(f"No camera_poses found in {pose_config_path}")

    poses = {}
    for image_name, pose_data in camera_poses.items():
        if "transform_matrix" not in pose_data:
            raise ValueError(f"Missing transform_matrix for {image_name} in {pose_config_path}")
        transform_matrix = np.asarray(pose_data["transform_matrix"], dtype=np.float64)
        if transform_matrix.shape != (4, 4):
            raise ValueError(f"transform_matrix for {image_name} is not 4x4")
        rotation = _orthogonalize_rotation(transform_matrix[:3, :3])
        translation = transform_matrix[:3, 3]
        center = _camera_center(rotation, translation)
        for variant in _pose_name_variants(image_name):
            poses.setdefault(variant, (rotation, translation, center))

    return poses


def pairs_from_prior_pose(
    output: Path,
    num_matched: int,
    pose_config_path: Path,
    image_list=None,
    query_list=None,
    db_list=None,
    candidate_multiplier: float = 3.0,
    verbose: int = 0,
):
    if num_matched <= 0:
        raise ValueError("num_matched must be positive")
    if image_list is None and (query_list is None or db_list is None):
        raise ValueError("Provide image_list or both query_list and db_list")

    if query_list is None:
        query_list = image_list
    if db_list is None:
        db_list = image_list

    query_names = _coerce_names(query_list)
    db_names = _coerce_names(db_list)

    poses = _load_prior_poses(pose_config_path)

    db_entries = []
    missing_db = 0
    for name in db_names:
        entry = None
        for variant in _pose_name_variants(name):
            if variant in poses:
                entry = poses[variant]
                break
        if entry is None:
            missing_db += 1
            continue
        rot, trans, center = entry
        db_entries.append((name, rot, trans, center))

    if len(db_entries) == 0:
        raise ValueError("No database images have valid prior poses")

    db_names_valid = [x[0] for x in db_entries]
    db_rot = np.stack([x[1] for x in db_entries], axis=0)
    db_center = np.stack([x[3] for x in db_entries], axis=0)

    pairs = []
    missing_query = 0
    for q_name in query_names:
        entry = None
        for variant in _pose_name_variants(q_name):
            if variant in poses:
                entry = poses[variant]
                break
        if entry is None:
            missing_query += 1
            continue
        rot_q, _, center_q = entry

        rot_rel = rot_q @ np.transpose(db_rot, (0, 2, 1))
        traces = np.einsum("nii->n", rot_rel)
        cos_theta = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
        angles = np.arccos(cos_theta)

        mask = np.array([name != q_name for name in db_names_valid], dtype=bool)
        if not np.any(mask):
            continue
        angles = angles[mask]
        db_names_filtered = [n for n, m in zip(db_names_valid, mask) if m]
        db_center_filtered = db_center[mask]

        pool_size = max(num_matched, int(np.ceil(num_matched * candidate_multiplier)))
        pool_size = min(pool_size, len(angles))
        rot_rank = np.argsort(angles)[:pool_size]
        trans_dist = np.linalg.norm(db_center_filtered[rot_rank] - center_q, axis=1)
        trans_rank = rot_rank[np.argsort(trans_dist)[:num_matched]]

        for idx in trans_rank:
            pairs.append((q_name, db_names_filtered[idx]))

    if verbose > 0:
        print(f"Found {len(pairs)} pairs from prior poses.")
        print(f"Missing query poses: {missing_query}, missing db poses: {missing_db}")
        print(f"Writing pairs to {output}.")

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))

    return pairs

