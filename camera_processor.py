import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE_DIR = PROJECT_ROOT / "local" / "example" / "sofa"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local" / "example"


def convert_jpg_to_png(images_dir: Path) -> int:
    """Convert all JPG/JPEG images in-place to PNG."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return 0

    jpg_files = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
        for path in images_dir.glob(pattern)
    )
    if not jpg_files:
        return 0

    print(f"检测到 {len(jpg_files)} 个 JPG/JPEG 图片，开始转换为 PNG...")
    converted = 0
    for jpg_file in jpg_files:
        try:
            png_file = jpg_file.with_suffix(".png")
            with Image.open(jpg_file) as image:
                image.save(png_file, "PNG")
            jpg_file.unlink()
            print(f"  ✓ {jpg_file.name} -> {png_file.name}")
            converted += 1
        except Exception as exc:
            print(f"  ✗ 转换失败 {jpg_file.name}: {exc}")

    print(f"成功转换 {converted} 个图片为 PNG 格式")
    return converted


def read_camera_matrix(csv_path: Path) -> np.ndarray:
    """Read a 3x3 camera intrinsic matrix from CSV."""
    try:
        with Path(csv_path).open("r", encoding="utf-8") as handle:
            rows = [[float(value.strip()) for value in row] for row in csv.reader(handle)]
        matrix = np.array(rows, dtype=np.float64)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"找不到相机内参文件: {csv_path}") from exc
    except Exception as exc:
        raise ValueError(f"读取相机内参文件失败: {csv_path}") from exc

    if matrix.shape != (3, 3):
        raise ValueError(f"相机内参矩阵应为 3x3，实际为 {matrix.shape}")
    return matrix


def read_odometry_data(csv_path: Path) -> list[dict]:
    """Read odometry CSV rows with pose and frame information."""
    try:
        with Path(csv_path).open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            return [
                {
                    "timestamp": float(row["timestamp"]),
                    "frame": row["frame"].strip(),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "qx": float(row["qx"]),
                    "qy": float(row["qy"]),
                    "qz": float(row["qz"]),
                    "qw": float(row["qw"]),
                }
                for row in reader
            ]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"找不到里程计文件: {csv_path}") from exc
    except Exception as exc:
        raise ValueError(f"读取里程计文件失败: {csv_path}") from exc


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to a rotation matrix."""
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("四元数范数为 0，无法转换为旋转矩阵")
    qx, qy, qz, qw = quat / norm
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def odometry_to_cam_from_world(odom: dict) -> list[list[float]]:
    """Build a COLMAP-style camera-from-world transform from odometry."""
    rotation_world_from_cam = quaternion_to_rotation_matrix(odom["qx"], odom["qy"], odom["qz"], odom["qw"])
    translation_world_from_cam = np.array([odom["x"], odom["y"], odom["z"]], dtype=np.float64)

    rotation_cam_from_world = rotation_world_from_cam.T
    translation_cam_from_world = -rotation_cam_from_world @ translation_world_from_cam

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_cam_from_world
    transform[:3, 3] = translation_cam_from_world
    return transform.tolist()


def process_csv_data(camera_matrix_csv: Path, odometry_csv: Path, sample_interval: int = 1):
    """Generate camera poses and intrinsics YAML payloads from CSV files."""
    if sample_interval <= 0:
        raise ValueError("sample_interval 必须大于 0")

    camera_matrix = read_camera_matrix(camera_matrix_csv)
    odometry_data = read_odometry_data(odometry_csv)
    if not odometry_data:
        raise ValueError("里程计文件为空")

    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])

    camera_poses = {"camera_poses": {}}
    intrinsics = {}

    for frame_idx, odom in enumerate(odometry_data[::sample_interval], start=1):
        frame_name = odom["frame"]
        camera_poses["camera_poses"][f"images/{frame_name}"] = {
            "transform_matrix": odometry_to_cam_from_world(odom)
        }
        intrinsics[frame_idx] = {
            "params": [fx, fy, cx, cy],
            "images": [f"{frame_name}.png"],
        }

    return camera_poses, intrinsics


def save_yaml(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, allow_unicode=True, sort_keys=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PriMo camera pose and intrinsics YAML from CSV files.")
    parser.add_argument("--camera-matrix", type=Path, default=DEFAULT_SCENE_DIR / "camera_matrix.csv")
    parser.add_argument("--odometry", type=Path, default=DEFAULT_SCENE_DIR / "odometry.csv")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "images")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-interval", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    convert_jpg_to_png(args.images_dir)
    camera_poses, intrinsics = process_csv_data(
        camera_matrix_csv=args.camera_matrix,
        odometry_csv=args.odometry,
        sample_interval=args.sample_interval,
    )
    save_yaml(camera_poses, args.output_dir / "camera_poses.yaml")
    save_yaml(intrinsics, args.output_dir / "intrinsics.yaml")
    print(f"已保存到: {args.output_dir}，相机数: {len(intrinsics)}")


if __name__ == "__main__":
    main()