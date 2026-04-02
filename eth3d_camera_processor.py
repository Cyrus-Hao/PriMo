import argparse
import math
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local" / "example"


def save_yaml(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, allow_unicode=True, sort_keys=False)


def skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def axis_angle_to_rotation(axis_angle: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    axis = axis_angle / theta
    axis_skew = skew(axis)
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) * axis_skew
        + (1.0 - math.cos(theta)) * (axis_skew @ axis_skew)
    )


def sample_unit_vector(rng: np.random.Generator) -> np.ndarray:
    while True:
        vec = rng.normal(size=3)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            return vec / norm


def orthogonalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def qvec_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = map(float, qvec)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        raise ValueError("qvec 范数过小，无法转换为旋转矩阵")

    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def camera_center_from_cam_from_world(rotation_cw: np.ndarray, translation_cw: np.ndarray) -> np.ndarray:
    return -rotation_cw.T @ translation_cw


def cam_from_world_from_center(rotation_cw: np.ndarray, camera_center_w: np.ndarray) -> np.ndarray:
    translation_cw = -rotation_cw @ camera_center_w
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_cw
    transform[:3, 3] = translation_cw
    return transform


def compute_translation_scale(camera_centers: np.ndarray, mode: str) -> float:
    if camera_centers.shape[0] <= 1:
        return 1.0

    if mode == "bbox_diagonal":
        return float(np.linalg.norm(camera_centers.max(axis=0) - camera_centers.min(axis=0)))

    if mode == "centroid_rms":
        centered = camera_centers - camera_centers.mean(axis=0, keepdims=True)
        return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))

    if mode == "path_length":
        diffs = camera_centers[1:] - camera_centers[:-1]
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    raise ValueError(f"未知的 translation scale 模式: {mode}")


def parse_cameras_txt(cameras_txt_path: Path) -> dict[int, dict]:
    cameras = {}
    with Path(cameras_txt_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"非法 cameras.txt 行: {line}")

            camera_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[camera_id] = {
                "model_name": model_name,
                "width": width,
                "height": height,
                "params": params,
            }
    if not cameras:
        raise ValueError(f"未在 {cameras_txt_path} 中解析到任何相机")
    return cameras


def parse_images_txt(images_txt_path: Path) -> list[dict]:
    image_records = []
    with Path(images_txt_path).open("r", encoding="utf-8") as handle:
        valid_lines = [line.rstrip("\n") for line in handle if line.strip() and not line.lstrip().startswith("#")]

    if len(valid_lines) % 2 != 0:
        raise ValueError(f"{images_txt_path} 中的有效行数不是偶数，无法按 COLMAP images.txt 双行格式解析")

    for idx in range(0, len(valid_lines), 2):
        parts = valid_lines[idx].split()
        if len(parts) < 10:
            raise ValueError(f"非法 images.txt pose 行: {valid_lines[idx]}")

        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        camera_id = int(parts[8])
        image_name = " ".join(parts[9:])
        image_records.append(
            {
                "image_id": image_id,
                "image_name": image_name,
                "camera_id": camera_id,
                "rotation_cw": qvec_to_rotation_matrix(qvec),
                "translation_cw": tvec,
            }
        )

    if not image_records:
        raise ValueError(f"未在 {images_txt_path} 中解析到任何图像位姿")
    return image_records


def parse_camera_params(camera: dict) -> tuple[float, float, float, float]:
    params = list(map(float, camera["params"]))
    model_name = camera["model_name"]

    if model_name in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy = params[:3]
        return f, f, cx, cy

    if model_name in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy

    raise ValueError(
        f"暂不支持相机模型 {model_name}。"
        "请先在 COLMAP 中导出为去畸变后的 PINHOLE / SIMPLE_PINHOLE 相机，"
        "或自行在脚本中补充该模型解析。"
    )


def format_pose_key(image_name: str, mode: str) -> str:
    image_path = Path(image_name)
    if mode == "stem":
        return f"images/{image_path.stem}"
    if mode == "relative":
        return str(image_path.with_suffix(""))
    raise ValueError(f"未知的 pose name 模式: {mode}")


def format_intrinsics_image_name(image_name: str, mode: str) -> str:
    image_path = Path(image_name)
    if mode == "basename":
        return image_path.name
    if mode == "relative":
        return image_name
    raise ValueError(f"未知的 intrinsics image 模式: {mode}")


def sample_rotation_noise(rng: np.random.Generator, rot_noise_deg: float, noise_type: str) -> np.ndarray:
    if rot_noise_deg <= 0:
        return np.eye(3, dtype=np.float64)

    rot_noise_rad = math.radians(rot_noise_deg)
    if noise_type == "gaussian":
        axis_angle = rng.normal(loc=0.0, scale=rot_noise_rad, size=3)
    elif noise_type == "uniform":
        axis_angle = rng.uniform(low=-rot_noise_rad, high=rot_noise_rad, size=3)
    elif noise_type == "fixed":
        axis_angle = sample_unit_vector(rng) * rot_noise_rad
    else:
        raise ValueError(f"未知的噪声类型: {noise_type}")

    return axis_angle_to_rotation(axis_angle)


def sample_translation_noise(
    rng: np.random.Generator,
    trans_noise_ratio: float,
    translation_scale: float,
    noise_type: str,
) -> np.ndarray:
    if trans_noise_ratio <= 0:
        return np.zeros(3, dtype=np.float64)

    sigma = float(trans_noise_ratio) * float(translation_scale)
    if noise_type == "gaussian":
        return rng.normal(loc=0.0, scale=sigma, size=3)
    if noise_type == "uniform":
        return rng.uniform(low=-sigma, high=sigma, size=3)
    if noise_type == "fixed":
        return sample_unit_vector(rng) * sigma
    raise ValueError(f"未知的噪声类型: {noise_type}")


def build_primo_yaml_from_colmap(
    model_dir: Path,
    rot_noise_deg: float,
    trans_noise_ratio: float,
    seed: int,
    noise_type: str,
    translation_scale_mode: str,
    pose_name_mode: str,
    intrinsics_image_mode: str,
) -> tuple[dict, dict, float]:
    cameras_txt_path = Path(model_dir) / "cameras.txt"
    images_txt_path = Path(model_dir) / "images.txt"
    if not cameras_txt_path.exists() or not images_txt_path.exists():
        raise FileNotFoundError(
            f"当前脚本仅支持 COLMAP text model，需要同时存在 cameras.txt 和 images.txt: {model_dir}"
        )

    cameras = parse_cameras_txt(cameras_txt_path)
    image_items = sorted(
        parse_images_txt(images_txt_path),
        key=lambda item: (Path(item["image_name"]).name, int(item["image_id"])),
    )

    camera_centers = []
    image_records = []
    for image in image_items:
        rotation_cw = image["rotation_cw"]
        translation_cw = image["translation_cw"]
        camera_center_w = camera_center_from_cam_from_world(rotation_cw, translation_cw)
        camera_centers.append(camera_center_w)
        image_records.append(
            {
                "image_name": image["image_name"],
                "camera_id": int(image["camera_id"]),
                "rotation_cw": rotation_cw,
                "camera_center_w": camera_center_w,
            }
        )

    camera_centers = np.stack(camera_centers, axis=0)
    translation_scale = compute_translation_scale(camera_centers, translation_scale_mode)
    if translation_scale < 1e-12:
        translation_scale = 1.0

    rng = np.random.default_rng(seed)
    camera_poses = {"camera_poses": {}}
    intrinsics = {}

    for frame_idx, record in enumerate(image_records, start=1):
        camera = cameras[record["camera_id"]]
        fx, fy, cx, cy = parse_camera_params(camera)

        delta_rot = sample_rotation_noise(rng, rot_noise_deg, noise_type)
        delta_center = sample_translation_noise(rng, trans_noise_ratio, translation_scale, noise_type)

        noisy_rotation_cw = orthogonalize_rotation(delta_rot @ record["rotation_cw"])
        noisy_camera_center_w = record["camera_center_w"] + delta_center
        noisy_transform = cam_from_world_from_center(noisy_rotation_cw, noisy_camera_center_w)

        pose_key = format_pose_key(record["image_name"], pose_name_mode)
        intrinsics_image_name = format_intrinsics_image_name(record["image_name"], intrinsics_image_mode)

        camera_poses["camera_poses"][pose_key] = {
            "transform_matrix": noisy_transform.tolist(),
        }
        intrinsics[frame_idx] = {
            "params": [float(fx), float(fy), float(cx), float(cy)],
            "images": [intrinsics_image_name],
        }

    return camera_poses, intrinsics, translation_scale


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 ETH3D / COLMAP GT 位姿转换为 PriMo 的 camera_poses.yaml 和 intrinsics.yaml，并可注入可控噪声。"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="COLMAP 模型目录，支持 pycolmap 可读取的 text/bin sparse model。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录，默认写到 PriMo/local/example。",
    )
    parser.add_argument(
        "--rot-noise-deg",
        type=float,
        default=3.0,
        help="旋转噪声强度，单位度。0 表示不加旋转噪声。",
    )
    parser.add_argument(
        "--trans-noise-ratio",
        type=float,
        default=0.05,
        help="平移噪声强度，占轨迹尺度的比例。0.05 对应 5%%。",
    )
    parser.add_argument(
        "--noise-type",
        choices=("fixed", "gaussian", "uniform"),
        default="fixed",
        help="噪声类型。fixed 表示固定旋转角和固定平移长度；gaussian / uniform 表示按分量采样。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子，保证可复现。",
    )
    parser.add_argument(
        "--translation-scale-mode",
        choices=("bbox_diagonal", "centroid_rms", "path_length"),
        default="bbox_diagonal",
        help="把 trans_noise_ratio 映射到实际米制噪声时使用的轨迹尺度定义。",
    )
    parser.add_argument(
        "--pose-name-mode",
        choices=("stem", "relative"),
        default="stem",
        help="camera_poses.yaml 中图像键名的生成方式。默认 stem 会生成 images/000123 这种键名。",
    )
    parser.add_argument(
        "--intrinsics-image-mode",
        choices=("basename", "relative"),
        default="basename",
        help="intrinsics.yaml 中 images 字段的命名方式。默认只写文件名。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    camera_poses, intrinsics, translation_scale = build_primo_yaml_from_colmap(
        model_dir=args.model_dir,
        rot_noise_deg=args.rot_noise_deg,
        trans_noise_ratio=args.trans_noise_ratio,
        seed=args.seed,
        noise_type=args.noise_type,
        translation_scale_mode=args.translation_scale_mode,
        pose_name_mode=args.pose_name_mode,
        intrinsics_image_mode=args.intrinsics_image_mode,
    )

    pose_output_path = args.output_dir / "camera_poses.yaml"
    intrinsics_output_path = args.output_dir / "intrinsics.yaml"
    save_yaml(camera_poses, pose_output_path)
    save_yaml(intrinsics, intrinsics_output_path)

    print(f"已保存 noisy priors 到: {pose_output_path}")
    print(f"已保存 intrinsics 到: {intrinsics_output_path}")
    print(
        "噪声设置: "
        f"rotation={args.rot_noise_deg} deg, "
        f"translation_ratio={args.trans_noise_ratio}, "
        f"noise_type={args.noise_type}, "
        f"translation_scale({args.translation_scale_mode})={translation_scale:.6f}"
    )
    print(f"图像数量: {len(intrinsics)}")


if __name__ == "__main__":
    main()
