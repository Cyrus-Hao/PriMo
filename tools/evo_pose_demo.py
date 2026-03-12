#!/usr/bin/env python3
import argparse
import csv
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from evo.core import metrics
    from evo.core import sync
    from evo.core import trajectory
    from evo.core import transformations
    from evo.tools import plot
except ImportError as exc:
    raise SystemExit(
        "需要先安装 evo: pip install evo --upgrade"
    ) from exc


CAMERA_MODEL_PARAMS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}

CAMERA_MODEL_ID_MAP = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def read_cameras_bin(path: Path):
    cameras = {}
    with path.open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id = read_next_bytes(fid, 4, "I")[0]
            model_id = read_next_bytes(fid, 4, "i")[0]
            model = CAMERA_MODEL_ID_MAP.get(model_id, f"UNKNOWN_{model_id}")
            width = read_next_bytes(fid, 8, "Q")[0]
            height = read_next_bytes(fid, 8, "Q")[0]
            num_params = CAMERA_MODEL_PARAMS.get(model, 0)
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=model,
                width=width,
                height=height,
                params=np.array(params, dtype=float),
            )
    return cameras


def read_images_bin(path: Path):
    images = {}
    with path.open("rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "I")[0]
            qvec = np.array(read_next_bytes(fid, 8 * 4, "dddd"), dtype=float)
            tvec = np.array(read_next_bytes(fid, 8 * 3, "ddd"), dtype=float)
            camera_id = read_next_bytes(fid, 4, "I")[0]

            name_chars = []
            while True:
                ch = fid.read(1)
                if ch == b"\x00" or ch == b"":
                    break
                name_chars.append(ch.decode("utf-8", errors="ignore"))
            name = "".join(name_chars)

            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points2d * (8 * 2 + 8))
            images[image_id] = ColmapImage(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
            )
    return images


def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=float,
    )


def rotmat_to_quat_wxyz(rot):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rot
    quat = transformations.quaternion_from_matrix(transform)
    return quat / np.linalg.norm(quat)


def normalize_frame_id(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return text


def extract_frame_id(name):
    match = re.search(r"(\d+)", name)
    if match:
        return normalize_frame_id(match.group(1))
    return normalize_frame_id(Path(name).stem)


def build_colmap_pose_map(images):
    pose_map = {}
    for image in images.values():
        rot_cw = qvec2rotmat(image.qvec)
        t_cw = image.tvec
        rot_wc = rot_cw.T
        trans_wc = -rot_cw.T @ t_cw
        quat_wxyz = rotmat_to_quat_wxyz(rot_wc)
        frame_id = extract_frame_id(image.name)
        pose_map[frame_id] = (trans_wc, quat_wxyz)
    return pose_map


def read_odometry_csv(path: Path):
    pose_map = {}
    with path.open("r") as fid:
        reader = csv.DictReader(fid, skipinitialspace=True)
        for row in reader:
            frame_id = normalize_frame_id(row["frame"].strip())
            position = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float)
            quat = np.array(
                [float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])],
                dtype=float,
            )
            quat = quat / np.linalg.norm(quat)
            pose_map[frame_id] = (position, quat)
    return pose_map


def sort_key(frame_id):
    if isinstance(frame_id, int):
        return (0, frame_id)
    return (1, str(frame_id))


def build_trajectory_from_map(pose_map, frames):
    positions = []
    orientations = []
    timestamps = []
    for idx, frame_id in enumerate(frames):
        position, quat = pose_map[frame_id]
        positions.append(position)
        orientations.append(quat)
        if isinstance(frame_id, int):
            timestamps.append(float(frame_id))
        else:
            timestamps.append(float(idx))
    return trajectory.PoseTrajectory3D(
        positions_xyz=np.array(positions),
        orientations_quat_wxyz=np.array(orientations),
        timestamps=np.array(timestamps),
    )


def compute_fov(width, height, fx, fy):
    fov_x = 2.0 * math.degrees(math.atan(width / (2.0 * fx)))
    fov_y = 2.0 * math.degrees(math.atan(height / (2.0 * fy)))
    return fov_x, fov_y


def set_axis_equal(ax, plot_mode, trajectories):
    coords = np.concatenate([traj.positions_xyz for traj in trajectories], axis=0)
    x_idx, y_idx, z_idx = plot.plot_mode_to_idx(plot_mode)
    xs = coords[:, x_idx]
    ys = coords[:, y_idx]
    if z_idx is not None:
        zs = coords[:, z_idx]
        max_range = max(xs.ptp(), ys.ptp(), zs.ptp())
        if max_range <= 0:
            max_range = 1.0
        mid_x, mid_y, mid_z = xs.mean(), ys.mean(), zs.mean()
        ax.set_xlim(mid_x - max_range / 2.0, mid_x + max_range / 2.0)
        ax.set_ylim(mid_y - max_range / 2.0, mid_y + max_range / 2.0)
        ax.set_zlim(mid_z - max_range / 2.0, mid_z + max_range / 2.0)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1, 1, 1))
    else:
        max_range = max(xs.ptp(), ys.ptp())
        if max_range <= 0:
            max_range = 1.0
        mid_x, mid_y = xs.mean(), ys.mean()
        ax.set_xlim(mid_x - max_range / 2.0, mid_x + max_range / 2.0)
        ax.set_ylim(mid_y - max_range / 2.0, mid_y + max_range / 2.0)
        ax.set_aspect("equal", adjustable="box")


def load_odometry_intrinsics(camera_matrix_csv, image_dir):
    if not camera_matrix_csv.exists():
        return None
    with camera_matrix_csv.open("r") as fid:
        rows = list(csv.reader(fid))
    fx = float(rows[0][0])
    fy = float(rows[1][1])
    width = height = None
    if image_dir.exists():
        first_image = next(image_dir.glob("*.png"), None)
        if first_image is None:
            first_image = next(image_dir.glob("*.jpg"), None)
        if first_image:
            with Image.open(first_image) as img:
                width, height = img.size
    if width is None or height is None:
        return None
    return width, height, fx, fy


def load_colmap_intrinsics(cameras):
    if not cameras:
        return None
    camera = next(iter(cameras.values()))
    params = camera.params
    if camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
        fx, fy = params[0], params[1]
    else:
        fx = fy = params[0]
    return camera.width, camera.height, float(fx), float(fy), camera.model


def main():
    parser = argparse.ArgumentParser(description="evo demo: COLMAP vs odometry")
    parser.add_argument(
        "--colmap-rec",
        default="/root/autodl-tmp/mpsfm/local/example/sp-lg-99-fod/rec",
        help="COLMAP rec 目录（包含 cameras.bin/images.bin）",
    )
    parser.add_argument(
        "--odometry-csv",
        default="/root/autodl-tmp/mpsfm/local/example/fod/odometry.csv",
        help="odometry.csv 路径",
    )
    parser.add_argument(
        "--odometry-images",
        default="/root/autodl-tmp/mpsfm/local/example/fod/rgb",
        help="odometry 图像目录（用于读取图像尺寸）",
    )
    parser.add_argument(
        "--camera-matrix",
        default="/root/autodl-tmp/mpsfm/local/example/fod/camera_matrix.csv",
        help="odometry 相机内参 CSV",
    )
    parser.add_argument("--plot-mode", default="xyz", choices=["xyz", "xz"], help="轨迹可视化模式")
    parser.add_argument("--no-plot", action="store_true", help="仅输出数值结果，不展示图像")
    parser.add_argument(
        "--save-dir",
        default="/root/autodl-tmp/mpsfm/local/example/evo_vis",
        help="可视化图片保存目录（默认保存）",
    )
    parser.add_argument("--save-prefix", default="refine_vs_origin", help="保存图片文件名前缀")
    parser.add_argument("--save-format", default="png", help="保存图片格式（png/jpg/pdf等）")
    parser.add_argument("--save-dpi", type=int, default=150, help="保存图片 DPI")
    parser.add_argument(
        "--no-show",
        dest="no_show",
        action="store_true",
        default=True,
        help="默认不弹窗显示（如需弹窗用 --show）",
    )
    parser.add_argument("--show", dest="no_show", action="store_false", help="弹窗显示")
    parser.add_argument(
        "--no-align",
        dest="align_umeyama",
        action="store_false",
        default=True,
        help="禁用Umeyama对齐（默认启用且包含尺度对齐）",
    )
    args = parser.parse_args()

    rec_dir = Path(args.colmap_rec)
    cameras = read_cameras_bin(rec_dir / "cameras.bin")
    images = read_images_bin(rec_dir / "images.bin")
    colmap_pose_map = build_colmap_pose_map(images)

    odom_pose_map = read_odometry_csv(Path(args.odometry_csv))

    common_frames = sorted(set(colmap_pose_map.keys()) & set(odom_pose_map.keys()), key=sort_key)
    if not common_frames:
        raise SystemExit("没有找到可对齐的 frame，请确认两套数据的帧号命名一致。")

    traj_colmap = build_trajectory_from_map(colmap_pose_map, common_frames)
    traj_odom = build_trajectory_from_map(odom_pose_map, common_frames)
    traj_ref, traj_est = sync.associate_trajectories(traj_colmap, traj_odom)

    if args.align_umeyama:
        r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)
        print(f"Umeyama对齐: scale={s:.6f}")

    ape = metrics.APE(pose_relation=metrics.PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_est))
    stats = ape.get_all_statistics()
    print("APE (translation) 统计:")
    for key in sorted(stats.keys()):
        print(f"  {key}: {stats[key]}")
    # 输出最大误差对应帧
    ape_errors = np.asarray(ape.error).reshape(-1)
    if ape_errors.size > 0 and np.isfinite(ape_errors).any():
        max_idx = int(np.nanargmax(ape_errors))
        max_frame = common_frames[max_idx]
        max_err = ape_errors[max_idx]
        print(f"APE max 对应帧: {max_frame} (error={max_err})")

    colmap_intrinsics = load_colmap_intrinsics(cameras)
    if colmap_intrinsics:
        width, height, fx, fy, model = colmap_intrinsics
        fov_x, fov_y = compute_fov(width, height, fx, fy)
        print(f"Refine FOV ({model}): fx={fx:.3f}, fy={fy:.3f}, size=({width}x{height})")
        print(f"  fov_x={fov_x:.3f} deg, fov_y={fov_y:.3f} deg")

    odom_intrinsics = load_odometry_intrinsics(Path(args.camera_matrix), Path(args.odometry_images))
    if odom_intrinsics:
        width, height, fx, fy = odom_intrinsics
        fov_x, fov_y = compute_fov(width, height, fx, fy)
        print(f"Origin FOV: fx={fx:.3f}, fy={fy:.3f}, size=({width}x{height})")
        print(f"  fov_x={fov_x:.3f} deg, fov_y={fov_y:.3f} deg")

    if args.no_plot and not args.save_dir:
        return

    import matplotlib
    if args.save_dir and args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plot_mode = plot.PlotMode.xyz if args.plot_mode == "xyz" else plot.PlotMode.xz
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, label="refine", color="black")
    plot.traj(ax, plot_mode, traj_est, label="origin", color="tab:blue")
    set_axis_equal(ax, plot_mode, [traj_ref, traj_est])
    ax.legend()
    ax.set_title("Trajectory comparison (refine vs origin)")

    fig_err, ax_err = plt.subplots()
    plot.error_array(ax_err, ape.error, x_array=traj_ref.timestamps, xlabel="t", ylabel="APE (m)")
    ax_err.set_title("APE translation error")

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        traj_path = save_dir / f"{args.save_prefix}_traj.{args.save_format}"
        err_path = save_dir / f"{args.save_prefix}_ape.{args.save_format}"
        fig.savefig(traj_path, dpi=args.save_dpi, bbox_inches="tight")
        fig_err.savefig(err_path, dpi=args.save_dpi, bbox_inches="tight")
        print(f"已保存可视化图片: {traj_path}")
        print(f"已保存误差曲线: {err_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

