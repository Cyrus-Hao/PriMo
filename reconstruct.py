import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
# 兼容旧导入：mpsfm.* -> PriMo/*
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "mpsfm_compat"))
sys.path.append(str(PROJECT_ROOT / "colmap" / "python" / "pycolmap"))

from mpsfm.test.simple import SimpleTest
from mpsfm.utils.tools import load_cfg
from mpsfm.vars import gvars

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, default="local/example", help="Main data dir storing inputs and later the outputs"
)
parser.add_argument("--images_dir", type=str, help="Images directory")
parser.add_argument(
    "--imnames", type=str, nargs="*", help="List of image names to process. Leave empty to process all images"
)
parser.add_argument("--intrinsics_pth", type=str, default=None, help="Path to intrinsics .yaml file")
parser.add_argument("--refrec_dir", type=str, default=None, help="Path to reference reconstruction")
parser.add_argument("--cache_dir", type=str, default=None, help="Path to cache directory")
parser.add_argument("--pose_config_path", type=str, default=None, help="Path to camera_poses.yaml")
parser.add_argument("-e", "--extract", nargs="*", type=str, default=[], help="List of priors to force reextract")
parser.add_argument("-c", "--conf", type=str, help="Name of the sfm config file", default="sp-lg_m3dv2")
parser.add_argument("-v", "--verbose", type=int, default=0)

args, _ = parser.parse_known_args()
conf = load_cfg(gvars.SFM_CONFIG_DIR / f"{args.conf}.yaml", return_name=False)
conf.extract = args.extract
conf.verbose = args.verbose

# Resolve prior-pose path robustly after project rename.
if getattr(conf.registration, "use_prior_poses", False):
    requested_pose_path = args.pose_config_path or getattr(conf.registration, "pose_config_path", None)
    candidate_paths = []
    if requested_pose_path:
        candidate_paths.append(Path(requested_pose_path))
    if args.data_dir:
        candidate_paths.append(Path(args.data_dir) / "camera_poses.yaml")

    resolved_pose_path = None
    for pth in candidate_paths:
        if pth is not None and pth.exists():
            resolved_pose_path = pth
            break

    if resolved_pose_path is not None:
        conf.registration.pose_config_path = str(resolved_pose_path)
    else:
        print(
            "[WARN] prior pose enabled but camera_poses.yaml not found. "
            "Falling back to non-prior retrieval/registration for this run."
        )
        conf.registration.pose_config_path = None

experiment = SimpleTest(conf)
mpsfm_rec = experiment(
    imnames=args.imnames,
    intrinsics_pth=args.intrinsics_pth,
    refrec_dir=args.refrec_dir,
    cache_dir=args.cache_dir,
    data_dir=args.data_dir,
    images_dir=args.images_dir,
)
sfm_outputs_dir = Path(args.data_dir) / "sfm_outputs"
sfm_outputs_dir.mkdir(exist_ok=True)
mpsfm_rec.write(sfm_outputs_dir)
