import sys

import numpy as np
import PIL
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from mpsfm.extraction.base_model import BaseModel
from mpsfm.vars import gvars

root_dir = str(gvars.ROOT / "third_party/DSINE")
sys.path.append(root_dir)  # noqa: E402

from pathlib import Path  # noqa: E402

from projects import get_default_parser  # noqa: E402
import sys
import os
# Add DSINE path to sys.path
dsine_path = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "third_party" / "DSINE"
if str(dsine_path) not in sys.path:
    sys.path.insert(0, str(dsine_path))
# Import DSINE utils specifically using importlib
import importlib.util
spec = importlib.util.spec_from_file_location("dsine_utils", dsine_path / "utils" / "utils.py")
dsine_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsine_utils)
utils = dsine_utils


def kappa_to_alpha(kappa):
    return (2 * kappa / (kappa**2 + 1)) + np.exp(-kappa * np.pi) * np.pi / (1 + np.exp(-kappa * np.pi))


class DSINE(BaseModel):
    default_conf = {
        "return_types": ["normals", "normals2", "normals_variance", "normals2_variance"],
        "scale": 0.5,
        "output_coords": "bni",
        "config_root": root_dir + "/projects/dsine/experiments",
        "config_dir_name": "exp002_kappa",
        "config_file_name": "dsine.txt",
        "model_name": "dsine.pth",
        "download_url": "1u8TdKXkR7-0zzRRcx-3x3rPN7gvAAM9N",
        "require_download": True,
        "download_method": "gdown",
    }
    device = "cuda:0"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _init(self, conf):
        args = dsine_get_args(Path(self.conf.config_root, self.conf.config_dir_name, self.conf.config_file_name))
        args.ckpt_path = Path(self.conf.models_dir, self.conf.model_name)

        if args.NNET_architecture == "v00":
            from models.dsine.v00 import DSINE_v00 as DSINE_model
        elif args.NNET_architecture == "v01":
            from models.dsine.v01 import DSINE_v01 as DSINE_model
        elif args.NNET_architecture == "v02":
            from models.dsine.v02 import DSINE_v02 as DSINE_model
        elif args.NNET_architecture == "v02_kappa":
            from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE_model
        else:
            raise Exception("invalid arch")

        self.model = DSINE_model(args).to(self.device)
        self.model = utils.load_checkpoint(args.ckpt_path, self.model)
        self.model.eval()

        if self.conf.output_coords == "bni":
            self.output_coords = self.omni_to_bni

    def post_step(self, pred, lrtb, orig_H, orig_W):
        pred = pred[:, :, lrtb[2] : lrtb[2] + orig_H, lrtb[0] : lrtb[0] + orig_W]

        pred = pred.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        pred_norm = self.output_coords(pred[..., :3])

        pred_uncert = kappa_to_alpha(pred[..., 3]) ** 2 if "kappa" in self.conf.config_dir_name else None

        return pred_norm, pred_uncert

    def _forward(self, data):
        with torch.no_grad():
            img = Image.fromarray(data["image"]).convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # pad input
            _, _, orig_H, orig_W = img.shape
            lrtb = utils.get_padding(orig_H, orig_W)
            img = F.pad(img, lrtb, mode="constant", value=0.0)
            img = self.normalize(img)

            intrins = data["intrinsics"]
            intrins = torch.tensor(
                [[intrins[0], 0, intrins[2]], [0, intrins[1], intrins[3]], [0, 0, 1]], device=self.device
            ).unsqueeze(0)

            pred = self.model(img, intrins=intrins)[-1]

            pred_norm, pred_uncert = self.post_step(pred, lrtb, orig_H, orig_W)

            flipped_pred = torch.flip(self.model(torch.flip(img, dims=[3]), intrins=intrins)[-1], dims=[3])
            flipped_pred_norm, flipped_pred_uncert = self.post_step(flipped_pred, lrtb, orig_H, orig_W)
            flipped_pred_norm[..., 0] *= -1

        out = {}
        if "normals" in self.conf.return_types:
            out["normals"] = pred_norm
        if "normals2" in self.conf.return_types:
            out["normals2"] = flipped_pred_norm
        if "normals_variance" in self.conf.return_types:
            out["normals_variance"] = pred_uncert
        if "normals2_variance" in self.conf.return_types:
            out["normals2_variance"] = flipped_pred_uncert
        return out

    def pad_to_square(self, img, multiple):
        w, h = img.size
        max_side = max(w, h)
        rounded_side = max_side if max_side % multiple == 0 else (max_side // multiple + 1) * multiple
        pad_w = (rounded_side - w) // 2
        extra_pad_w = (rounded_side - w) % 2
        pad_h = (rounded_side - h) // 2
        extra_pad_h = (rounded_side - h) % 2
        self.padding_values = [pad_w, pad_h, pad_w + extra_pad_w, pad_h + extra_pad_h]
        out = transforms.functional.pad(img, self.padding_values, fill=0, padding_mode="constant")

        return out

    def unpad(self, tensor):
        pad_wa, pad_ha, pad_wb, pad_hb = self.padding_values
        original_w = tensor.size(3) - (pad_wa + pad_wb)  # width
        original_h = tensor.size(2) - (pad_ha + pad_hb)  # height
        return tensor[:, :, pad_ha : pad_ha + original_h, pad_wa : pad_wa + original_w]

    def resize(self, img, scale):

        W, H = self.original_size = img.size
        return transforms.functional.resize(
            img, size=(int(H / scale), int(W / scale)), interpolation=PIL.Image.BILINEAR
        )

    @staticmethod
    def omni_to_bni(normals):
        normals[..., 0] = -normals[..., 0]
        return normals


def dsine_get_args(config_pth):
    """Adapted from DSINE repo to avoid making dirs"""
    parser = get_default_parser()

    # ↓↓↓↓
    # NOTE: project-specific args
    parser.add_argument("--NNET_architecture", type=str, default="v02")
    parser.add_argument("--NNET_output_dim", type=int, default=3, help="{3, 4}")
    parser.add_argument("--NNET_output_type", type=str, default="R", help="{R, G}")
    parser.add_argument("--NNET_feature_dim", type=int, default=64)
    parser.add_argument("--NNET_hidden_dim", type=int, default=64)

    parser.add_argument("--NNET_encoder_B", type=int, default=5)

    parser.add_argument("--NNET_decoder_NF", type=int, default=2048)
    parser.add_argument("--NNET_decoder_BN", default=False, action="store_true")
    parser.add_argument("--NNET_decoder_down", type=int, default=8)
    parser.add_argument("--NNET_learned_upsampling", default=False, action="store_true")

    parser.add_argument("--NRN_prop_ps", type=int, default=5)
    parser.add_argument("--NRN_num_iter_train", type=int, default=5)
    parser.add_argument("--NRN_num_iter_test", type=int, default=5)
    parser.add_argument("--NRN_ray_relu", default=False, action="store_true")

    parser.add_argument("--loss_fn", type=str, default="AL")
    parser.add_argument("--loss_gamma", type=float, default=0.8)
    # ↑↑↑↑

    # read arguments from txt file
    args = parser.parse_args([f"@{config_pth}"])

    # ↓↓↓↓
    # NOTE: update args
    # args.exp_root = os.path.join(EXPERIMENT_DIR, 'dsine')
    args.load_normal = True
    args.load_intrins = True
    # ↑↑↑↑

    return args
