import logging
import subprocess
import sys

import torch
from PIL import Image

from mpsfm.extraction.base_model import BaseModel
from mpsfm.vars import gvars

from .utils.warp import (
    assign_keypoints,
    kpids_to_matches0,
    simple_nms,
)

roma_path = gvars.ROOT / "third_party/RoMa"
sys.path.append(str(roma_path))  # noqa: E402

from romatch.models.model_zoo.roma_models import roma_model  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class Roma(BaseModel):
    default_conf = {
        "model_name": "roma_outdoor.pth",
        "model_utils_name": "dinov2_vitl14_pretrain.pth",
        "sample_thresh": 0.1,
        "nms_radius": 8,
        "max_error": 2,
        "require_download": False,
    }
    required_inputs = [
        "image0",
        "image1",
    ]
    weight_urls = {
        "roma": {
            "roma_outdoor.pth": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
            "roma_indoor.pth": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
        },
        "dinov2_vitl14_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    }

    # Initialize the line matcher
    def _init(self, conf):
        model_path = roma_path / "pretrained" / conf["model_name"]
        dinov2_weights = roma_path / "pretrained" / conf["model_utils_name"]

        # Download the model as implemented by ROMA
        if not model_path.exists():
            model_path.parent.mkdir(exist_ok=True)
            link = self.weight_urls["roma"][conf["model_name"]]
            cmd = ["wget", link, "-O", str(model_path)]
            logger.info(f"Downloading the Roma model with `{cmd}`.")
            subprocess.run(cmd, check=True)

        if not dinov2_weights.exists():
            dinov2_weights.parent.mkdir(exist_ok=True)
            link = self.weight_urls[conf["model_utils_name"]]
            cmd = ["wget", link, "-O", str(dinov2_weights)]
            logger.info(f"Downloading the dinov2 model with `{cmd}`.")
            subprocess.run(cmd, check=True)
        logger.info("Loading Roma model...")
        # load the model

        weights = torch.load(model_path, map_location="cpu")
        dinov2_weights = torch.load(dinov2_weights, map_location="cpu")

        self.net = roma_model(
            resolution=(14 * 8 * 6, 14 * 8 * 6),
            upsample_preds=False,
            weights=weights,
            dinov2_weights=dinov2_weights,
            device=device,
        )
        self.conf = conf
        logger.info("Load Roma model done.")

    def _forward(self, data, mode="sparse"):
        img0 = data["image0"].cpu().numpy().squeeze() * 255
        img1 = data["image1"].cpu().numpy().squeeze() * 255
        img0 = img0.transpose(1, 2, 0)
        img1 = img1.transpose(1, 2, 0)
        img0 = Image.fromarray(img0.astype("uint8"))
        img1 = Image.fromarray(img1.astype("uint8"))
        W_A, H_A = img0.size
        W_B, H_B = img1.size

        warp, certainty = self.net.match(img0, img1, device=device)

        matches = warp.reshape(-1, 4)
        if "dense" in mode:
            certainty_nms = simple_nms(certainty, self.conf["nms_radius"])
            certainty_nms = certainty_nms.reshape(-1)
            mask = certainty_nms > self.conf["sample_thresh"]
            certainty_nms = certainty_nms[mask]
            matches_nms = matches[mask]
            dkps0, dkps1 = self.net.to_pixel_coordinates(matches_nms, H_A, W_A, H_B, W_B)
            dkps0 = dkps0.cpu().numpy()
            dkps1 = dkps1.cpu().numpy()
            scores0 = certainty_nms.cpu().numpy()
        if "sparse" in mode:
            certainty = certainty.reshape(-1).cpu().numpy()
            kpts0, kpts1 = self.net.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
            kpts0 = kpts0.cpu().numpy()
            kpts1 = kpts1.cpu().numpy()
            skpts0 = data["skpts0"]
            skpts1 = data["skpts1"]
            mkp_ids0 = assign_keypoints(kpts0 * data["scale0"], skpts0, self.conf.max_error)
            mkp_ids1 = assign_keypoints(kpts1 * data["scale1"], skpts1, self.conf.max_error)
            smatches, sscores = kpids_to_matches0(mkp_ids0, mkp_ids1, certainty)

        pred = {}
        if "dense" in mode:
            pred["dkeypoints0"], pred["dkeypoints1"] = dkps0, dkps1
            pred["dscores"] = scores0
        if "sparse" in mode:
            pred["smatches0"] = smatches
            pred["smatching_scores0"] = sscores

        return pred
