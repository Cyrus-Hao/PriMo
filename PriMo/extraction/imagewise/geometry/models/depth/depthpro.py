import sys
from pathlib import Path

import numpy as np
import torch

from mpsfm.extraction.base_model import BaseModel
from mpsfm.vars import gvars

root_dir = str(gvars.ROOT / "third_party/ml-depth-pro")
sys.path.append(root_dir)  # noqa: E402

import depth_pro  # noqa: E402
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT  # noqa: E402


class DepthPro(BaseModel):
    default_conf = {
        "return_types": ["depth"],
        "scale": 1,
        "model_name": "depth_pro.pt",
        "download_url": "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt",
        "download_method": "wget",
        "require_download": True,
    }
    name = "depthpro"

    def _init(self, conf):
        DEFAULT_MONODEPTH_CONFIG_DICT.checkpoint_uri = Path(self.conf.models_dir, self.conf.model_name)
        self.model, self.transform = depth_pro.create_model_and_transforms(DEFAULT_MONODEPTH_CONFIG_DICT)
        self.model.eval()
        self.model = self.model.to("cuda")

    def _forward(self, data):
        image = self.transform(data["image"].copy()).cuda()

        flipped_image = torch.flip(image, dims=[2])
        intrinsics = data["intrinsics"]
        f_px = np.mean(intrinsics[:2])
        pred_depth, pred_depth_flipped = self.model.infer(torch.cat([image[None], flipped_image[None]]), f_px=f_px)[
            "depth"
        ]

        pred_depth_flipped = torch.flip(pred_depth_flipped, dims=[1])

        valid = (pred_depth < 250) * (pred_depth_flipped < 250)

        out_kwargs = {
            key: val.cpu().numpy()
            for key, val in dict(
                depth=pred_depth,
                depth2=pred_depth_flipped,
                valid=valid,
            ).items()
        }
        return out_kwargs
