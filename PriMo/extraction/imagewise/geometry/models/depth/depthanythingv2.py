import sys
from pathlib import Path

import numpy as np
import torch

from mpsfm.extraction.base_model import BaseModel
from mpsfm.vars import gvars

root_dir = str(gvars.ROOT / "third_party/Depth-Anything-V2/metric_depth")
sys.path.append(root_dir)  # noqa: E402

from depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2_  # noqa: E402

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


class DepthAnythingV2(BaseModel):
    default_conf = {
        "return_types": ["depth"],
        "scale": 1,
        "encoder": "vitl",
        "model_name": "depth_anything_v2_metric_vkitti_vitl.pth",
        "model_type": "metric",
        "max_depth": 80,  # 20 for indoor
        "datset_name": "vkitti",
        "require_download": True,
        "download_method": "wget",
        "download_url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth",
    }
    name = "depthanythingv2"

    def _init(self, conf):
        self.model = DepthAnythingV2_(
            **{
                **model_configs[self.conf.encoder],
                "max_depth": self.conf.max_depth,
            }
        )
        self.model.load_state_dict(
            torch.load(
                Path(self.conf.models_dir, self.conf.model_name),
                map_location="cpu",
            )
        )
        self.model = self.model.to("cuda").eval()

    def _forward(self, data):
        image = data["image"]
        flipped_image = np.flip(image, axis=1)
        pred_depth = self.model.infer_image(image)
        pred_depth_flipped = self.model.infer_image(flipped_image)
        pred_depth_flipped = np.flip(pred_depth_flipped, axis=1)
        out_kwargs = {
            key: val
            for key, val in dict(
                depth=pred_depth,
                depth2=pred_depth_flipped,
            ).items()
        }

        return out_kwargs
