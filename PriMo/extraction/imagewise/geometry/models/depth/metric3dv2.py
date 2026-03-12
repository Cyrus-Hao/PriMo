import sys
from pathlib import Path

import torch

from mpsfm.extraction.base_model import BaseModel
from mpsfm.vars import gvars

from ..normals.dsine import kappa_to_alpha

root_dir = str(gvars.ROOT / "third_party/Metric3D")
try:
    from mmcv.utils import Config
except ImportError:
    from mmengine import Config

sys.path.append(root_dir)  # noqa: E402

from mono.model.monodepth_model import get_configured_monodepth_model  # noqa: E402
from mono.utils.do_test import transform_test_data_scalecano  # noqa: E402
from mono.utils.running import load_ckpt  # noqa: E402


def slice_and_interpolate(tensor, pad_info, ori_shape, mode="bilinear"):
    sliced_tensor = tensor[:, pad_info[0] : tensor.shape[1] - pad_info[1], pad_info[2] : tensor.shape[2] - pad_info[3]]
    return torch.nn.functional.interpolate(sliced_tensor[None], ori_shape, mode=mode).squeeze(0)


class Metric3Dv2(BaseModel):
    default_conf = {
        "return_types": ["depth", "depth_variance", "normals", "normals_variance", "valid"],
        "model_name": "metric_depth_vit_giant2_800k.pth",
        "download_url": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth",
        "config_name": "vit.raft5.giant2.py",
        "output_coords": "bni",
        "require_download": True,
        "download_method": "wget",
    }
    name = "metric3dv2"

    def _init(self, conf):
        self.metric3d_cfg = Config.fromfile(root_dir + "/mono/configs/HourglassDecoder/" + self.conf.config_name)

        self.model = get_configured_monodepth_model(
            self.metric3d_cfg,
        )
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model, _, _, _ = load_ckpt(
            Path(self.conf.models_dir, self.conf.model_name), self.model, strict_match=False
        )
        self.model.eval()

        if self.conf.output_coords == "bni":
            self.output_coords = self.omni_to_bni

    def _forward(self, data):
        image = data["image"]
        intrinsics = data["intrinsics"]
        intrinsics = intrinsics

        ori_shape = [image.shape[0], image.shape[1]]

        rgb_input, cam_models_stacks, pad_info, label_scale_factor = transform_test_data_scalecano(
            image, intrinsics, self.metric3d_cfg.data_basic
        )
        rgb_input = rgb_input[None]
        normalize_scale = self.metric3d_cfg.data_basic.depth_range[1]

        data = dict(
            input=rgb_input,
            cam_model=None,  # default method inputs cam model but doesn not use it
        )
        pred_depth, normals, error, normal_confidence, valid = self.step(
            data, pad_info, ori_shape, normalize_scale, label_scale_factor
        )
        normals = self.output_coords(normals.permute(1, 2, 0))
        depth_variance = error**2
        outdict = dict(
            depth=pred_depth,
            depth_variance=depth_variance,
            normals=normals,
            normals_confidence=normal_confidence,
            valid=valid,
        )
        if any(s.endswith("2") for s in self.conf.return_types):
            flipped_data = dict(
                input=torch.flip(rgb_input, dims=[3]),
                cam_model=None,  # default method inputs cam model but doesn not use it
            )
            pred_depth_flipped, normals2, error_flipped, normal_confidence_flipped, valid_flipped = self.step(
                flipped_data, pad_info, ori_shape, normalize_scale, label_scale_factor
            )
            pred_depth_flipped, normals2, error_flipped, normal_confidence_flipped, valid_flipped = [
                torch.flip(tensor, dims=[2])
                for tensor in [pred_depth_flipped, normals2, error_flipped, normal_confidence_flipped, valid_flipped]
            ]

            normals2 = self.output_coords(normals2.permute(1, 2, 0))
            normals2[..., 0] *= -1
            depth_variance_flipped = error_flipped**2
            outdict.update(
                dict(
                    depth2=pred_depth_flipped,
                    depth_variance2=depth_variance_flipped,
                    normals2=normals2,
                    normals2_confidence=normal_confidence_flipped,
                    valid2=valid_flipped,
                )
            )

        out_kwargs = {key: val.cpu().numpy() for key, val in outdict.items()}

        out_kwargs["normals_variance"] = kappa_to_alpha(out_kwargs["normals_confidence"]) ** 2
        if any(s.endswith("2") for s in self.conf.return_types):
            out_kwargs["normals2_variance"] = kappa_to_alpha(out_kwargs["normals2_confidence"]) ** 2
        out_kwargs = {k: v.squeeze() for k, v in out_kwargs.items() if k in self.conf.return_types}
        return out_kwargs

    def step(self, data, pad_info, ori_shape, normalize_scale, label_scale_factor):
        _, _, output = self.model.module.inference(data)
        pred_depth_canon, pred_normal_canon_, confidence_canon = [
            output[key].squeeze(0) for key in ["prediction", "prediction_normal", "confidence"]
        ]

        pred_normal_canon = pred_normal_canon_[:-1]
        normal_confidence_canon = pred_normal_canon_[-1, None]
        valid_canon = (pred_depth_canon < 200).float()

        pred_depth, valid, pred_normal, confidence, normal_confidence = [
            slice_and_interpolate(tensor, pad_info, ori_shape)
            for tensor in [pred_depth_canon, valid_canon, pred_normal_canon, confidence_canon, normal_confidence_canon]
        ]

        pred_depth = pred_depth * normalize_scale / label_scale_factor
        confidence = torch.clamp(confidence, 0, 1)
        error = pred_depth * (1 - confidence)
        return pred_depth, pred_normal, error, normal_confidence, valid == 1

    @staticmethod
    def omni_to_bni(normals):
        normals[..., 1:] = -normals[..., 1:]
        return normals
