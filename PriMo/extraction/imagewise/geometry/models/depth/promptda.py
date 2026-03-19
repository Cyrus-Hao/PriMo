import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mpsfm.extraction.base_model import BaseModel
from mpsfm.vars import gvars

root_dir = str(gvars.ROOT / "third_party/PromptDA")
sys.path.append(root_dir)  # noqa: E402

try:
    from promptda.promptda import PromptDA
except ImportError as exc:  # pragma: no cover - import failure is environment-specific
    raise ImportError(
        "PromptDAEstimator could not import `promptda`. Expected PromptDA under "
        f"`{root_dir}` or an installed `promptda` package."
    ) from exc


class PromptDAEstimator(BaseModel):
    """Prompt Depth Anything estimator for metric monocular depth.

    The estimator follows PriMo's `BaseModel` interface and accepts an RGB image
    together with a rough metric depth prompt. The prompt can be supplied
    directly in `data["prompt_depth"]` or resolved from disk using either
    `prompt_depth_path` or `prompt_depth_dir` in the model configuration.

    Expected `data` keys:
        - `image`: RGB image as `H x W x 3` or `3 x H x W`.
        - `prompt_depth` (optional): rough metric depth as `H x W`,
          `1 x H x W`, or `1 x 1 x H x W`.
        - `prompt_depth_path` (optional): explicit per-sample prompt path.
        - `meta["image_name"]` (optional): used to resolve per-image prompt
          files when `prompt_depth_dir` is configured.

    Notes:
        - PNG prompt depths are interpreted as millimeters by default and are
          converted to meters via `png_depth_scale=1000.0`.
        - Outputs are resized back to the input RGB resolution used by PriMo's
          geometry extraction pipeline.
    """

    default_conf = {
        "return_types": ["depth", "valid"],
        "scale": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pretrained_model_name_or_path": "depth-anything/prompt-depth-anything-vitl",
        "pretrained_path": None,
        "encoder": "vitl",
        "max_image_size": 1008,
        "ensure_multiple_of": 14,
        "force_multiple_of": True,
        "prompt_depth_key": "prompt_depth",
        "prompt_depth_path": None,
        "prompt_depth_dir": None,
        "prompt_depth_suffix": "",
        "prompt_depth_ext": ".png",
        "prompt_scale_to_meter": 1.0,
        "png_depth_scale": 1000.0,
        "min_depth": 1e-6,
        "max_depth": None,
        "use_flip": False,
        "require_download": False,
    }
    name = "promptda"
    required_inputs = ["image"]

    def _init(self, conf):
        requested_device = conf.device
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        if requested_device == "mps" and not torch.backends.mps.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        model_kwargs = {}
        if getattr(self.conf, "encoder", None):
            model_kwargs["encoder"] = self.conf.encoder

        pretrained_source = self.conf.pretrained_path or self.conf.pretrained_model_name_or_path
        self.model = PromptDA.from_pretrained(pretrained_source, model_kwargs=model_kwargs).to(self.device).eval()

    def _forward(self, data):
        image = self._prepare_image_array(data["image"])
        target_shape = image.shape[:2]

        prompt_depth = self._resolve_prompt_depth(data)
        image_tensor = self._preprocess_image(image)
        prompt_depth_tensor = self._preprocess_prompt_depth(prompt_depth)

        pred_depth = self._predict(image_tensor, prompt_depth_tensor, target_shape)
        out_kwargs = {
            "depth": pred_depth,
            "valid": self._depth_is_valid(pred_depth),
        }

        if self.conf.use_flip or any(name.endswith("2") for name in self.conf.return_types):
            pred_depth_flipped = self._predict(
                torch.flip(image_tensor, dims=[3]),
                torch.flip(prompt_depth_tensor, dims=[3]),
                target_shape,
            )
            pred_depth_flipped = np.flip(pred_depth_flipped, axis=1).copy()
            out_kwargs["depth2"] = pred_depth_flipped
            out_kwargs["valid2"] = self._depth_is_valid(pred_depth_flipped)

        return {key: value for key, value in out_kwargs.items() if key in self.conf.return_types}

    def _predict(self, image_tensor, prompt_depth_tensor, target_shape):
        depth = self.model.predict(image_tensor, prompt_depth_tensor)
        depth = self._squeeze_depth(depth)
        if tuple(depth.shape) != tuple(target_shape):
            depth = F.interpolate(
                depth[None, None],
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        return depth.detach().cpu().numpy().astype(np.float32)

    def _preprocess_image(self, image):
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        height, width = image.shape[:2]
        target_height, target_width = height, width
        max_size = self._nearest_multiple(self.conf.max_image_size)

        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            target_height = self._nearest_multiple(height * scale)
            target_width = self._nearest_multiple(width * scale)

        if self.conf.force_multiple_of:
            target_height = self._nearest_multiple(target_height)
            target_width = self._nearest_multiple(target_width)

        if (target_height, target_width) != (height, width):
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

        return (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._model_device(), dtype=torch.float32)
        )

    def _preprocess_prompt_depth(self, prompt_depth):
        prompt_depth = prompt_depth.astype(np.float32)
        prompt_depth = np.nan_to_num(prompt_depth, nan=0.0, posinf=0.0, neginf=0.0)
        prompt_depth = np.maximum(prompt_depth, 0.0)
        return (
            torch.from_numpy(prompt_depth)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self._model_device(), dtype=torch.float32)
        )

    def _resolve_prompt_depth(self, data):
        prompt_key = self.conf.prompt_depth_key
        if prompt_key in data:
            return self._prepare_depth_array(data[prompt_key]) * float(self.conf.prompt_scale_to_meter)

        if "prompt_depth_path" in data:
            return self._load_prompt_depth(data["prompt_depth_path"])

        if self.conf.prompt_depth_path:
            return self._load_prompt_depth(self.conf.prompt_depth_path)

        if self.conf.prompt_depth_dir:
            prompt_path = self._find_prompt_depth_path(data)
            return self._load_prompt_depth(prompt_path)

        raise ValueError(
            "PromptDAEstimator requires a depth prompt. Provide `data['prompt_depth']`, "
            "`data['prompt_depth_path']`, `model.prompt_depth_path`, or `model.prompt_depth_dir`."
        )

    def _find_prompt_depth_path(self, data):
        if "meta" not in data or "image_name" not in data["meta"]:
            raise ValueError("Prompt depth lookup via `prompt_depth_dir` requires `data['meta']['image_name']`.")

        prompt_dir = Path(self.conf.prompt_depth_dir)
        image_name = self._unwrap_meta_value(data["meta"]["image_name"], "image_name")
        image_stem = Path(image_name).stem
        suffix = self.conf.prompt_depth_suffix

        if self.conf.prompt_depth_ext:
            candidates = [prompt_dir / f"{image_stem}{suffix}{self.conf.prompt_depth_ext}"]
        else:
            candidates = [
                prompt_dir / f"{image_stem}{suffix}{ext}"
                for ext in [".png", ".npz", ".npy", ".exr"]
            ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find prompt depth for {image_stem}. Tried: {tried}")

    def _load_prompt_depth(self, path_like):
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"Prompt depth file does not exist: {path}")

        suffix = path.suffix.lower()
        if suffix == ".png":
            depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(f"Failed to read prompt depth PNG: {path}")
            depth = depth.astype(np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth = depth / float(self.conf.png_depth_scale)
        elif suffix == ".npz":
            loaded = np.load(path)
            if "depth" in loaded:
                depth = loaded["depth"]
            else:
                first_key = next(iter(loaded.files))
                depth = loaded[first_key]
            depth = np.asarray(depth, dtype=np.float32)
        elif suffix == ".npy":
            depth = np.asarray(np.load(path), dtype=np.float32)
        elif suffix == ".exr":
            depth = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise ValueError(f"Failed to read prompt depth EXR: {path}")
            depth = np.asarray(depth, dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
        else:
            raise ValueError(f"Unsupported prompt depth format: {path.suffix}")

        depth = self._prepare_depth_array(depth)
        return depth * float(self.conf.prompt_scale_to_meter)

    def _depth_is_valid(self, depth):
        valid = np.isfinite(depth) & (depth > float(self.conf.min_depth))
        if self.conf.max_depth is not None:
            valid &= depth < float(self.conf.max_depth)
        return valid

    def _model_device(self):
        return next(self.model.parameters()).device

    def _prepare_image_array(self, image):
        image = self._to_numpy(image)
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape HxWx3 or 3xHxW, got {image.shape}")
        return np.ascontiguousarray(image)

    def _prepare_depth_array(self, depth):
        depth = self._to_numpy(depth)
        if depth.ndim == 4:
            if depth.shape[0] != 1:
                raise ValueError(f"Expected a single prompt depth map, got {depth.shape}")
            depth = depth[0]
        if depth.ndim == 3:
            if depth.shape[0] == 1 and depth.shape[-1] != 1:
                depth = depth[0]
            elif depth.shape[-1] == 1:
                depth = depth[..., 0]
            else:
                depth = depth[0]
        if depth.ndim != 2:
            raise ValueError(f"Expected prompt depth with shape HxW, got {depth.shape}")
        return np.ascontiguousarray(depth.astype(np.float32))

    def _squeeze_depth(self, depth):
        if not torch.is_tensor(depth):
            depth = torch.as_tensor(depth)
        depth = depth.float()
        while depth.ndim > 2 and depth.shape[0] == 1:
            depth = depth.squeeze(0)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"Unexpected PromptDA output shape: {tuple(depth.shape)}")
        return depth

    @staticmethod
    def _to_numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _unwrap_meta_value(value, key):
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Expected a single `{key}` entry, got {len(value)}")
            return value[0]
        if isinstance(value, np.ndarray):
            flat = value.reshape(-1)
            if flat.size != 1:
                raise ValueError(f"Expected a single `{key}` entry, got array with shape {value.shape}")
            return flat[0].item() if hasattr(flat[0], "item") else flat[0]
        return value

    def _nearest_multiple(self, value):
        multiple = int(self.conf.ensure_multiple_of)
        if multiple <= 1:
            return int(round(value))
        return max(multiple, int(np.floor(float(value) / multiple)) * multiple)
