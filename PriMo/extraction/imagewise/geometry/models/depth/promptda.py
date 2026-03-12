import os
from pathlib import Path

import cv2
import numpy as np

from mpsfm.extraction.base_model import BaseModel


class PromptDA(BaseModel):
    default_conf = {
        "return_types": [
            "depth",
            "valid",
        ],
        # 输入目录：深度 PNG
        # depth_png_dir 允许 uint16(mm) 或 float PNG/EXR
        "depth_png_dir": None,
        # 文件命名规则：将输入图像名的数字stem映射到6位零填充，如 12 -> 000012.png
        "pattern_depth": "{stem6}.png",
        # 尺度/单位：uint16 视为毫米->米，其余按米处理
        "scale": 1,
        # 不需要下载模型（直接从文件读取）
        "require_download": False,
    }
    name = "promptda"

    def _init(self, conf):
        self.depth_dir = Path(self.conf.depth_png_dir) if self.conf.depth_png_dir is not None else None
        if self.depth_dir is None:
            raise ValueError("PromptDA requires depth_png_dir to be set in config")

    def _forward(self, data):
        image = data["image"]
        name = data["meta"]["image_name"][0]
        stem = Path(name).stem
        # 将非零填充数字文件名映射为6位，如 9 -> 000009
        try:
            idx = int(stem)
            stem6 = f"{idx:06d}"
        except Exception:
            # 若非纯数字，则尝试直接用原始名（去扩展名）
            stem6 = stem

        depth_path = self.depth_dir / self.conf.pattern_depth.format(stem6=stem6, stem=stem)
        depth = self._read_depth(depth_path)
        
        # 应用 scale 参数
        if self.conf.scale != 1:
            depth = depth * self.conf.scale

        H, W = image.shape[:2]
        if depth.shape[:2] != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        valid = (depth > 0).astype(bool)

        out = {}
        if "depth" in self.conf.return_types:
            out["depth"] = depth.astype(np.float32)
        if "valid" in self.conf.return_types:
            out["valid"] = valid
        # 注意：不返回 depth_variance，由 depth.py 根据 depth_uncertainty 计算
        return out

    @staticmethod
    def _read_depth(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(path)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.dtype == np.uint16:
            depth = img.astype(np.float32) / 1000.0
        else:
            depth = img.astype(np.float32)
        return depth

    # 移除 _read_conf 方法，不再需要


