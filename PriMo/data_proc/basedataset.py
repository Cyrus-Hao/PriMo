"""Base dataset class for loading and processing datasets for MP-SfM pipeline."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from mpsfm.data_proc.hloc.imagedataset import resize_image


class BaseDatasetParser(ABC):
    """Base class for dataset parsers. This class is used to parse the dataset and create a reconstruction object."""

    base_default_conf = {}
    default_conf = {}
    reconstruction_dir = None
    rec = None
    rgb_dir = None

    def __init__(self, *args, **kwargs):
        default_conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf),
            OmegaConf.create(self.default_conf),
        )
        if "conf" not in kwargs:
            kwargs["conf"] = {}
        conf = OmegaConf.create(kwargs["conf"]) if isinstance(kwargs["conf"], dict) else kwargs["conf"]
        self.conf = OmegaConf.merge(default_conf, conf)
        self._init(*args, **kwargs)

    def camera(self, imid):
        """Get camera object for the image id."""
        return self.rec.cameras[self.rec.images[imid].camera_id]

    def pose(self, imid):
        """Get camera pose for the image id."""
        return self.rec.images[imid].cam_from_world

    def image_name(self, imid):
        """Get image name for the image id."""
        return self.rec.images[imid].name

    def rgb(self, imid):
        """Get RGB image for the image id."""
        imname = self.image_name(imid)
        im = Image.open(self.rgb_dir / imname)
        im = np.array(im)
        return im

    @abstractmethod
    def _init(self, *args, **kwargs):
        """To be implemented by the child class."""


class BaseDataset:
    """Base class for datasets. This class is used to load and process the dataset."""

    base_default_conf = {
        "return_types": "???",
        "batch_size": 1,
        "num_workers": 0,
        "depth_details": {"error_to_conf_lambda": None},
        "resize_max": 1200,
        "interpolation": "pil_lanczos",
    }
    default_conf = {}
    scenes = None

    def __init__(self, conf=None, image_list=None, **kwargs):
        if conf is None:
            conf = {}
        merged_conf_dict = {**self.base_default_conf, **self.default_conf, **conf}
        self.conf = OmegaConf.create(merged_conf_dict)
        self._init(image_list=image_list, **kwargs)

    def _init(self, image_list=None, scene_parser=None, **kwargs):
        assert self.conf.batch_size == 1, "Batch size must be 1 for this dataset"
        self.imids = []
        self.scene_parser = scene_parser
        if image_list is None:
            image_ids = list(self.scene_parser.rec.images.keys())
        else:
            image_ids = [k for k, v in self.scene_parser.rec.images.items() if Path(v.name).name in image_list]

        image_ids.sort()
        self.imids = image_ids

    def collect_meta(self, out):
        """Collect meta information for the item."""
        meta = out["item"]["meta"] = {}
        meta["scene"] = self.scene_parser.scene
        meta["image_name"] = self.scene_parser.rec.images[out["imid"]].name
        meta["image_id"] = out["imid"]
        return out

    def load_source(self, types, imid):
        """Load source data for the given image id."""
        out = {}
        if "image" in types:
            im = self.scene_parser.rgb(imid)
            im = im.astype(np.float32)
            size = np.array(im.shape[:2][::-1])
            if self.conf.resize_max:
                scale = self.conf.resize_max / max(size)
                size_new = tuple(int(round(x * scale)) for x in size)
                im = resize_image(im, size_new, self.conf.interpolation)
                scales = (size / size_new).astype(np.float32)
            im = im.transpose((2, 0, 1))
            scales = scales if self.conf.resize_max else np.array([1.0, 1.0])
            out["image"] = im / 255
            out["original_size"] = size
            out["scales"] = scales
        if "intrinsics" in types:
            out["intrinsics"] = np.copy(self.scene_parser.camera(imid).params)
            out["intrinsics"][:2] /= scales
            out["intrinsics"][2:] /= scales
        return out

    def get_item(self, idx):
        """Get item from the dataset."""
        item = {k: torch.tensor(v) for k, v in self.load_source(self.conf.return_types, self.imids[idx]).items()}
        item = {"item": item, "imid": self.imids[idx]}
        return item

    def __len__(self):
        return len(self.imids)

    def __getitem__(self, idx):
        out = self.get_item(idx)
        out = self.collect_meta(out)
        item = self.end_of_getitem(**out)
        return item

    def end_of_getitem(self, item, **kwargs):
        """Optional function to be implemented by the child class."""
        return item

    def get_dataloader(self):
        """Get dataloader for the dataset."""
        return DataLoader(self, batch_size=self.conf.batch_size, shuffle=False, num_workers=self.conf.num_workers)
