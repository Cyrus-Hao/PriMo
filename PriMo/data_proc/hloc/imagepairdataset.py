from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from mpsfm.data_proc.hloc.imagedataset import resize_image
from mpsfm.utils.io import read_image


class ImagePairDataset(torch.utils.data.Dataset):
    default_conf = {
        "grayscale": False,
        "resize_max": False,
        "dfactor": "???",
        "cache_images": False,
        "resize": False,
        "interpolation": "cv2_area",
    }

    def __init__(self, image_dir, conf, pairs):
        self.image_dir = image_dir
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.pairs = pairs
        if self.conf.cache_images:
            image_names = set(sum(pairs, ()))  # unique image names in pairs
            print(f"Loading and caching {len(image_names)} unique images.")
            self.images = {}
            self.scales = {}
            for name in tqdm(image_names):
                image = read_image(self.image_dir / name, self.conf.grayscale)
                self.images[name], self.scales[name] = self.preprocess(image)

    def preprocess(self, image: np.ndarray):
        image = image.astype(np.float32, copy=False)
        size = image.shape[:2][::-1]
        scale = np.array([1.0, 1.0])

        if self.conf.resize_max:
            scale = self.conf.resize_max / max(size)
            if scale < 1.0:
                size_new = tuple(int(round(x * scale)) for x in size)
                image = resize_image(image, size_new, self.conf.interpolation)
                scale = np.array(size) / np.array(size_new)

        if self.conf.grayscale:
            assert image.ndim == 2, image.shape
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = torch.from_numpy(image / 255.0).float()

        # assure that the size is divisible by dfactor
        if self.conf.resize:
            size_new = tuple(
                map(
                    lambda x: int(x // self.conf.dfactor * self.conf.dfactor),
                    image.shape[-2:],
                )
            )
            image = F.resize(image, size=size_new)
            scale = np.array(size) / np.array(size_new)[::-1]
        return image, scale

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        if self.conf.cache_images:
            image0, scale0 = self.images[name0], self.scales[name0]
            image1, scale1 = self.images[name1], self.scales[name1]
        else:
            image0 = read_image(self.image_dir / name0, self.conf.grayscale)
            image1 = read_image(self.image_dir / name1, self.conf.grayscale)
            image0, scale0 = self.preprocess(image0)
            image1, scale1 = self.preprocess(image1)
        return image0, image1, scale0, scale1, name0, name1
