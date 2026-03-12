import os
import subprocess
import sys
from abc import ABCMeta, abstractmethod

from omegaconf import OmegaConf
from torch import nn

from mpsfm.vars import gvars


class BaseModel(nn.Module, metaclass=ABCMeta):
    """Base class for extraction models."""

    base_default_conf = {
        "models_dir": str(gvars.ROOT / "local/weights"),
    }

    default_conf = {}

    required_inputs = []

    def __init__(self, conf):
        super().__init__()
        merged_conf_dict = {**self.base_default_conf, **self.default_conf, **conf}
        self.conf = OmegaConf.create(merged_conf_dict)

        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)

        self.conf = conf = OmegaConf.merge(merged_conf_dict, conf)
        if self.conf.require_download:
            self._download_model()
        self._init(conf)

        for p in self.parameters():
            p.requires_grad = False

    def _download_model(self):
        model_path = gvars.ROOT / "local/weights" / self.conf.model_name
        if self.conf.require_download and model_path.exists():
            return

        os.makedirs(model_path.parent, exist_ok=True)
            
        if self.conf.download_method == "wget":
            print(f"📥 Downloading via wget: {self.conf.download_url}")
            subprocess.run(
                ["wget", self.conf.download_url, "-O", model_path], stdout=sys.stdout, stderr=sys.stderr, check=True
            )

        elif self.conf.download_method == "gdown":
            print(f"📥 Downloading via gdown: {self.conf.download_url}")
            subprocess.run(["gdown", self.conf.download_url, "-O", str(model_path)], check=True)

        else:
            raise ValueError(f"Unknown download method: {self.conf.download_method}")

    def forward(self, data, **kwargs):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_inputs:
            assert key in data, f"Missing key {key} in data"
        return self._forward(data, **kwargs)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError
