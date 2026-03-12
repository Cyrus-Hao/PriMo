"""Extraction module for MP-SfM pipeline."""

import importlib.util
import os
import sys

import torch

from mpsfm.extraction.base_model import BaseModel
from mpsfm.utils.tools import get_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_python_modules(root_path, base_module, name=None):
    """Recursively find all Python modules in a directory."""
    modules = []
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if fname.endswith(".py") and fname != "__init__.py":
                mod_name = os.path.splitext(fname)[0]
                if name and mod_name != name:
                    continue
                rel_path = os.path.relpath(os.path.join(dirpath, fname), root_path)
                mod_path = os.path.splitext(rel_path)[0].replace(os.sep, ".")
                modules.append(f"{base_module}.{mod_path}")
    return modules


def get_model(name):
    """Return a model class by name."""
    base_module = sys.modules[__name__]
    import_paths = find_python_modules(base_module.__path__[0], base_module.__name__, name)
    for path in import_paths:
        spec = importlib.util.find_spec(path)

        if spec is not None:
            obj = get_class(path, BaseModel)
            if obj is not None:
                return obj
    raise RuntimeError(f'Model {name} not found in any of [{" ".join(import_paths)}]')


def load_model(conf):
    """Load a model from the configuration."""
    model = get_model(conf["model"]["name"])(conf["model"]).eval().to(device)
    print(f"Loaded {conf['model']['name']} model.")
    return model


from .base import Extraction  # noqa: E402

__all__ = ["Extraction", "load_model", "get_model", "device"]
