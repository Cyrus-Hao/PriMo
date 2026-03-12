"""Dataset factory and base accessors for MP-SfM data processing."""

import importlib.util

from mpsfm.utils.tools import get_class

from .basedataset import BaseDataset, BaseDatasetParser
from .hloc.featurepairsdataset import FeaturePairsDataset
from .hloc.imagedataset import ImageDataset
from .hloc.imagepairdataset import ImagePairDataset
from .hloc.utils import WorkQueue, writer_fn


def get_dataset(name):
    """Return a dataset class by name."""
    import_paths = [name, f"{__name__}.{name}"]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                return get_class(path, BaseDataset)
            except AssertionError:
                mod = __import__(path, fromlist=[""])
                try:
                    return mod.__main_dataset__
                except AttributeError as exc:
                    print(exc)
                    continue

    raise RuntimeError(f'Dataset {name} not found in any of [{" ".join(import_paths)}]')


def get_dataset_parser(name):
    """Return a dataset parser class by name."""
    import_paths = [name, f"{__name__}.{name}"]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                return get_class(path, BaseDatasetParser)
            except AssertionError:
                mod = __import__(path, fromlist=[""])
                try:
                    return mod.__main_parser__
                except AttributeError as exc:
                    print(exc)
                    continue
    raise RuntimeError(f'Dataset parser {name} not found in any of [{" ".join(import_paths)}]')


__all__ = [
    "ImagePairDataset",
    "ImageDataset",
    "FeaturePairsDataset",
    "WorkQueue",
    "writer_fn",
    "get_dataset",
    "get_dataset_parser",
]
