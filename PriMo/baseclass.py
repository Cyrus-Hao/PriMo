from time import time

from omegaconf import OmegaConf

from mpsfm.utils.tools import freeze_top_level_cfg


class BaseClass:
    """Base class for mappers. This class is used to load and process the dataset."""

    freeze_conf = True
    default_conf = {
        "verbose": 0,
    }

    def __init__(self, conf=None, *args, **kwargs):
        """Perform some logic and call the _init method of the child model."""
        self.default_conf = OmegaConf.create(self.default_conf)
        if self.freeze_conf:
            freeze_top_level_cfg(self.default_conf)
        if conf is None:
            conf = {}
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self._assert_configs()
        self._propagate_conf()
        self._init(*args, **kwargs)
        self.tstart = None

    def _init(self, *args, **kwargs):
        """To be implemented by the child class."""

    def _assert_configs(self):
        """Assert that the configs are valid."""

    def _propagate_conf(self):
        """Propagate the configuration to the child class."""

    def log(self, *message, level=0, tstart=False, tend=False, **kwargs):
        """Log a message."""

        if self.conf.verbose >= level:
            if tstart:
                self.tstart = time()
            elif tend:
                assert len(message) == 0, "Message should be None when tend is True"
                assert self.tstart is not None, "tstart should be set before tend"
                assert not tstart, "tstart should not be set when tend is True"
                message = [f"{time() - self.tstart:.3f} s"]
            print(*message, end=" " if tstart else "\n", **kwargs)
