import inspect
import re
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_class(mod_path, BaseClass):
    """
    Get the class object which inherits from BaseClass and is defined in the
    module or package specified by mod_path.
    """
    mod = __import__(mod_path, fromlist=[""])
    classes = inspect.getmembers(mod, inspect.isclass)
    classes = [c for c in classes if c[1].__module__.startswith(mod_path)]
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    classes = [c for c in classes if issubclass(c[1], BaseClass) and c[1] is not BaseClass]
    if len(classes) != 1:
        return None
    return classes[0][1]


def load_cfg(path, return_name=True):
    """Load the config file at path, handling nested defaults recursively."""

    def recursive_load(conf_path, base_conf=None):
        """Helper function to recursively load and merge configurations."""
        conf = OmegaConf.load(conf_path)
        parent_dir = conf_path.parent

        # Process defaults recursively
        if "defaults" in conf:
            defaults = conf.pop("defaults")
            for d in defaults:
                if isinstance(d, str):
                    default_path = parent_dir / f"{d}.yaml"
                    default_conf = recursive_load(default_path)
                    conf = OmegaConf.merge(default_conf, conf)
                elif isinstance(d, (dict, DictConfig)):
                    for key, value in d.items():
                        match = re.match(r"(.+?)@(.+)", key)
                        if match:
                            config_dir, target_path = match.groups()
                            default_path = parent_dir / config_dir / f"{value}.yaml"
                            default_conf = recursive_load(default_path)

                            sub_conf = OmegaConf.select(conf, target_path)
                            if target_path == ".":
                                # Overwrite the entire configuration
                                conf = OmegaConf.merge(default_conf, conf)
                            else:
                                if sub_conf:
                                    merged_sub_conf = OmegaConf.merge(default_conf, sub_conf)
                                    OmegaConf.update(conf, target_path, merged_sub_conf)
                                else:
                                    OmegaConf.update(conf, target_path, default_conf)
                        else:
                            raise ValueError(f"Unexpected format in defaults: {d}")
                else:
                    raise TypeError(f"Expected string or dict in defaults, got {type(d).__name__}")

        if base_conf:
            conf = OmegaConf.merge(base_conf, conf)

        return conf

    out_conf = recursive_load(Path(path))
    if return_name:
        out_conf.name = Path(path).stem
    return out_conf


def summarize_cfg(config):
    # Recursively summarize the configuration
    def summarize(node, depth=0):
        indent = "  " * depth
        if isinstance(node, (dict, DictConfig)):
            for key, value in node.items():
                if value == "<--->":
                    continue
                print(f"{indent}{key}:")
                summarize(value, depth + 1)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                print(f"{indent}- [{i}]")
                summarize(item, depth + 1)
        else:
            print(f"{indent}{node}")

    # Print the summary
    summarize(config)


def freeze_top_level_cfg(conf: DictConfig):
    OmegaConf.set_struct(conf, True)
    for k in conf:
        if isinstance(conf[k], DictConfig):
            OmegaConf.set_struct(conf[k], False)


def log_status(count, total, msg, width=80):
    status = f"({count}/{total}) {msg}"
    sys.stdout.write("\r" + status.ljust(width))
    sys.stdout.flush()
