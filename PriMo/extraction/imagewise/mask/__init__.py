from pathlib import Path

from .base import main

__all__ = [name for name in globals() if not name.startswith("_")]

CONFIG_DIR = Path(__file__).parent / "models/configs"

__all__ += ["CONFIG_DIR", "main"]
