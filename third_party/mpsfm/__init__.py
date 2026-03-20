"""Compatibility shim for legacy ``mpsfm.*`` imports.

The codebase package was renamed to ``PriMo``. This shim keeps
`import mpsfm.xxx` working by redirecting module resolution to `PriMo/`.
"""

from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
_primo_pkg = _project_root / "PriMo"
__path__ = [str(_primo_pkg)]
