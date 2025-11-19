"""Impression – parametric modeling playground."""

from __future__ import annotations

import os

from ._env import ensure_shell_env
from ._config import ensure_user_config

__all__ = ["__version__"]

__version__ = "0.1.0"

os.environ.setdefault("VTK_PYTHON_ALLOW_DUPLICATE_LIBS", "1")

ensure_shell_env()
ensure_user_config()
