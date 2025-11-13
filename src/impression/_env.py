from __future__ import annotations

import os
import sys
from pathlib import Path


ENV_DIR = Path.home() / ".impression"
ENV_FILE = ENV_DIR / "env"
ENV_EXPORT_LINE = 'export IMPRESSION_PY="{interpreter}"\n'


def ensure_shell_env() -> None:
    """Write ~/.impression/env so shells can export IMPRESSION_PY automatically.

    The VS Code extension (and other tooling) can read either the environment
    variable or source the env file. We don't modify shell rc files directly,
    but we do provide a consistent file to source.
    """

    interpreter = str(Path(sys.executable).resolve())
    os.environ.setdefault("IMPRESSION_PY", interpreter)

    try:
        ENV_DIR.mkdir(parents=True, exist_ok=True)
        desired = ENV_EXPORT_LINE.format(interpreter=interpreter)
        current = ENV_FILE.read_text() if ENV_FILE.exists() else None
        if current != desired:
            ENV_FILE.write_text(desired)
    except OSError:
        return

