from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".impression"
CONFIG_FILE = CONFIG_DIR / "impression.cfg"
DEFAULT_CONFIG = {
    "_comment": "Valid units: millimeters (default), meters, inches.",
    "units": "millimeters",
}


def ensure_user_config() -> None:
    """Ensure ~/.impression/impression.cfg exists with sane defaults."""

    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    if CONFIG_FILE.exists():
        return

    try:
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n")
    except OSError:
        return
