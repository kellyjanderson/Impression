from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path.home() / ".impression"
CONFIG_FILE = CONFIG_DIR / "impression.cfg"
DEFAULT_CONFIG = {
    "_comment": "Valid units: millimeters (default), meters, inches. Value is case-insensitive.",
    "units": "millimeters",
}
_UNIT_INFO: Dict[str, Dict[str, Any]] = {
    "millimeters": {"label": "mm", "scale_to_mm": 1.0},
    "meters": {"label": "m", "scale_to_mm": 1000.0},
    "inches": {"label": "in", "scale_to_mm": 25.4},
}
_UNIT_ALIASES = {
    "millimeter": "millimeters",
    "millimeters": "millimeters",
    "mm": "millimeters",
    "meter": "meters",
    "meters": "meters",
    "m": "meters",
    "inch": "inches",
    "inches": "inches",
    "in": "inches",
}


@dataclass(frozen=True)
class UnitSettings:
    """Resolved units from impression.cfg."""

    name: str
    label: str
    scale_to_mm: float


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


def _load_user_config() -> Dict[str, Any]:
    ensure_user_config()
    try:
        return json.loads(CONFIG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return DEFAULT_CONFIG.copy()


def _normalize_units(value: str) -> str | None:
    key = value.strip().lower()
    if key in _UNIT_INFO:
        return key
    return _UNIT_ALIASES.get(key)


def get_unit_settings() -> UnitSettings:
    """Return the configured units and the conversion to millimeters."""

    raw_config = _load_user_config()
    raw_units = str(raw_config.get("units", DEFAULT_CONFIG["units"]))
    normalized = _normalize_units(raw_units)
    if normalized is None:
        normalized = DEFAULT_CONFIG["units"]

    info = _UNIT_INFO[normalized]
    return UnitSettings(name=normalized, label=info["label"], scale_to_mm=info["scale_to_mm"])
