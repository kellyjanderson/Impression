from __future__ import annotations

import tomllib
from pathlib import Path

import impression


def test_runtime_version_matches_package_metadata() -> None:
    project_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text())

    assert impression.__version__ == metadata["project"]["version"]
