from __future__ import annotations

import tomllib
from pathlib import Path

import impression


def test_runtime_version_matches_package_metadata() -> None:
    project_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text())

    assert impression.__version__ == metadata["project"]["version"]


def test_shelved_sdf_endcap_experiment_is_not_packaged() -> None:
    project_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text())
    modeling_root = project_root / "src" / "impression" / "modeling"

    assert not (modeling_root / "sdf.py").exists()
    assert not (modeling_root / "extrude.py").exists()
    assert not (modeling_root / "_profile2d.py").exists()

    modeling_exports = (modeling_root / "__init__.py").read_text()
    assert "extrude_sdf" not in modeling_exports
    assert "loft_sdf" not in modeling_exports
    assert not any(
        dependency.lower().startswith("scikit-image")
        for dependency in metadata["project"]["dependencies"]
    )


def test_experimental_half_pipe_payload_is_not_packaged() -> None:
    project_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text())

    assert not (project_root / "examples" / "half_pipe.py").exists()
    assert not (project_root / "src" / "impression" / "cad.py").exists()
    assert all(
        "build123d" not in dependency.lower()
        for dependency in metadata["project"]["dependencies"]
    )
