from __future__ import annotations

import os
from pathlib import Path

import pytest

from impression.cli import _scene_factory_from_module
from impression.preview import PyVistaPreviewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure():
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def load_scene_mesh(model_path: Path):
    """Load a model module and combine its datasets into a single PolyData."""
    scene_factory = _scene_factory_from_module(model_path)
    scene = scene_factory()
    previewer = PyVistaPreviewer(console=None)  # console unused for combine
    datasets = previewer.collect_datasets(scene)
    merged = previewer.combine_to_polydata(datasets)
    return merged


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def data_dir(project_root: Path) -> Path:
    return project_root / "tests" / "data"
