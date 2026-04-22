from __future__ import annotations

import os
from pathlib import Path

import pytest

from impression.cli import _scene_factory_from_module
from impression.preview import PyVistaPreviewer
from impression.mesh import mesh_to_pyvista

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_IMAGE_ROOT = PROJECT_ROOT / "project" / "reference-images"
REFERENCE_STL_ROOT = PROJECT_ROOT / "project" / "reference-stl"
_SURFACEBODY_FILES = {
    "test_surface.py",
    "test_surface_kernel.py",
    "test_surface_replacements.py",
    "test_surface_csg.py",
    "test_surface_csg_docs.py",
    "test_surface_threading.py",
    "test_surface_threading_docs.py",
    "test_surface_hinges.py",
    "test_surface_hinges_docs.py",
}
_LOFT_FILES = {"test_loft_kernel.py", "test_loft_showcase.py", "test_loft_suite.py", "test_loft_correspondence.py", "test_reference_images.py"}
_DEPRECATION_FILES = {
    "test_drafting.py",
    "test_heightmap.py",
    "test_hinges.py",
    "test_text.py",
    "test_threading.py",
    "test_mesh_deprecations.py",
}
_DEPRECATION_NODEID_SUBSTRINGS = (
    "tests/test_surface.py::test_surface_mesh_adapter_and_bridge_contract_are_explicit",
    "tests/test_loft.py::test_private_surface_loft_consumer_handoff_uses_standard_surface_collection_and_tessellation",
    "tests/test_loft.py::test_private_surface_loft_consumer_handoff_supports_staged_split_merge_output",
)


def pytest_configure():
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--update-dirty-reference-images",
        action="store_true",
        default=False,
        help="Rewrite dirty reference images under project/reference-images/dirty.",
    )


def pytest_collection_modifyitems(config, items) -> None:
    for item in items:
        filename = Path(str(item.fspath)).name
        if filename in _SURFACEBODY_FILES:
            item.add_marker(pytest.mark.surfacebody)
        if filename in _LOFT_FILES:
            item.add_marker(pytest.mark.loft)
        if filename == "test_reference_images.py":
            item.add_marker(pytest.mark.reference_image)
        if filename in _DEPRECATION_FILES or any(token in item.nodeid for token in _DEPRECATION_NODEID_SUBSTRINGS):
            item.add_marker(pytest.mark.deprecation)
            item.add_marker(pytest.mark.filterwarnings("ignore::DeprecationWarning"))


def load_scene_datasets(model_path: Path):
    """Load a model module and return its datasets."""
    scene_factory = _scene_factory_from_module(model_path)
    scene = scene_factory()
    previewer = PyVistaPreviewer(console=None)  # console unused for combine
    return previewer.collect_datasets(scene)


def load_scene_mesh(model_path: Path):
    """Load a model module and combine its datasets into a single PolyData."""
    datasets = load_scene_datasets(model_path)
    previewer = PyVistaPreviewer(console=None)  # console unused for combine
    combined = previewer.combine_to_mesh(datasets)
    return mesh_to_pyvista(combined)


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def data_dir(project_root: Path) -> Path:
    return project_root / "tests" / "data"


@pytest.fixture
def reference_image_root() -> Path:
    return REFERENCE_IMAGE_ROOT


@pytest.fixture
def reference_stl_root() -> Path:
    return REFERENCE_STL_ROOT


@pytest.fixture
def update_dirty_reference_images(pytestconfig) -> bool:
    return bool(pytestconfig.getoption("--update-dirty-reference-images"))
