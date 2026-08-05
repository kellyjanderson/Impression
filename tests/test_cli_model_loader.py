from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import pytest

import impression.modeling as modeling
from impression.cli import _drop_owned_user_model_modules, _load_module
from impression.modeling.surface import SurfaceBody


@pytest.fixture(autouse=True)
def _clean_user_model_modules():
    _drop_owned_user_model_modules()
    yield
    _drop_owned_user_model_modules()


def test_model_load_preserves_canonical_impression_module_and_class_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.py"
    source.write_text(
        "from impression.modeling import make_box\n\n"
        "def build():\n"
        "    return make_box(size=(1.0, 2.0, 3.0))\n"
    )
    canonical_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "impression.modeling" or name.startswith("impression.modeling.")
    }

    loaded = _load_module(source)
    result = loaded.build()

    assert isinstance(result, SurfaceBody)
    assert type(result) is SurfaceBody
    assert modeling.SurfaceBody is SurfaceBody
    assert all(sys.modules[name] is module for name, module in canonical_modules.items())


def test_sequential_loads_refresh_model_owned_local_helpers(tmp_path: Path) -> None:
    helper = tmp_path / "loader_owned_helper.py"
    model = tmp_path / "model.py"
    helper.write_text("VALUE = 1\n")
    model.write_text(
        "from loader_owned_helper import VALUE\n\n"
        "def build():\n"
        "    return VALUE\n"
    )

    first = _load_module(model)
    helper.write_text("VALUE = 2\n")
    second = _load_module(model)

    assert first.build() == 1
    assert second.build() == 2
    assert first is not second


def test_preloaded_unrelated_module_is_not_claimed_or_removed(tmp_path: Path) -> None:
    module_name = "loader_unrelated_sentinel"
    sentinel = ModuleType(module_name)
    sentinel.__file__ = str(tmp_path.parent / "unrelated" / f"{module_name}.py")
    sentinel.VALUE = "sentinel"
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = sentinel
    model = tmp_path / "model.py"
    model.write_text(
        f"import {module_name}\n\n"
        "def build():\n"
        f"    return {module_name}.VALUE\n"
    )
    try:
        loaded = _load_module(model)
        _load_module(model)

        assert loaded.build() == "sentinel"
        assert sys.modules[module_name] is sentinel
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def test_failed_model_load_removes_only_modules_owned_by_that_attempt(tmp_path: Path) -> None:
    helper_name = "loader_failed_helper"
    helper = tmp_path / f"{helper_name}.py"
    model = tmp_path / "model.py"
    helper.write_text("VALUE = 1\n")
    model.write_text(
        f"import {helper_name}\n"
        "raise RuntimeError('controlled import failure')\n"
    )
    canonical_modeling = sys.modules["impression.modeling"]

    with pytest.raises(RuntimeError, match="controlled import failure"):
        _load_module(model)

    assert "impression_user_model" not in sys.modules
    assert helper_name not in sys.modules
    assert sys.modules["impression.modeling"] is canonical_modeling

    model.write_text(
        "from impression.modeling import make_box\n\n"
        "def build():\n"
        "    return make_box()\n"
    )
    recovered = _load_module(model).build()
    assert isinstance(recovered, SurfaceBody)
