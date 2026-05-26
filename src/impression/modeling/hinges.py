from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def _load_extension() -> ModuleType:
    package_name = "impression_hinges"
    try:
        return importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        if exc.name != package_name:
            raise

    sibling_src = Path(__file__).resolve().parents[4] / "impression-hinges" / "src"
    if sibling_src.exists():
        sys.path.insert(0, str(sibling_src))
        return importlib.import_module(package_name)

    raise ModuleNotFoundError(
        "impression.modeling.hinges now lives in the sibling "
        "impression-hinges project. Install it with "
        "`python -m pip install -e ../impression-hinges`."
    )


_extension = _load_extension()
__all__ = list(getattr(_extension, "__all__", ()))
globals().update({name: getattr(_extension, name) for name in __all__})


def make_traditional_hinge_leaf(*args, backend="surface", **kwargs):
    if backend == "mesh":
        return _call_with_legacy_mesh_primitives(_extension.make_traditional_hinge_leaf, *args, backend=backend, **kwargs)
    return _extension.make_traditional_hinge_leaf(*args, backend=backend, **kwargs)


def make_traditional_hinge_pair(*args, backend="surface", **kwargs):
    if backend == "mesh":
        return _call_with_legacy_mesh_primitives(_extension.make_traditional_hinge_pair, *args, backend=backend, **kwargs)
    return _extension.make_traditional_hinge_pair(*args, backend=backend, **kwargs)


def make_living_hinge(*args, backend="surface", **kwargs):
    if backend == "mesh":
        return _call_with_legacy_mesh_primitives(_extension.make_living_hinge, *args, backend=backend, **kwargs)
    return _extension.make_living_hinge(*args, backend=backend, **kwargs)


def make_bistable_hinge(*args, backend="surface", **kwargs):
    if backend == "mesh":
        return _call_with_legacy_mesh_primitives(_extension.make_bistable_hinge, *args, backend=backend, **kwargs)
    return _extension.make_bistable_hinge(*args, backend=backend, **kwargs)


def _call_with_legacy_mesh_primitives(fn, *args, **kwargs):
    from .primitives import make_box_mesh, make_cylinder_mesh

    globals_dict = getattr(fn, "__globals__", {})
    original_box = globals_dict.get("make_box")
    original_cylinder = globals_dict.get("make_cylinder")
    globals_dict["make_box"] = make_box_mesh
    globals_dict["make_cylinder"] = make_cylinder_mesh
    try:
        return fn(*args, **kwargs)
    finally:
        if original_box is not None:
            globals_dict["make_box"] = original_box
        if original_cylinder is not None:
            globals_dict["make_cylinder"] = original_cylinder


globals().update(
    {
        "make_traditional_hinge_leaf": make_traditional_hinge_leaf,
        "make_traditional_hinge_pair": make_traditional_hinge_pair,
        "make_living_hinge": make_living_hinge,
        "make_bistable_hinge": make_bistable_hinge,
    }
)
