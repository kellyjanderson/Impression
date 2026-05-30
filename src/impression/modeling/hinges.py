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


def hinge_feature_csg_dependencies():
    from .csg import SurfaceCSGFeatureDependencyRecord

    return (
        SurfaceCSGFeatureDependencyRecord(
            caller_id="hinges.make_traditional_hinge_leaf",
            module="impression.modeling.hinges",
            operation=None,
            surface_builder="make_traditional_hinge_leaf",
            explicit_mesh_route="backend='mesh'",
        ),
        SurfaceCSGFeatureDependencyRecord(
            caller_id="hinges.make_traditional_hinge_pair",
            module="impression.modeling.hinges",
            operation="union",
            surface_builder="make_traditional_hinge_pair",
            explicit_mesh_route="backend='mesh'",
        ),
        SurfaceCSGFeatureDependencyRecord(
            caller_id="hinges.make_living_hinge",
            module="impression.modeling.hinges",
            operation=None,
            surface_builder="make_living_hinge",
            explicit_mesh_route="backend='mesh'",
        ),
        SurfaceCSGFeatureDependencyRecord(
            caller_id="hinges.make_bistable_hinge",
            module="impression.modeling.hinges",
            operation=None,
            surface_builder="make_bistable_hinge",
            explicit_mesh_route="backend='mesh'",
        ),
    )


if "hinge_feature_csg_dependencies" not in __all__:
    __all__.append("hinge_feature_csg_dependencies")
