from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def _load_extension() -> ModuleType:
    package_name = "impression_threading"
    try:
        return importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        if exc.name != package_name:
            raise

    sibling_src = Path(__file__).resolve().parents[4] / "impression-threading" / "src"
    if sibling_src.exists():
        sys.path.insert(0, str(sibling_src))
        return importlib.import_module(package_name)

    raise ModuleNotFoundError(
        "impression.modeling.threading now lives in the sibling "
        "impression-threading project. Install it with "
        "`python -m pip install -e ../impression-threading`."
    )


_extension = _load_extension()
__all__ = list(getattr(_extension, "__all__", ()))
globals().update({name: getattr(_extension, name) for name in __all__})


def make_external_thread(spec, *args, backend="surface", **kwargs):
    return _extension.make_external_thread(spec, *args, backend=backend, **kwargs)


def make_internal_thread(spec, *args, backend="surface", **kwargs):
    return _extension.make_internal_thread(spec, *args, backend=backend, **kwargs)


def make_threaded_rod(spec, *args, backend="surface", **kwargs):
    return _extension.make_threaded_rod(spec, *args, backend=backend, **kwargs)


def make_tapped_hole_cutter(spec, *args, backend="surface", **kwargs):
    return _extension.make_tapped_hole_cutter(spec, *args, backend=backend, **kwargs)


globals().update(
    {
        "make_external_thread": make_external_thread,
        "make_internal_thread": make_internal_thread,
        "make_threaded_rod": make_threaded_rod,
        "make_tapped_hole_cutter": make_tapped_hole_cutter,
    }
)
