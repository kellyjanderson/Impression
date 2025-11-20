from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

_VTK_PATCH_FLAG = "IMPRESSION_SKIP_VTK_PATCHES"
_VTK_PATCH_RAN = False


def ensure_vtk_runtime() -> None:
    """Apply minimal VTK runtime patches to avoid duplicate/missing dylibs."""

    global _VTK_PATCH_RAN
    if _VTK_PATCH_RAN:
        return
    if os.environ.get(_VTK_PATCH_FLAG, "").lower() in {"1", "true", "yes"}:
        _VTK_PATCH_RAN = True
        return

    os.environ.setdefault("VTK_PYTHON_ALLOW_DUPLICATE_LIBS", "1")
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

    try:
        spec = importlib.util.find_spec("vtkmodules")
        if not spec or not spec.submodule_search_locations:
            _VTK_PATCH_RAN = True
            return
        base_path = Path(next(iter(spec.submodule_search_locations)))
        dylib_dir = base_path / ".dylibs"
        if not dylib_dir.is_dir():
            _VTK_PATCH_RAN = True
            return
    except Exception:
        _VTK_PATCH_RAN = True
        return

    duplicates = [
        "libvtkRenderingUI",
        "libvtkRenderingOpenGL2",
    ]
    for base_name in duplicates:
        canonical = dylib_dir / f"{base_name}.dylib"
        disabled = canonical.with_suffix(canonical.suffix + ".disabled")
        numbered = sorted(dylib_dir.glob(f"{base_name}-*.dylib"))
        if canonical.exists():
            continue
        if disabled.exists():
            try:
                disabled.rename(canonical)
                continue
            except OSError:
                pass
        if numbered:
            try:
                canonical.symlink_to(numbered[-1].name)
            except OSError:
                pass

    try:
        importlib.import_module("vtkmodules.vtkRenderingOpenGL2")
    except Exception:
        pass

    _VTK_PATCH_RAN = True
