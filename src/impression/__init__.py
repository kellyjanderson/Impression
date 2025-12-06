"""Impression – parametric modeling playground."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from ._env import ensure_shell_env

__all__ = ["__version__"]

__version__ = "0.1.0"

# Allow VTK to load side-by-side libs without crashing and repair any missing symlinked dylibs.
os.environ.setdefault("VTK_PYTHON_ALLOW_DUPLICATE_LIBS", "1")


def _restore_vtk_dylibs() -> None:
    try:
        spec = importlib.util.find_spec("vtkmodules")
        if not spec or not spec.submodule_search_locations:
            return
        base_path = Path(next(iter(spec.submodule_search_locations)))
        dylib_dir = base_path / ".dylibs"
        if not dylib_dir.is_dir():
            return
    except Exception:
        return

    for base in ("libvtkRenderingUI", "libvtkRenderingOpenGL2"):
        canonical = dylib_dir / f"{base}.dylib"
        disabled = canonical.with_suffix(canonical.suffix + ".disabled")
        numbered = sorted(dylib_dir.glob(f"{base}-*.dylib"), reverse=True)
        primary = numbered[0] if numbered else None

        if canonical.is_symlink():
            orig = canonical.with_suffix(canonical.suffix + ".orig")
            if orig.exists():
                try:
                    canonical.unlink()
                    orig.rename(canonical)
                except OSError:
                    pass

        if not canonical.exists() and disabled.exists():
            try:
                disabled.rename(canonical)
            except OSError:
                pass

        if primary and not canonical.exists():
            try:
                canonical.symlink_to(primary.name)
            except OSError:
                pass


_restore_vtk_dylibs()
ensure_shell_env()
