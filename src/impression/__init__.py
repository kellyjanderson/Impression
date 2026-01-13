"""Impression – parametric modeling playground."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

from ._env import ensure_shell_env
from ._config import ensure_user_config

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

    def has_symbol(path: Path, symbol: str) -> bool:
        try:
            output = subprocess.check_output(
                ["nm", "-gU", str(path)],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return False
        return symbol in output

    def pick_primary(base_name: str) -> Path | None:
        canonical = dylib_dir / f"{base_name}.dylib"
        orig = canonical.with_suffix(canonical.suffix + ".orig")
        disabled = canonical.with_suffix(canonical.suffix + ".disabled")
        numbered = sorted(dylib_dir.glob(f"{base_name}-*.dylib"), reverse=True)

        candidates = []
        if orig.exists():
            candidates.append(orig)
        if canonical.exists() and not canonical.is_symlink():
            candidates.append(canonical)
        candidates.extend(numbered)

        if base_name == "libvtkRenderingUI":
            for candidate in candidates:
                if has_symbol(candidate, "vtkCocoaAutoreleasePool"):
                    return candidate
        if candidates:
            return candidates[0]
        if disabled.exists():
            return disabled
        return None

    for base in ("libvtkRenderingUI", "libvtkRenderingOpenGL2"):
        canonical = dylib_dir / f"{base}.dylib"
        numbered = sorted(dylib_dir.glob(f"{base}-*.dylib"), reverse=True)
        primary = pick_primary(base)
        if primary is None:
            continue

        targets = [canonical] + numbered
        for target in targets:
            try:
                if target.exists() and not target.is_symlink():
                    if target.resolve() == primary.resolve():
                        continue
            except OSError:
                pass

            if target.exists() and not target.is_symlink():
                backup = target.with_suffix(target.suffix + ".orig")
                try:
                    if not backup.exists():
                        target.rename(backup)
                    else:
                        target.rename(target.with_suffix(target.suffix + ".disabled"))
                except OSError:
                    pass
            try:
                if target.exists() or target.is_symlink():
                    target.unlink()
            except OSError:
                pass
            try:
                target.symlink_to(primary.name)
            except OSError:
                pass


_restore_vtk_dylibs()
ensure_shell_env()
ensure_user_config()
