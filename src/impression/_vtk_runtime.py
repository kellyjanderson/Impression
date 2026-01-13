from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
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

    for base_name in duplicates:
        canonical = dylib_dir / f"{base_name}.dylib"
        numbered = sorted(dylib_dir.glob(f"{base_name}-*.dylib"), reverse=True)
        primary = pick_primary(base_name)
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

    try:
        importlib.import_module("vtkmodules.vtkRenderingOpenGL2")
    except Exception:
        pass

    _VTK_PATCH_RAN = True
