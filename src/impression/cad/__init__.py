"""CAD backend helpers (currently disabled)."""

from __future__ import annotations

from impression.mesh import Mesh


# CAD tessellation is currently disabled.

def shape_to_polydata(shape: object, tolerance: float = 0.05) -> Mesh:
    raise RuntimeError("shape_to_polydata is disabled in this build.")


__all__ = ["shape_to_polydata"]
