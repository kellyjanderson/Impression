"""CAD backend helpers (currently build123d) that return PyVista meshes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:  # pragma: no cover - typing only
    from build123d import Shape


def shape_to_polydata(shape: "Shape", tolerance: float = 0.05) -> pv.PolyData:
    """Convert a build123d shape into a cleaned ``pyvista.PolyData`` mesh.

    Parameters
    ----------
    shape:
        Any build123d topology object (Sketch, Face, Solid, Part, Compound, ...)
        that supports ``tessellate``. The tessellated triangles are re-packed
        into VTK's face array layout.
    tolerance:
        Linear deflection passed to ``tessellate``. Smaller values capture more
        curvature detail at the expense of more triangles.
    """

    if not hasattr(shape, "tessellate"):
        raise TypeError("shape_to_polydata expects a build123d shape with tessellate().")

    vertices, triangles = shape.tessellate(tolerance)
    if not vertices or not triangles:
        raise ValueError("tessellation returned no geometry; check the input shape or tolerance.")

    points = np.asarray([[vec.X, vec.Y, vec.Z] for vec in vertices], dtype=float)
    face_chunks = [np.array([3, tri[0], tri[1], tri[2]], dtype=np.int64) for tri in triangles]
    faces = np.hstack(face_chunks)
    mesh = pv.PolyData(points, faces)
    return mesh.clean()


__all__ = ["shape_to_polydata"]
