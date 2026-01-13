"""CAD backend helpers (currently build123d) that return internal meshes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from impression.mesh import Mesh

if TYPE_CHECKING:  # pragma: no cover - typing only
    from build123d import Shape


def shape_to_polydata(shape: "Shape", tolerance: float = 0.05) -> Mesh:
    """Convert a build123d shape into a triangle mesh.

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
    faces_arr = faces.reshape(-1, 4)[:, 1:]
    return Mesh(points, faces_arr)


__all__ = ["shape_to_polydata"]
