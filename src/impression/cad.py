from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pyvista as pv
from build123d import Shape


def shape_to_polydata(
    shape: Shape,
    tolerance: float = 0.2,
) -> pv.PolyData:
    """Convert a build123d Shape into a PyVista PolyData mesh."""

    vertices: Sequence = []
    faces: Sequence[tuple[int, int, int]] = []
    try:
        vertices, faces = shape.tessellate(tolerance)
    except Exception as exc:  # pragma: no cover - surfaces may fail to tessellate
        raise RuntimeError(f"Failed to tessellate CAD shape: {exc}") from exc

    if not vertices or not faces:
        return pv.PolyData()

    points = np.array([[vec.X, vec.Y, vec.Z] for vec in vertices], dtype=float)
    face_stream: list[int] = []
    for tri in faces:
        face_stream.extend((3, tri[0], tri[1], tri[2]))

    mesh = pv.PolyData(points, np.array(face_stream))
    mesh.clean(inplace=True)
    return mesh
