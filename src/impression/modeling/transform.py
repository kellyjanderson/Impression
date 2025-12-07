from __future__ import annotations

from typing import Sequence

import numpy as np
import pyvista as pv


def translate(mesh: pv.DataSet, offset: Sequence[float]) -> pv.DataSet:
    """Translate the mesh in-place and return it."""
    vec = np.asarray(offset, dtype=float).reshape(3)
    mesh.translate(vec, inplace=True)
    return mesh


def rotate(
    mesh: pv.DataSet,
    axis: Sequence[float],
    angle_deg: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> pv.DataSet:
    """Rotate the mesh in-place around an arbitrary axis and return it."""
    axis_vec = np.asarray(axis, dtype=float).reshape(3)
    norm = np.linalg.norm(axis_vec)
    if norm == 0:
        raise ValueError("Rotation axis must be non-zero.")
    axis_vec = axis_vec / norm
    center = np.asarray(origin, dtype=float).reshape(3)

    mesh.rotate_vector(axis_vec, angle_deg, point=tuple(center), inplace=True)
    return mesh
