from __future__ import annotations

from typing import Sequence

import numpy as np
import pyvista as pv


def translate(mesh: pv.DataSet, offset: Sequence[float]) -> pv.DataSet:
    """Return a translated copy of the mesh."""
    vec = np.asarray(offset, dtype=float).reshape(3)
    shifted = mesh.copy()
    shifted.translate(vec, inplace=True)
    return shifted


def rotate(
    mesh: pv.DataSet,
    axis: Sequence[float],
    angle_deg: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> pv.DataSet:
    """Return a rotated copy of the mesh around an arbitrary axis."""
    axis_vec = np.asarray(axis, dtype=float).reshape(3)
    norm = np.linalg.norm(axis_vec)
    if norm == 0:
        raise ValueError("Rotation axis must be non-zero.")
    axis_vec = axis_vec / norm
    center = np.asarray(origin, dtype=float).reshape(3)

    rotated = mesh.copy()
    rotated.rotate_vector(axis_vec, angle_deg, point=tuple(center), inplace=True)
    return rotated
