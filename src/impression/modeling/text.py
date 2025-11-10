from __future__ import annotations

from typing import Literal, Sequence, Tuple

import numpy as np
import pyvista as pv

from ._color import set_mesh_color

Justify = Literal["left", "center", "right"]


def make_text(
    content: str,
    depth: float = 0.2,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    font_size: float = 1.0,
    justify: Justify = "center",
    color: Sequence[float] | str | None = None,
) -> pv.PolyData:
    """Create a text mesh that can be previewed/exported like other primitives."""

    if not content:
        raise ValueError("content must be a non-empty string")

    extrude = depth > 0
    kwargs: dict = {}
    if extrude:
        kwargs["depth"] = depth
    text = pv.Text3D(content, extrude=extrude, conn=True, **kwargs)

    if font_size != 1.0:
        text.scale(font_size, inplace=True)

    text = _justify_and_center(text, justify)
    text = _orient_mesh(text, direction)
    text.translate(center, inplace=True)

    if color is not None:
        set_mesh_color(text, color)

    return text


def _justify_and_center(mesh: pv.PolyData, justify: Justify) -> pv.PolyData:
    bounds = mesh.bounds
    min_x, max_x, min_y, max_y, min_z, max_z = bounds

    if justify == "left":
        anchor_x = min_x
    elif justify == "right":
        anchor_x = max_x
    else:
        anchor_x = (min_x + max_x) / 2.0

    offset = (
        anchor_x,
        (min_y + max_y) / 2.0,
        (min_z + max_z) / 2.0,
    )
    mesh.translate((-offset[0], -offset[1], -offset[2]), inplace=True)
    return mesh


def _orient_mesh(mesh: pv.PolyData, direction: Sequence[float]) -> pv.PolyData:
    target = _normalize(direction)
    default = np.array([0.0, 0.0, 1.0])
    target_vec = np.array(target)

    if np.allclose(target_vec, default):
        return mesh

    axis = np.cross(default, target_vec)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        axis = np.array([1.0, 0.0, 0.0])
        angle_deg = 180.0
    else:
        axis = axis / axis_norm
        angle_rad = np.arccos(np.clip(np.dot(default, target_vec), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

    mesh = mesh.copy()
    mesh.rotate_vector(axis, angle_deg, point=(0.0, 0.0, 0.0), inplace=True)
    return mesh


def _normalize(vector: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Direction vector must be non-zero.")
    arr = arr / norm
    return float(arr[0]), float(arr[1]), float(arr[2])
