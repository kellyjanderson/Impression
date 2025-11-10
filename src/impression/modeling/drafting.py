from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np
import pyvista as pv

from ._color import set_mesh_color
from .primitives import _orient_mesh, _normalize

Axis = Literal["x", "y", "z"]


def make_line(
    start: Sequence[float],
    end: Sequence[float],
    thickness: float = 0.02,
    color: Sequence[float] | str | None = None,
) -> pv.PolyData:
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Line requires two distinct points.")

    half = thickness / 2.0
    base = np.array(
        [
            (-half, -half, 0),
            (half, -half, 0),
            (half, half, 0),
            (-half, half, 0),
        ]
    )
    top = base + np.array((0, 0, length))
    points = np.vstack([base, top])
    faces = np.array(
        [
            [4, 0, 1, 2, 3],
            [4, 4, 5, 6, 7],
            [4, 0, 1, 5, 4],
            [4, 1, 2, 6, 5],
            [4, 2, 3, 7, 6],
            [4, 3, 0, 4, 7],
        ]
    )
    mesh = pv.PolyData(points, faces)
    mesh = _orient_mesh(mesh, _normalize(direction))
    mesh.translate(start, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_plane(
    size: Sequence[float] = (1.0, 1.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
    normal: Sequence[float] = (0.0, 0.0, 1.0),
    color: Sequence[float] | str | None = None,
) -> pv.PolyData:
    sx, sy = size
    half_x, half_y = sx / 2.0, sy / 2.0
    points = np.array(
        [
            (-half_x, -half_y, 0.0),
            (half_x, -half_y, 0.0),
            (half_x, half_y, 0.0),
            (-half_x, half_y, 0.0),
        ]
    )
    faces = np.array([[4, 0, 1, 2, 3]])
    mesh = pv.PolyData(points, faces)
    mesh = _orient_mesh(mesh, normal)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_arrow(
    start: Sequence[float],
    end: Sequence[float],
    shaft_diameter: float = 0.04,
    head_length: float = 0.15,
    head_diameter: float = 0.12,
    color: Sequence[float] | str | None = None,
) -> pv.PolyData:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Arrow requires distinct start/end points.")
    head_length = min(head_length, length * 0.5)
    shaft_length = length - head_length

    shaft = make_line(start, start + direction * (shaft_length / length), thickness=shaft_diameter, color=color)
    head_height = head_length
    base = np.array(
        [
            (-head_diameter / 2.0, -head_diameter / 2.0, 0),
            (head_diameter / 2.0, -head_diameter / 2.0, 0),
            (head_diameter / 2.0, head_diameter / 2.0, 0),
            (-head_diameter / 2.0, head_diameter / 2.0, 0),
            (0, 0, head_height),
        ]
    )
    faces = np.array(
        [
            [4, 0, 1, 2, 3],
            [3, 0, 1, 4],
            [3, 1, 2, 4],
            [3, 2, 3, 4],
            [3, 3, 0, 4],
        ]
    )
    head = pv.PolyData(base, faces)
    head = _orient_mesh(head, direction / length)
    head.translate(start + direction * (shaft_length / length), inplace=True)
    if color is not None:
        set_mesh_color(head, color)
    return shaft.merge(head)


def make_dimension(
    start: Sequence[float],
    end: Sequence[float],
    offset: float = 0.1,
    text: str | None = None,
    color: Sequence[float] | str | None = None,
) -> list[pv.PolyData]:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Dimension requires distinct points.")
    norm_dir = direction / length
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, norm_dir)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, norm_dir)
    right /= np.linalg.norm(right)
    offset_vec = right * offset
    arrow_start = start + offset_vec
    arrow_end = end + offset_vec

    meshes = [make_arrow(arrow_start, arrow_end, color=color)]

    if text is None:
        text = f"{length:.2f}"
    mid = (arrow_start + arrow_end) / 2.0
    text_mesh = pv.Text3D(text, depth=0.0)
    text_mesh.scale(0.1, inplace=True)
    text_mesh = _orient_mesh(text_mesh, up)
    text_mesh.translate(mid + up * 0.02, inplace=True)
    if color is not None:
        set_mesh_color(text_mesh, color)
    meshes.append(text_mesh)
    return meshes
