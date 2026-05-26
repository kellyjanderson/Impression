from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from impression.mesh import Mesh, triangulate_faces


@dataclass(frozen=True)
class LegacyMeshPrimitiveHelperRecord:
    legacy_name: str
    owner_module: str
    classification: str = "explicit-mesh-compatibility"


LEGACY_MESH_PRIMITIVE_HELPERS: tuple[LegacyMeshPrimitiveHelperRecord, ...] = (
    LegacyMeshPrimitiveHelperRecord("_box_mesh", __name__),
    LegacyMeshPrimitiveHelperRecord("_circular_frustum_mesh", __name__),
    LegacyMeshPrimitiveHelperRecord("_rectangular_frustum_mesh", __name__),
)


def box_mesh(size: Sequence[float], center: Sequence[float]) -> Mesh:
    sx, sy, sz = size
    cx, cy, cz = center
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    points = np.array(
        [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ]
    )
    points += np.array([cx, cy, cz], dtype=float)
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
        ],
        dtype=int,
    )
    return Mesh(points, faces)


def circular_frustum_mesh(
    bottom_radius: float,
    top_radius: float,
    height: float,
    resolution: int,
    capping: bool = True,
) -> Mesh:
    resolution = max(int(resolution), 3)
    bottom_radius = max(bottom_radius, 0.0)
    top_radius = max(top_radius, 0.0)
    z_bottom = -height / 2.0
    z_top = height / 2.0
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    points = []
    faces: list[list[int]] = []

    def ring_points(radius: float, z: float) -> np.ndarray:
        return np.column_stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.full_like(angles, z),
            ]
        )

    bottom_has_ring = bottom_radius > 0
    top_has_ring = top_radius > 0

    if bottom_has_ring:
        bottom = ring_points(bottom_radius, z_bottom)
        bottom_indices = np.arange(len(bottom))
        points.append(bottom)
    else:
        bottom = np.array([[0.0, 0.0, z_bottom]])
        bottom_indices = np.array([0])
        points.append(bottom)

    if top_has_ring:
        top = ring_points(top_radius, z_top)
        top_indices = np.arange(len(points[0]), len(points[0]) + len(top))
        points.append(top)
    else:
        top = np.array([[0.0, 0.0, z_top]])
        top_indices = np.array([len(points[0])])
        points.append(top)

    points_arr = np.vstack(points)

    if bottom_has_ring and top_has_ring:
        count = resolution
        for i in range(count):
            j = (i + 1) % count
            faces.append([bottom_indices[i], bottom_indices[j], top_indices[j], top_indices[i]])
        if capping:
            faces.append(list(bottom_indices[::-1]))
            faces.append(list(top_indices))
    elif bottom_has_ring:
        apex = int(top_indices[0])
        count = resolution
        for i in range(count):
            j = (i + 1) % count
            faces.append([bottom_indices[i], bottom_indices[j], apex])
        if capping:
            faces.append(list(bottom_indices[::-1]))
    else:
        apex = int(bottom_indices[0])
        count = resolution
        for i in range(count):
            j = (i + 1) % count
            faces.append([apex, top_indices[j], top_indices[i]])
        if capping:
            faces.append(list(top_indices))

    faces_arr = triangulate_faces(faces)
    return Mesh(points_arr, faces_arr)


def rectangular_frustum_mesh(
    base_size: Tuple[float, float],
    top_size: Tuple[float, float],
    height: float,
) -> Mesh:
    hx, hy = base_size[0] / 2.0, base_size[1] / 2.0
    tx, ty = top_size[0] / 2.0, top_size[1] / 2.0
    z_bottom = -height / 2.0
    z_top = height / 2.0

    bottom_pts = np.array(
        [
            (-hx, -hy, z_bottom),
            (hx, -hy, z_bottom),
            (hx, hy, z_bottom),
            (-hx, hy, z_bottom),
        ]
    )

    if top_size[0] == 0 and top_size[1] == 0:
        top_pts = np.array([[0.0, 0.0, z_top]])
        apex_only = True
    else:
        top_pts = np.array(
            [
                (-tx, -ty, z_top),
                (tx, -ty, z_top),
                (tx, ty, z_top),
                (-tx, ty, z_top),
            ]
        )
        apex_only = False

    points = np.vstack([bottom_pts, top_pts])
    faces: list[list[int]] = []
    bottom_indices = np.arange(4)

    if apex_only:
        apex_idx = 4
        for i in range(4):
            j = (i + 1) % 4
            faces.append([bottom_indices[i], bottom_indices[j], apex_idx])
        faces.append(list(bottom_indices[::-1]))
    else:
        top_indices = np.arange(4, 8)
        for i in range(4):
            j = (i + 1) % 4
            faces.append([bottom_indices[i], bottom_indices[j], top_indices[j], top_indices[i]])
        faces.append(list(bottom_indices[::-1]))
        faces.append(list(top_indices))

    faces_arr = triangulate_faces(faces)
    return Mesh(points, faces_arr)
