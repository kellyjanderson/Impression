from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from impression.mesh import Mesh
from impression.modeling.group import MeshGroup


def translate(mesh: Mesh | MeshGroup, offset: Sequence[float]) -> Mesh | MeshGroup:
    """Translate the mesh/group in-place and return it."""
    if isinstance(mesh, MeshGroup):
        mesh.translate(offset)
        return mesh
    vec = np.asarray(offset, dtype=float).reshape(3)
    mesh.translate(vec, inplace=True)
    return mesh


def rotate(
    mesh: Mesh | MeshGroup,
    axis: Sequence[float],
    angle_deg: float | None = None,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    order: str = "xyz",
) -> Mesh:
    """Rotate around an axis or Euler angles (degrees)."""
    if angle_deg is None:
        return rotate_euler(mesh, angles_deg=axis, origin=origin, order=order)
    if isinstance(mesh, MeshGroup):
        mesh.rotate(axis=axis, angle_deg=angle_deg, origin=origin)
        return mesh
    axis_vec = np.asarray(axis, dtype=float).reshape(3)
    norm = np.linalg.norm(axis_vec)
    if norm == 0:
        raise ValueError("Rotation axis must be non-zero.")
    axis_vec = axis_vec / norm
    center = np.asarray(origin, dtype=float).reshape(3)

    mesh.rotate_vector(axis_vec, angle_deg, point=tuple(center), inplace=True)
    return mesh


def rotate_euler(
    mesh: Mesh | MeshGroup,
    angles_deg: Sequence[float],
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    order: str = "xyz",
) -> Mesh | MeshGroup:
    """Rotate by Euler angles in degrees (default order: X then Y then Z)."""
    angles = np.asarray(angles_deg, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    order = order.lower()
    if set(order) != {"x", "y", "z"}:
        raise ValueError("order must be a permutation of 'xyz'.")

    matrices = {
        "x": _axis_rotation_matrix((1.0, 0.0, 0.0), angles[0]),
        "y": _axis_rotation_matrix((0.0, 1.0, 0.0), angles[1]),
        "z": _axis_rotation_matrix((0.0, 0.0, 1.0), angles[2]),
    }
    mat = np.eye(4)
    for axis in order:
        mat = matrices[axis] @ mat
    mat = _apply_about_origin(mat, origin)

    return multmatrix(mesh, mat)


def scale(
    mesh: Mesh | MeshGroup,
    factors: Sequence[float],
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> Mesh | MeshGroup:
    """Scale the mesh/group in-place about an origin."""
    sx, sy, sz = np.asarray(factors, dtype=float).reshape(3)
    mat = np.eye(4)
    mat[0, 0] = sx
    mat[1, 1] = sy
    mat[2, 2] = sz
    mat = _apply_about_origin(mat, np.asarray(origin, dtype=float).reshape(3))
    return multmatrix(mesh, mat)


def resize(
    mesh: Mesh | MeshGroup,
    size: Sequence[float],
    auto: bool | Sequence[bool] = False,
) -> Mesh | MeshGroup:
    """Resize mesh/group to the target size, preserving aspect on auto axes."""
    target = np.asarray(size, dtype=float).reshape(3)
    auto_mask = _normalize_auto(auto)
    bounds = _bounds(mesh)
    extents = np.array(
        [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]],
        dtype=float,
    )

    scales = np.ones(3, dtype=float)
    for i in range(3):
        if target[i] <= 0 or auto_mask[i]:
            continue
        if extents[i] == 0:
            raise ValueError("Cannot resize along axis with zero extent.")
        scales[i] = target[i] / extents[i]

    anchor = next((scales[i] for i in range(3) if not auto_mask[i] and target[i] > 0), 1.0)
    for i in range(3):
        if target[i] <= 0 or auto_mask[i]:
            scales[i] = anchor

    center = np.array(
        [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0],
        dtype=float,
    )
    return scale(mesh, scales, origin=center)


def mirror(mesh: Mesh | MeshGroup, axis: Sequence[float]) -> Mesh | MeshGroup:
    """Mirror across the plane through the origin with the given normal."""
    axis_vec = np.asarray(axis, dtype=float).reshape(3)
    norm = np.linalg.norm(axis_vec)
    if norm == 0:
        raise ValueError("Mirror axis must be non-zero.")
    axis_vec = axis_vec / norm
    mat = np.eye(4)
    mat[:3, :3] -= 2.0 * np.outer(axis_vec, axis_vec)
    return multmatrix(mesh, mat)


def multmatrix(mesh: Mesh | MeshGroup, matrix: Sequence[Sequence[float]]) -> Mesh | MeshGroup:
    """Apply a 4x4 transform matrix."""
    mat = np.asarray(matrix, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError("multmatrix requires a 4x4 matrix.")
    if isinstance(mesh, MeshGroup):
        mesh.multmatrix(mat)
        return mesh
    mesh.transform(mat, inplace=True)
    return mesh


def _axis_rotation_matrix(axis: Sequence[float], angle_deg: float) -> np.ndarray:
    axis_vec = np.asarray(axis, dtype=float).reshape(3)
    norm = np.linalg.norm(axis_vec)
    if norm == 0:
        raise ValueError("Rotation axis must be non-zero.")
    axis_vec = axis_vec / norm
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    x, y, z = axis_vec
    C = 1.0 - c
    rot = np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s, 0.0],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s, 0.0],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rot


def _apply_about_origin(matrix: np.ndarray, origin: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=float)
    to_origin = np.eye(4)
    to_origin[:3, 3] = -origin
    back = np.eye(4)
    back[:3, 3] = origin
    return back @ mat @ to_origin


def _normalize_auto(auto: bool | Sequence[bool]) -> np.ndarray:
    if isinstance(auto, (list, tuple, np.ndarray)):
        if len(auto) != 3:
            raise ValueError("auto must be a bool or 3-length sequence.")
        return np.array([bool(x) for x in auto], dtype=bool)
    return np.array([bool(auto)] * 3, dtype=bool)


def _bounds(mesh: Mesh | MeshGroup) -> tuple[float, float, float, float, float, float]:
    if isinstance(mesh, Mesh):
        return mesh.bounds
    meshes = mesh.to_meshes()
    if not meshes:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    for m in meshes:
        b = m.bounds
        mins = np.minimum(mins, np.array([b[0], b[2], b[4]]))
        maxs = np.maximum(maxs, np.array([b[1], b[3], b[5]]))
    return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))
