from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np

from impression.mesh import Mesh, combine_meshes


def _normalize_axis(axis: Sequence[float]) -> np.ndarray:
    vec = np.asarray(axis, dtype=float).reshape(3)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Rotation axis must be non-zero.")
    return vec / norm


def _translation_matrix(offset: Sequence[float]) -> np.ndarray:
    dx, dy, dz = np.asarray(offset, dtype=float).reshape(3)
    mat = np.eye(4)
    mat[:3, 3] = [dx, dy, dz]
    return mat


def _rotation_matrix(axis: Sequence[float], angle_deg: float, origin: Sequence[float]) -> np.ndarray:
    axis_vec = _normalize_axis(axis)
    angle_rad = np.deg2rad(angle_deg)
    x, y, z = axis_vec
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
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
    origin = np.asarray(origin, dtype=float).reshape(3)
    to_origin = _translation_matrix(-origin)
    back = _translation_matrix(origin)
    return back @ rot @ to_origin


@dataclass
class MeshGroup:
    """Hold multiple meshes and apply shared transforms."""

    meshes: List[Mesh] = field(default_factory=list)
    _transform: np.ndarray = field(default_factory=lambda: np.eye(4))

    def add(self, mesh: Mesh) -> "MeshGroup":
        self.meshes.append(mesh)
        return self

    def translate(self, offset: Sequence[float]) -> "MeshGroup":
        self._transform = self._transform @ _translation_matrix(offset)
        return self

    def rotate(
        self,
        axis: Sequence[float],
        angle_deg: float,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "MeshGroup":
        self._transform = self._transform @ _rotation_matrix(axis, angle_deg, origin)
        return self

    def scale(self, factors: Sequence[float]) -> "MeshGroup":
        sx, sy, sz = np.asarray(factors, dtype=float).reshape(3)
        mat = np.eye(4)
        mat[0, 0] = sx
        mat[1, 1] = sy
        mat[2, 2] = sz
        self._transform = self._transform @ mat
        return self

    def _apply_transform(self, mesh: Mesh) -> Mesh:
        return mesh.transform(self._transform, inplace=False)

    def to_meshes(self) -> list[Mesh]:
        return [self._apply_transform(mesh) for mesh in self.meshes]

    def to_mesh(self) -> Mesh:
        return combine_meshes(self.to_meshes())


def group(meshes: Iterable[Mesh]) -> MeshGroup:
    grp = MeshGroup()
    for m in meshes:
        grp.add(m)
    return grp
