from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np

from impression.mesh import Mesh, combine_meshes

MESH_GROUP_COMPATIBILITY_CLASSIFICATION = "mesh-compatibility"
MESH_GROUP_COMPATIBILITY_BOUNDARY = "explicit-mesh-group-compatibility"


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


def _scale_matrix(factors: Sequence[float], origin: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    sx, sy, sz = np.asarray(factors, dtype=float).reshape(3)
    mat = np.eye(4)
    mat[0, 0] = sx
    mat[1, 1] = sy
    mat[2, 2] = sz
    origin = np.asarray(origin, dtype=float).reshape(3)
    to_origin = _translation_matrix(-origin)
    back = _translation_matrix(origin)
    return back @ mat @ to_origin


def _mirror_matrix(axis: Sequence[float]) -> np.ndarray:
    axis_vec = np.asarray(axis, dtype=float).reshape(3)
    norm = np.linalg.norm(axis_vec)
    if norm == 0:
        raise ValueError("Mirror axis must be non-zero.")
    axis_vec = axis_vec / norm
    mat = np.eye(4)
    mat[:3, :3] -= 2.0 * np.outer(axis_vec, axis_vec)
    return mat


def _axis_rotation_matrix(axis: Sequence[float], angle_deg: float) -> np.ndarray:
    axis_vec = _normalize_axis(axis)
    angle_rad = np.deg2rad(angle_deg)
    x, y, z = axis_vec
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s, 0.0],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s, 0.0],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _rotation_euler_matrix(angles_deg: Sequence[float], origin: Sequence[float], order: str = "xyz") -> np.ndarray:
    angles = np.asarray(angles_deg, dtype=float).reshape(3)
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
    origin = np.asarray(origin, dtype=float).reshape(3)
    to_origin = _translation_matrix(-origin)
    back = _translation_matrix(origin)
    return back @ mat @ to_origin


@dataclass(frozen=True)
class MeshGroupCompatibilityDiagnostic:
    """Diagnostic proving MeshGroup is an explicit mesh compatibility boundary."""

    target_type: str
    reason: str = "MeshGroup accepts mesh inputs only"
    boundary: str = MESH_GROUP_COMPATIBILITY_BOUNDARY
    classification: str = MESH_GROUP_COMPATIBILITY_CLASSIFICATION

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary": self.boundary,
            "classification": self.classification,
            "reason": self.reason,
            "target_type": self.target_type,
        }


class MeshGroupCompatibilityError(TypeError):
    """Raised when a non-mesh value crosses into MeshGroup compatibility APIs."""

    def __init__(self, diagnostic: MeshGroupCompatibilityDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            f"Unsupported {diagnostic.target_type} at MeshGroup compatibility boundary: {diagnostic.reason}."
        )


@dataclass
class MeshGroup:
    """Hold multiple meshes and apply shared transforms."""

    meshes: List[Mesh] = field(default_factory=list)
    _transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._transform = np.asarray(self._transform, dtype=float).reshape(4, 4)
        self.metadata = dict(self.metadata)
        self.metadata.setdefault("mesh_group_compatibility", self.compatibility_diagnostic().canonical_payload())
        for mesh in self.meshes:
            _ensure_mesh_input(mesh)

    @property
    def classification(self) -> str:
        return MESH_GROUP_COMPATIBILITY_CLASSIFICATION

    def compatibility_diagnostic(self) -> MeshGroupCompatibilityDiagnostic:
        return MeshGroupCompatibilityDiagnostic(target_type=type(self).__name__)

    def add(self, mesh: Mesh) -> "MeshGroup":
        _ensure_mesh_input(mesh)
        self.meshes.append(mesh)
        return self

    def translate(self, offset: Sequence[float]) -> "MeshGroup":
        self._transform = self._transform @ _translation_matrix(offset)
        return self

    def rotate(
        self,
        axis: Sequence[float],
        angle_deg: float | None = None,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        order: str = "xyz",
    ) -> "MeshGroup":
        if angle_deg is None:
            self._transform = self._transform @ _rotation_euler_matrix(axis, origin, order=order)
        else:
            self._transform = self._transform @ _rotation_matrix(axis, angle_deg, origin)
        return self

    def scale(self, factors: Sequence[float]) -> "MeshGroup":
        self._transform = self._transform @ _scale_matrix(factors)
        return self

    def mirror(self, axis: Sequence[float]) -> "MeshGroup":
        self._transform = self._transform @ _mirror_matrix(axis)
        return self

    def multmatrix(self, matrix: np.ndarray) -> "MeshGroup":
        mat = np.asarray(matrix, dtype=float)
        if mat.shape != (4, 4):
            raise ValueError("multmatrix requires a 4x4 matrix.")
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


def mesh_group_compatibility_diagnostic(target: object) -> MeshGroupCompatibilityDiagnostic:
    if isinstance(target, MeshGroup):
        return target.compatibility_diagnostic()
    if isinstance(target, Mesh):
        return MeshGroupCompatibilityDiagnostic(target_type=type(target).__name__)
    return MeshGroupCompatibilityDiagnostic(
        target_type=type(target).__name__,
        reason="surface-authored values must use SurfaceComposition instead of MeshGroup",
    )


def _ensure_mesh_input(target: object) -> None:
    if isinstance(target, Mesh):
        return
    raise MeshGroupCompatibilityError(mesh_group_compatibility_diagnostic(target))
