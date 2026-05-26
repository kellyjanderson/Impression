from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from impression.modeling.path3d import Path3D


@dataclass(frozen=True)
class PatchFamilyCapabilityRecord:
    """Declared support phase and operation coverage for one surface patch family."""

    family: str
    support_phase: Literal["available", "planned"]
    operations: tuple[str, ...]
    notes: str = ""


SUPPORTED_SURFACE_PATCH_FAMILIES: tuple[str, ...] = (
    "planar",
    "ruled",
    "revolution",
    "bspline",
    "nurbs",
    "sweep",
    "subdivision",
    "implicit",
)
REQUIRED_V1_PATCH_FAMILIES: tuple[str, ...] = ("planar", "ruled", "revolution")
PATCH_FAMILY_CAPABILITY_MATRIX: dict[str, PatchFamilyCapabilityRecord] = {
    "planar": PatchFamilyCapabilityRecord(
        family="planar",
        support_phase="available",
        operations=("caps", "planar-primitives", "trimmed-faces", "tessellation", ".impress"),
    ),
    "ruled": PatchFamilyCapabilityRecord(
        family="ruled",
        support_phase="available",
        operations=("extrude", "loft", "linear-bridge-surfaces", "tessellation", ".impress"),
    ),
    "revolution": PatchFamilyCapabilityRecord(
        family="revolution",
        support_phase="available",
        operations=("rotate-extrude", "revolved-primitives", "tessellation", ".impress"),
    ),
    "bspline": PatchFamilyCapabilityRecord(
        family="bspline",
        support_phase="planned",
        operations=("surface-record", "evaluation", "tessellation", ".impress"),
    ),
    "nurbs": PatchFamilyCapabilityRecord(
        family="nurbs",
        support_phase="planned",
        operations=("rational-surface-record", "evaluation", "tessellation", ".impress"),
    ),
    "sweep": PatchFamilyCapabilityRecord(
        family="sweep",
        support_phase="planned",
        operations=("sweep-record", "frame-policy", "evaluation", "tessellation", ".impress"),
    ),
    "subdivision": PatchFamilyCapabilityRecord(
        family="subdivision",
        support_phase="planned",
        operations=("control-cage", "crease-payload", "evaluation", "tessellation", ".impress"),
    ),
    "implicit": PatchFamilyCapabilityRecord(
        family="implicit",
        support_phase="planned",
        operations=("field-node-payload", "validation-security", "evaluation", "tessellation", ".impress"),
    ),
}
SURFACE_SPEC_66_RETIREMENT_NOTE = (
    "Surface Spec 66 is superseded by PATCH_FAMILY_CAPABILITY_MATRIX; no patch family is architecturally deferred."
)
PATCH_FAMILY_FEATURE_COVERAGE: dict[str, tuple[str, ...]] = {
    family: record.operations for family, record in PATCH_FAMILY_CAPABILITY_MATRIX.items()
}


def _as_vec2(value: Sequence[float], *, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=float).reshape(2)
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} must contain only finite values.")
    return vec


def _as_vec3(value: Sequence[float], *, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} must contain only finite values.")
    return vec


def _as_points3(value: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    pts = np.asarray(value, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two 3D points.")
    if not np.all(np.isfinite(pts)):
        raise ValueError(f"{name} must contain only finite values.")
    return pts


def _as_points2(value: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    pts = np.asarray(value, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two 2D points.")
    if not np.all(np.isfinite(pts)):
        raise ValueError(f"{name} must contain only finite values.")
    return pts


def _as_control_net3(value: Sequence[Sequence[Sequence[float]]] | np.ndarray, *, name: str) -> np.ndarray:
    net = np.asarray(value, dtype=float)
    if net.ndim != 3 or net.shape[2] != 3:
        raise ValueError(f"{name} must be a 3D control net with shape (u_count, v_count, 3).")
    if net.shape[0] < 2 or net.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two control points along each parameter direction.")
    if not np.all(np.isfinite(net)):
        raise ValueError(f"{name} must contain only finite values.")
    return net


def _as_control_points3(value: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    pts = np.asarray(value, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 3:
        raise ValueError(f"{name} must contain at least three 3D points.")
    if not np.all(np.isfinite(pts)):
        raise ValueError(f"{name} must contain only finite values.")
    return pts


def _normalize_subdivision_faces(
    faces: Sequence[Sequence[int]],
    *,
    control_point_count: int,
) -> tuple[tuple[int, ...], ...]:
    normalized: list[tuple[int, ...]] = []
    for face_index, face in enumerate(faces):
        indices = tuple(int(index) for index in face)
        if len(indices) < 3:
            raise ValueError("SubdivisionSurfacePatch faces must contain at least three vertices.")
        if len(set(indices)) != len(indices):
            raise ValueError("SubdivisionSurfacePatch faces may not repeat a vertex.")
        for index in indices:
            if index < 0 or index >= control_point_count:
                raise ValueError(f"SubdivisionSurfacePatch face {face_index} references a control point outside the cage.")
        normalized.append(indices)
    if not normalized:
        raise ValueError("SubdivisionSurfacePatch requires at least one face.")
    return tuple(normalized)


def _subdivision_face_edges(faces: Sequence[Sequence[int]]) -> frozenset[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        for start, end in zip(face, (*face[1:], face[0])):
            edge = tuple(sorted((int(start), int(end))))
            edges.add((edge[0], edge[1]))
    return frozenset(edges)


def _subdivision_edge_faces(faces: Sequence[Sequence[int]]) -> dict[tuple[int, int], list[int]]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for start, end in zip(face, (*face[1:], face[0])):
            edge = tuple(sorted((int(start), int(end))))
            edge_faces.setdefault((edge[0], edge[1]), []).append(face_index)
    return edge_faces


def _normalize_degree(value: int, *, name: str) -> int:
    degree = int(value)
    if degree < 1:
        raise ValueError(f"{name} must be >= 1.")
    return degree


def _normalize_knot_vector(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    knots = tuple(float(value) for value in values)
    if len(knots) < 4:
        raise ValueError(f"{name} is too short.")
    if not all(np.isfinite(value) for value in knots):
        raise ValueError(f"{name} values must be finite.")
    if any(b < a for a, b in zip(knots, knots[1:])):
        raise ValueError(f"{name} must be nondecreasing.")
    return knots


def _validate_bspline_axis(*, control_point_count: int, degree: int, knots: tuple[float, ...], name: str) -> tuple[float, float]:
    if control_point_count <= degree:
        raise ValueError(f"{name} control point count must be greater than degree.")
    expected = control_point_count + degree + 1
    if len(knots) != expected:
        raise ValueError(f"{name} knot vector length must equal control_point_count + degree + 1.")
    start = float(knots[degree])
    end = float(knots[-degree - 1])
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError(f"{name} parameter range must be finite and increasing.")
    return start, end


def _find_bspline_span(*, degree: int, knots: tuple[float, ...], control_point_count: int, parameter: float) -> int:
    n = control_point_count - 1
    if parameter >= knots[n + 1]:
        return n
    if parameter <= knots[degree]:
        return degree
    low = degree
    high = n + 1
    mid = (low + high) // 2
    while parameter < knots[mid] or parameter >= knots[mid + 1]:
        if parameter < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def _bspline_basis_functions(span: int, parameter: float, degree: int, knots: tuple[float, ...]) -> np.ndarray:
    left = np.zeros(degree + 1, dtype=float)
    right = np.zeros(degree + 1, dtype=float)
    basis = np.zeros(degree + 1, dtype=float)
    basis[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = parameter - knots[span + 1 - j]
        right[j] = knots[span + j] - parameter
        saved = 0.0
        for r in range(j):
            denominator = right[r + 1] + left[j - r]
            term = 0.0 if denominator == 0.0 else basis[r] / denominator
            basis[r] = saved + right[r + 1] * term
            saved = left[j - r] * term
        basis[j] = saved
    return basis


def _evaluate_bspline_surface_net(
    *,
    control_net: np.ndarray,
    degree_u: int,
    degree_v: int,
    knots_u: tuple[float, ...],
    knots_v: tuple[float, ...],
    u: float,
    v: float,
) -> np.ndarray:
    span_u = _find_bspline_span(degree=degree_u, knots=knots_u, control_point_count=control_net.shape[0], parameter=u)
    span_v = _find_bspline_span(degree=degree_v, knots=knots_v, control_point_count=control_net.shape[1], parameter=v)
    basis_u = _bspline_basis_functions(span_u, u, degree_u, knots_u)
    basis_v = _bspline_basis_functions(span_v, v, degree_v, knots_v)
    point = np.zeros(control_net.shape[2], dtype=float)
    for i in range(degree_u + 1):
        u_index = span_u - degree_u + i
        for j in range(degree_v + 1):
            v_index = span_v - degree_v + j
            point += basis_u[i] * basis_v[j] * control_net[u_index, v_index]
    return point


def _bspline_surface_derivative_net(control_net: np.ndarray, degree: int, knots: tuple[float, ...], *, axis: int) -> np.ndarray:
    if axis == 0:
        derived = np.zeros((control_net.shape[0] - 1, control_net.shape[1], 3), dtype=float)
        for i in range(control_net.shape[0] - 1):
            denominator = knots[i + degree + 1] - knots[i + 1]
            if denominator != 0.0:
                derived[i, :, :] = (degree / denominator) * (control_net[i + 1, :, :] - control_net[i, :, :])
        return derived
    derived = np.zeros((control_net.shape[0], control_net.shape[1] - 1, 3), dtype=float)
    for j in range(control_net.shape[1] - 1):
        denominator = knots[j + degree + 1] - knots[j + 1]
        if denominator != 0.0:
            derived[:, j, :] = (degree / denominator) * (control_net[:, j + 1, :] - control_net[:, j, :])
    return derived


def _as_matrix4(value: Sequence[Sequence[float]] | np.ndarray, *, name: str = "matrix") -> np.ndarray:
    mat = np.asarray(value, dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(mat)):
        raise ValueError(f"{name} must contain only finite values.")
    return mat


def _normalize_axis(value: Sequence[float], *, name: str) -> np.ndarray:
    vec = _as_vec3(value, name=name)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return vec / norm


def _normalize_loop_points_2d(points: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        return pts
    if not np.all(np.isfinite(pts)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def _signed_area_2d(points: np.ndarray) -> float:
    pts = _normalize_loop_points_2d(points, name="points")
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_loop_orientation(points: np.ndarray, *, clockwise: bool) -> np.ndarray:
    pts = _normalize_loop_points_2d(points, name="points")
    if pts.shape[0] < 3:
        return pts
    is_clockwise = _signed_area_2d(pts) < 0.0
    return pts[::-1].copy() if is_clockwise != clockwise else pts.copy()


def _normalize_metadata(metadata: dict[str, object] | None) -> dict[str, object]:
    if metadata is None:
        return {}
    return dict(metadata)


def _split_metadata(metadata: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    metadata = _normalize_metadata(metadata)
    if "kernel" in metadata or "consumer" in metadata:
        extra_keys = set(metadata) - {"kernel", "consumer"}
        if extra_keys:
            raise ValueError("Namespaced metadata may only use 'kernel' and 'consumer' top-level keys.")
        kernel = metadata.get("kernel", {})
        consumer = metadata.get("consumer", {})
        if not isinstance(kernel, dict) or not isinstance(consumer, dict):
            raise ValueError("'kernel' and 'consumer' metadata values must be dictionaries.")
        return dict(kernel), dict(consumer)
    return dict(metadata), {}


def _is_identity_matrix(matrix: np.ndarray) -> bool:
    return bool(np.allclose(matrix, np.eye(4), atol=1e-12))


def _compose_transform(current: np.ndarray, applied: np.ndarray) -> np.ndarray:
    return applied @ current


def _transform_point(matrix: np.ndarray, point: Sequence[float] | np.ndarray) -> np.ndarray:
    vec = np.asarray(point, dtype=float).reshape(3)
    homogeneous = np.array([vec[0], vec[1], vec[2], 1.0], dtype=float)
    return (matrix @ homogeneous)[:3]


def _transform_vector(matrix: np.ndarray, vector: Sequence[float] | np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=float).reshape(3)
    return matrix[:3, :3] @ vec


def _polyline_point_at(points: np.ndarray, t: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 2:
        return pts[0] + ((pts[1] - pts[0]) * float(t))
    deltas = np.diff(pts, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    total = float(lengths.sum())
    if total == 0.0:
        return pts[0].copy()
    target = float(np.clip(t, 0.0, 1.0)) * total
    consumed = 0.0
    for index, seg_length in enumerate(lengths):
        if seg_length == 0.0:
            continue
        next_consumed = consumed + float(seg_length)
        if target <= next_consumed or index == len(lengths) - 1:
            local = (target - consumed) / float(seg_length)
            return pts[index] + ((pts[index + 1] - pts[index]) * local)
        consumed = next_consumed
    return pts[-1].copy()


def _polyline_point_at_2d(points: np.ndarray, t: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 2:
        return pts[0] + ((pts[1] - pts[0]) * float(t))
    deltas = np.diff(pts, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    total = float(lengths.sum())
    if total == 0.0:
        return pts[0].copy()
    target = float(np.clip(t, 0.0, 1.0)) * total
    consumed = 0.0
    for index, seg_length in enumerate(lengths):
        if seg_length == 0.0:
            continue
        next_consumed = consumed + float(seg_length)
        if target <= next_consumed or index == len(lengths) - 1:
            local = (target - consumed) / float(seg_length)
            return pts[index] + ((pts[index + 1] - pts[index]) * local)
        consumed = next_consumed
    return pts[-1].copy()


def _frame_for_tangent(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w_axis = _normalize_axis(tangent, name="path_tangent")
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(w_axis, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    u_axis = _normalize_axis(np.cross(reference, w_axis), name="sweep_u_axis")
    v_axis = _normalize_axis(np.cross(w_axis, u_axis), name="sweep_v_axis")
    return u_axis, v_axis, w_axis


def _transform_bounds(bounds: tuple[float, float, float, float, float, float], matrix: np.ndarray) -> tuple[float, float, float, float, float, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
        ],
        dtype=float,
    )
    transformed = np.array([_transform_point(matrix, corner) for corner in corners], dtype=float)
    mins = transformed.min(axis=0)
    maxs = transformed.max(axis=0)
    return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))


def _canonicalize(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _canonicalize(value.tolist())
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_canonicalize(item) for item in value)
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        scalar = float(value)
        if np.isfinite(scalar):
            return scalar
        raise ValueError("Canonical payloads must contain only finite numeric values.")
    return repr(value)


def _stable_hash(payload: object) -> str:
    canonical = _canonicalize(payload)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ParameterDomain:
    """A rectangular parameter-space domain for a surface patch."""

    u_range: tuple[float, float] = (0.0, 1.0)
    v_range: tuple[float, float] = (0.0, 1.0)
    normalized: bool = True

    def __post_init__(self) -> None:
        u0, u1 = (float(self.u_range[0]), float(self.u_range[1]))
        v0, v1 = (float(self.v_range[0]), float(self.v_range[1]))
        if not np.isfinite([u0, u1, v0, v1]).all():
            raise ValueError("ParameterDomain ranges must be finite.")
        if u1 <= u0:
            raise ValueError("u_range must have positive span.")
        if v1 <= v0:
            raise ValueError("v_range must have positive span.")
        object.__setattr__(self, "u_range", (u0, u1))
        object.__setattr__(self, "v_range", (v0, v1))

    def contains(self, u: float, v: float, *, epsilon: float = 1e-9) -> bool:
        u0, u1 = self.u_range
        v0, v1 = self.v_range
        return (u0 - epsilon) <= u <= (u1 + epsilon) and (v0 - epsilon) <= v <= (v1 + epsilon)

    @property
    def u_span(self) -> float:
        return float(self.u_range[1] - self.u_range[0])

    @property
    def v_span(self) -> float:
        return float(self.v_range[1] - self.v_range[0])

    def canonical_payload(self) -> dict[str, object]:
        return {
            "u_range": self.u_range,
            "v_range": self.v_range,
            "normalized": self.normalized,
        }


@dataclass(frozen=True)
class TrimLoop:
    """A closed trim loop expressed in patch-local parameter space."""

    points_uv: np.ndarray
    category: Literal["outer", "inner"] = "outer"

    def __post_init__(self) -> None:
        points = _normalize_loop_points_2d(self.points_uv, name="points_uv")
        category = str(self.category).lower()
        if category not in {"outer", "inner"}:
            raise ValueError("TrimLoop.category must be 'outer' or 'inner'.")
        if points.shape[0] < 3:
            raise ValueError("TrimLoop requires at least three distinct points.")
        object.__setattr__(self, "points_uv", points)
        object.__setattr__(self, "category", category)

    @property
    def area(self) -> float:
        return _signed_area_2d(self.points_uv)

    @property
    def is_clockwise(self) -> bool:
        return self.area < 0.0

    def normalized(self) -> "TrimLoop":
        clockwise = self.category == "inner"
        return TrimLoop(_ensure_loop_orientation(self.points_uv, clockwise=clockwise), category=self.category)

    def validate_against_domain(self, domain: ParameterDomain) -> None:
        for point in self.points_uv:
            if not domain.contains(float(point[0]), float(point[1])):
                raise ValueError("TrimLoop contains points outside the patch domain.")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "category": self.category,
            "points_uv": self.points_uv,
        }


@dataclass(frozen=True)
class SurfaceBoundaryRef:
    """A stable reference to one named boundary of one patch in a shell."""

    patch_index: int
    boundary_id: str

    def __post_init__(self) -> None:
        patch_index = int(self.patch_index)
        boundary_id = str(self.boundary_id).strip()
        if patch_index < 0:
            raise ValueError("SurfaceBoundaryRef.patch_index must be >= 0.")
        if not boundary_id:
            raise ValueError("SurfaceBoundaryRef.boundary_id must be non-empty.")
        object.__setattr__(self, "patch_index", patch_index)
        object.__setattr__(self, "boundary_id", boundary_id)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "patch_index": self.patch_index,
            "boundary_id": self.boundary_id,
        }


@dataclass(frozen=True)
class SurfaceAdjacencyRecord:
    """One directional adjacency relationship originating from one boundary."""

    source: SurfaceBoundaryRef
    target: SurfaceBoundaryRef | None
    seam_id: str | None = None
    continuity: str = "C0"

    def __post_init__(self) -> None:
        continuity = str(self.continuity).strip()
        seam_id = None if self.seam_id is None else str(self.seam_id).strip()
        if not continuity:
            raise ValueError("SurfaceAdjacencyRecord.continuity must be non-empty.")
        if seam_id == "":
            raise ValueError("SurfaceAdjacencyRecord.seam_id must be non-empty when provided.")
        object.__setattr__(self, "continuity", continuity)
        object.__setattr__(self, "seam_id", seam_id)

    @property
    def is_open(self) -> bool:
        return self.target is None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "source": self.source.canonical_payload(),
            "target": None if self.target is None else self.target.canonical_payload(),
            "seam_id": self.seam_id,
            "continuity": self.continuity,
        }


@dataclass(frozen=True)
class SurfaceSeam:
    """A first-class seam object connecting one or two patch boundaries."""

    seam_id: str
    boundaries: tuple[SurfaceBoundaryRef, ...]
    continuity: str = "C0"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seam_id = str(self.seam_id).strip()
        if not seam_id:
            raise ValueError("SurfaceSeam.seam_id must be non-empty.")
        boundaries = tuple(self.boundaries)
        continuity = str(self.continuity).strip()
        if len(boundaries) not in {1, 2}:
            raise ValueError("SurfaceSeam boundaries must contain one open boundary or two shared boundaries.")
        if not continuity:
            raise ValueError("SurfaceSeam.continuity must be non-empty.")
        if len({(boundary.patch_index, boundary.boundary_id) for boundary in boundaries}) != len(boundaries):
            raise ValueError("SurfaceSeam boundaries must be unique.")
        object.__setattr__(self, "seam_id", seam_id)
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "continuity", continuity)
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    @property
    def is_open(self) -> bool:
        return len(self.boundaries) == 1

    def canonical_payload(self) -> dict[str, object]:
        return {
            "seam_id": self.seam_id,
            "boundaries": [boundary.canonical_payload() for boundary in self.boundaries],
            "continuity": self.continuity,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SurfacePatch(ABC):
    """Abstract base class for surface-native patch geometry."""

    family: str
    domain: ParameterDomain = field(default_factory=ParameterDomain)
    capability_flags: frozenset[str] = field(default_factory=frozenset)
    trim_loops: tuple[TrimLoop, ...] = field(default_factory=tuple)
    transform_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family = self.family.strip()
        if not family:
            raise ValueError("SurfacePatch.family must be non-empty.")
        trim_loops = tuple(self.trim_loops)
        outer_count = sum(1 for loop in trim_loops if loop.category == "outer")
        if outer_count > 1:
            raise ValueError("SurfacePatch may not have more than one outer trim loop.")
        transform_matrix = _as_matrix4(self.transform_matrix, name="transform_matrix")
        metadata = _normalize_metadata(self.metadata)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "capability_flags", frozenset(str(flag) for flag in self.capability_flags))
        object.__setattr__(self, "trim_loops", trim_loops)
        object.__setattr__(self, "transform_matrix", transform_matrix)
        object.__setattr__(self, "metadata", metadata)
        for trim_loop in trim_loops:
            trim_loop.validate_against_domain(self.domain)

    @abstractmethod
    def point_at(self, u: float, v: float) -> np.ndarray:
        """Return a 3D point evaluated at parameter coordinates ``(u, v)``."""

    @abstractmethod
    def derivatives_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the first partial derivatives with respect to ``u`` and ``v``."""

    @abstractmethod
    def geometry_payload(self) -> dict[str, object]:
        """Return canonical geometry fields for identity and caching."""

    def validate_parameters(self, u: float, v: float) -> None:
        if not self.domain.contains(u, v):
            raise ValueError(f"Parameters {(u, v)} are outside patch domain {self.domain}.")

    @property
    def outer_trim(self) -> TrimLoop | None:
        for trim_loop in self.trim_loops:
            if trim_loop.category == "outer":
                return trim_loop
        return None

    @property
    def inner_trims(self) -> tuple[TrimLoop, ...]:
        return tuple(trim_loop for trim_loop in self.trim_loops if trim_loop.category == "inner")

    @property
    def stable_identity(self) -> str:
        return _stable_hash(self.canonical_payload())

    @property
    def cache_key(self) -> str:
        return self.stable_identity

    def canonical_payload(self) -> dict[str, object]:
        return {
            "kind": type(self).__name__,
            "family": self.family,
            "domain": self.domain.canonical_payload(),
            "capability_flags": sorted(self.capability_flags),
            "trim_loops": [trim_loop.normalized().canonical_payload() for trim_loop in self.trim_loops],
            "transform_matrix": self.transform_matrix,
            "metadata": self.metadata,
            "geometry": self.geometry_payload(),
        }

    def kernel_metadata(self) -> dict[str, object]:
        kernel, _consumer = _split_metadata(self.metadata)
        return kernel

    def consumer_metadata(self) -> dict[str, object]:
        _kernel, consumer = _split_metadata(self.metadata)
        return consumer

    def merged_kernel_metadata(self, inherited: dict[str, object] | None = None) -> dict[str, object]:
        merged = {} if inherited is None else dict(inherited)
        merged.update(self.kernel_metadata())
        return merged

    def merged_consumer_metadata(self, inherited: dict[str, object] | None = None) -> dict[str, object]:
        merged = {} if inherited is None else dict(inherited)
        merged.update(self.consumer_metadata())
        return merged

    def with_transform(self, matrix: Sequence[Sequence[float]] | np.ndarray) -> "SurfacePatch":
        applied = _as_matrix4(matrix)
        return replace(self, transform_matrix=_compose_transform(self.transform_matrix, applied))

    def normal_at(self, u: float, v: float) -> np.ndarray:
        du, dv = self.derivatives_at(u, v)
        normal = np.cross(du, dv)
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            raise ValueError("Patch derivatives are degenerate; normal is undefined.")
        return normal / norm

    def frame_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        du, dv = self.derivatives_at(u, v)
        u_axis = _normalize_axis(du, name="du")
        normal = self.normal_at(u, v)
        v_axis = np.cross(normal, u_axis)
        return u_axis, _normalize_axis(v_axis, name="dv_frame"), normal

    def sample_grid(self, u_count: int, v_count: int) -> np.ndarray:
        if u_count < 2 or v_count < 2:
            raise ValueError("sample_grid requires at least 2 samples on each axis.")
        us = np.linspace(self.domain.u_range[0], self.domain.u_range[1], u_count)
        vs = np.linspace(self.domain.v_range[0], self.domain.v_range[1], v_count)
        samples = np.zeros((u_count, v_count, 3), dtype=float)
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                samples[i, j] = self.point_at(float(u), float(v))
        return samples

    def bounds_estimate(self, *, u_count: int = 3, v_count: int = 3) -> tuple[float, float, float, float, float, float]:
        samples = self.sample_grid(u_count, v_count).reshape(-1, 3)
        mins = samples.min(axis=0)
        maxs = samples.max(axis=0)
        return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))


@dataclass(frozen=True)
class PlanarSurfacePatch(SurfacePatch):
    """A simple planar patch parameterized by two in-plane basis vectors."""

    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    u_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    v_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=float))

    def __post_init__(self) -> None:
        super().__post_init__()
        origin = _as_vec3(self.origin, name="origin")
        u_axis = _as_vec3(self.u_axis, name="u_axis")
        v_axis = _as_vec3(self.v_axis, name="v_axis")
        cross = np.cross(u_axis, v_axis)
        if float(np.linalg.norm(cross)) == 0.0:
            raise ValueError("PlanarSurfacePatch axes must be linearly independent.")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "u_axis", u_axis)
        object.__setattr__(self, "v_axis", v_axis)

    def geometry_payload(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "u_axis": self.u_axis,
            "v_axis": self.v_axis,
        }

    def point_at(self, u: float, v: float) -> np.ndarray:
        self.validate_parameters(u, v)
        local = self.origin + (self.u_axis * float(u)) + (self.v_axis * float(v))
        return _transform_point(self.transform_matrix, local)

    def derivatives_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        self.validate_parameters(u, v)
        return (
            _transform_vector(self.transform_matrix, self.u_axis),
            _transform_vector(self.transform_matrix, self.v_axis),
        )


@dataclass(frozen=True)
class RuledSurfacePatch(SurfacePatch):
    """A ruled patch interpolating between two 3D guide curves."""

    start_curve: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float))
    end_curve: np.ndarray = field(default_factory=lambda: np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=float))

    def __post_init__(self) -> None:
        super().__post_init__()
        start_curve = _as_points3(self.start_curve, name="start_curve")
        end_curve = _as_points3(self.end_curve, name="end_curve")
        if start_curve.shape != end_curve.shape:
            raise ValueError("RuledSurfacePatch start_curve and end_curve must have identical shape.")
        object.__setattr__(self, "start_curve", start_curve)
        object.__setattr__(self, "end_curve", end_curve)

    def geometry_payload(self) -> dict[str, object]:
        return {
            "start_curve": self.start_curve,
            "end_curve": self.end_curve,
        }

    def _normalized_parameters(self, u: float, v: float) -> tuple[float, float]:
        self.validate_parameters(u, v)
        u0, u1 = self.domain.u_range
        v0, v1 = self.domain.v_range
        u_norm = 0.0 if u1 == u0 else (float(u) - u0) / (u1 - u0)
        v_norm = 0.0 if v1 == v0 else (float(v) - v0) / (v1 - v0)
        return float(np.clip(u_norm, 0.0, 1.0)), float(np.clip(v_norm, 0.0, 1.0))

    def _edge_points(self, v_norm: float) -> tuple[np.ndarray, np.ndarray]:
        return (
            _polyline_point_at(self.start_curve, v_norm),
            _polyline_point_at(self.end_curve, v_norm),
        )

    def point_at(self, u: float, v: float) -> np.ndarray:
        u_norm, v_norm = self._normalized_parameters(u, v)
        start_point, end_point = self._edge_points(v_norm)
        local = start_point + ((end_point - start_point) * u_norm)
        return _transform_point(self.transform_matrix, local)

    def derivatives_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        u_norm, v_norm = self._normalized_parameters(u, v)
        start_point, end_point = self._edge_points(v_norm)
        du = (end_point - start_point) / self.domain.u_span

        step = min(1e-4, 0.25)
        v_prev = max(0.0, v_norm - step)
        v_next = min(1.0, v_norm + step)
        if np.isclose(v_prev, v_next):
            dv_local = np.zeros(3, dtype=float)
        else:
            start_prev, end_prev = self._edge_points(v_prev)
            start_next, end_next = self._edge_points(v_next)
            point_prev = start_prev + ((end_prev - start_prev) * u_norm)
            point_next = start_next + ((end_next - start_next) * u_norm)
            dv_local = (point_next - point_prev) / ((v_next - v_prev) * self.domain.v_span)
        return (
            _transform_vector(self.transform_matrix, du),
            _transform_vector(self.transform_matrix, dv_local),
        )


def _rotate_point_around_axis(point: np.ndarray, axis_origin: np.ndarray, axis_direction: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize_axis(axis_direction, name="axis_direction")
    vec = np.asarray(point, dtype=float).reshape(3) - axis_origin
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    cross = np.cross(axis, vec)
    dot = float(np.dot(vec, axis))
    rotated = (vec * cos_a) + (cross * sin_a) + (axis * dot * (1.0 - cos_a))
    return axis_origin + rotated


@dataclass(frozen=True)
class RevolutionSurfacePatch(SurfacePatch):
    """A surface patch formed by revolving a profile curve around an axis."""

    profile_curve: np.ndarray = field(default_factory=lambda: np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=float))
    axis_origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    axis_direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))
    start_angle_deg: float = 0.0
    sweep_angle_deg: float = 360.0

    def __post_init__(self) -> None:
        super().__post_init__()
        profile_curve = _as_points3(self.profile_curve, name="profile_curve")
        axis_origin = _as_vec3(self.axis_origin, name="axis_origin")
        axis_direction = _normalize_axis(self.axis_direction, name="axis_direction")
        start_angle_deg = float(self.start_angle_deg)
        sweep_angle_deg = float(self.sweep_angle_deg)
        if not np.isfinite(start_angle_deg) or not np.isfinite(sweep_angle_deg):
            raise ValueError("RevolutionSurfacePatch angles must be finite.")
        if sweep_angle_deg == 0.0:
            raise ValueError("RevolutionSurfacePatch sweep_angle_deg must be non-zero.")
        object.__setattr__(self, "profile_curve", profile_curve)
        object.__setattr__(self, "axis_origin", axis_origin)
        object.__setattr__(self, "axis_direction", axis_direction)
        object.__setattr__(self, "start_angle_deg", start_angle_deg)
        object.__setattr__(self, "sweep_angle_deg", sweep_angle_deg)

    def geometry_payload(self) -> dict[str, object]:
        return {
            "profile_curve": self.profile_curve,
            "axis_origin": self.axis_origin,
            "axis_direction": self.axis_direction,
            "start_angle_deg": self.start_angle_deg,
            "sweep_angle_deg": self.sweep_angle_deg,
        }

    def _normalized_parameters(self, u: float, v: float) -> tuple[float, float]:
        self.validate_parameters(u, v)
        u0, u1 = self.domain.u_range
        v0, v1 = self.domain.v_range
        u_norm = (float(u) - u0) / (u1 - u0)
        v_norm = (float(v) - v0) / (v1 - v0)
        return float(np.clip(u_norm, 0.0, 1.0)), float(np.clip(v_norm, 0.0, 1.0))

    def _angle_rad_at(self, u_norm: float) -> float:
        angle_deg = self.start_angle_deg + (self.sweep_angle_deg * u_norm)
        return float(np.deg2rad(angle_deg))

    def _profile_point(self, v_norm: float) -> np.ndarray:
        return _polyline_point_at(self.profile_curve, v_norm)

    def point_at(self, u: float, v: float) -> np.ndarray:
        u_norm, v_norm = self._normalized_parameters(u, v)
        local = _rotate_point_around_axis(
            self._profile_point(v_norm),
            self.axis_origin,
            self.axis_direction,
            self._angle_rad_at(u_norm),
        )
        return _transform_point(self.transform_matrix, local)

    def derivatives_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        u_norm, v_norm = self._normalized_parameters(u, v)
        angle_rad = self._angle_rad_at(u_norm)
        point = _rotate_point_around_axis(self._profile_point(v_norm), self.axis_origin, self.axis_direction, angle_rad)
        radial = point - self.axis_origin
        radial -= self.axis_direction * float(np.dot(radial, self.axis_direction))
        angle_rate = float(np.deg2rad(self.sweep_angle_deg)) / self.domain.u_span
        du = np.cross(self.axis_direction, radial) * angle_rate

        step = min(1e-4, 0.25)
        v_prev = max(0.0, v_norm - step)
        v_next = min(1.0, v_norm + step)
        if np.isclose(v_prev, v_next):
            dv_local = np.zeros(3, dtype=float)
        else:
            prev_point = _rotate_point_around_axis(self._profile_point(v_prev), self.axis_origin, self.axis_direction, angle_rad)
            next_point = _rotate_point_around_axis(self._profile_point(v_next), self.axis_origin, self.axis_direction, angle_rad)
            dv_local = (next_point - prev_point) / ((v_next - v_prev) * self.domain.v_span)
        return (
            _transform_vector(self.transform_matrix, du),
            _transform_vector(self.transform_matrix, dv_local),
        )

    def bounds_estimate(self, *, u_count: int = 3, v_count: int = 3) -> tuple[float, float, float, float, float, float]:
        return super().bounds_estimate(u_count=max(u_count, 17), v_count=max(v_count, 17))


@dataclass(frozen=True)
class BSplineSurfacePatch(SurfacePatch):
    """A non-rational tensor-product B-spline surface patch record."""

    degree_u: int = 1
    degree_v: int = 1
    knots_u: tuple[float, ...] = (0.0, 0.0, 1.0, 1.0)
    knots_v: tuple[float, ...] = (0.0, 0.0, 1.0, 1.0)
    control_net: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            ],
            dtype=float,
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.family != "bspline":
            raise ValueError("BSplineSurfacePatch.family must be 'bspline'.")
        degree_u = _normalize_degree(self.degree_u, name="degree_u")
        degree_v = _normalize_degree(self.degree_v, name="degree_v")
        knots_u = _normalize_knot_vector(self.knots_u, name="knots_u")
        knots_v = _normalize_knot_vector(self.knots_v, name="knots_v")
        control_net = _as_control_net3(self.control_net, name="control_net")
        u_range = _validate_bspline_axis(
            control_point_count=control_net.shape[0],
            degree=degree_u,
            knots=knots_u,
            name="u",
        )
        v_range = _validate_bspline_axis(
            control_point_count=control_net.shape[1],
            degree=degree_v,
            knots=knots_v,
            name="v",
        )
        if not np.allclose(self.domain.u_range, u_range) or not np.allclose(self.domain.v_range, v_range):
            raise ValueError("BSplineSurfacePatch domain must match knot parameter ranges.")
        object.__setattr__(self, "degree_u", degree_u)
        object.__setattr__(self, "degree_v", degree_v)
        object.__setattr__(self, "knots_u", knots_u)
        object.__setattr__(self, "knots_v", knots_v)
        object.__setattr__(self, "control_net", control_net)

    def geometry_payload(self) -> dict[str, object]:
        return {
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "knots_u": self.knots_u,
            "knots_v": self.knots_v,
            "control_net": self.control_net,
        }

    def point_at(self, u: float, v: float) -> np.ndarray:
        self.validate_parameters(u, v)
        local = _evaluate_bspline_surface_net(
            control_net=self.control_net,
            degree_u=self.degree_u,
            degree_v=self.degree_v,
            knots_u=self.knots_u,
            knots_v=self.knots_v,
            u=float(u),
            v=float(v),
        )
        return _transform_point(self.transform_matrix, local)

    def derivatives_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        self.validate_parameters(u, v)
        du_net = _bspline_surface_derivative_net(self.control_net, self.degree_u, self.knots_u, axis=0)
        dv_net = _bspline_surface_derivative_net(self.control_net, self.degree_v, self.knots_v, axis=1)
        du = _evaluate_bspline_surface_net(
            control_net=du_net,
            degree_u=self.degree_u - 1,
            degree_v=self.degree_v,
            knots_u=self.knots_u[1:-1],
            knots_v=self.knots_v,
            u=float(u),
            v=float(v),
        )
        dv = _evaluate_bspline_surface_net(
            control_net=dv_net,
            degree_u=self.degree_u,
            degree_v=self.degree_v - 1,
            knots_u=self.knots_u,
            knots_v=self.knots_v[1:-1],
            u=float(u),
            v=float(v),
        )
        return (_transform_vector(self.transform_matrix, du), _transform_vector(self.transform_matrix, dv))


@dataclass(frozen=True)
class NURBSSurfacePatch(SurfacePatch):
    """A rational tensor-product NURBS surface patch."""

    degree_u: int = 1
    degree_v: int = 1
    knots_u: tuple[float, ...] = (0.0, 0.0, 1.0, 1.0)
    knots_v: tuple[float, ...] = (0.0, 0.0, 1.0, 1.0)
    control_net: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            ],
            dtype=float,
        )
    )
    weights: np.ndarray = field(default_factory=lambda: np.ones((2, 2), dtype=float))

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.family != "nurbs":
            raise ValueError("NURBSSurfacePatch.family must be 'nurbs'.")
        degree_u = _normalize_degree(self.degree_u, name="degree_u")
        degree_v = _normalize_degree(self.degree_v, name="degree_v")
        knots_u = _normalize_knot_vector(self.knots_u, name="knots_u")
        knots_v = _normalize_knot_vector(self.knots_v, name="knots_v")
        control_net = _as_control_net3(self.control_net, name="control_net")
        weights = np.asarray(self.weights, dtype=float)
        if weights.shape != control_net.shape[:2]:
            raise ValueError("NURBSSurfacePatch weights must match the control net parameter shape.")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("NURBSSurfacePatch weights must be finite and positive.")
        u_range = _validate_bspline_axis(
            control_point_count=control_net.shape[0],
            degree=degree_u,
            knots=knots_u,
            name="u",
        )
        v_range = _validate_bspline_axis(
            control_point_count=control_net.shape[1],
            degree=degree_v,
            knots=knots_v,
            name="v",
        )
        if not np.allclose(self.domain.u_range, u_range) or not np.allclose(self.domain.v_range, v_range):
            raise ValueError("NURBSSurfacePatch domain must match knot parameter ranges.")
        object.__setattr__(self, "degree_u", degree_u)
        object.__setattr__(self, "degree_v", degree_v)
        object.__setattr__(self, "knots_u", knots_u)
        object.__setattr__(self, "knots_v", knots_v)
        object.__setattr__(self, "control_net", control_net)
        object.__setattr__(self, "weights", weights)

    def geometry_payload(self) -> dict[str, object]:
        return {
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "knots_u": self.knots_u,
            "knots_v": self.knots_v,
            "control_net": self.control_net,
            "weights": self.weights,
        }

    @property
    def weighted_control_net(self) -> np.ndarray:
        return self.control_net * self.weights[:, :, np.newaxis]

    def _rational_components(self, u: float, v: float) -> tuple[np.ndarray, float]:
        numerator = _evaluate_bspline_surface_net(
            control_net=self.weighted_control_net,
            degree_u=self.degree_u,
            degree_v=self.degree_v,
            knots_u=self.knots_u,
            knots_v=self.knots_v,
            u=float(u),
            v=float(v),
        )
        denominator = float(
            _evaluate_bspline_surface_net(
                control_net=self.weights[:, :, np.newaxis],
                degree_u=self.degree_u,
                degree_v=self.degree_v,
                knots_u=self.knots_u,
                knots_v=self.knots_v,
                u=float(u),
                v=float(v),
            )[0]
        )
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("NURBSSurfacePatch evaluated weight must be positive and finite.")
        return numerator, denominator

    def point_at(self, u: float, v: float) -> np.ndarray:
        self.validate_parameters(u, v)
        numerator, denominator = self._rational_components(float(u), float(v))
        return _transform_point(self.transform_matrix, numerator / denominator)

    def derivatives_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        self.validate_parameters(u, v)
        u_value = float(u)
        v_value = float(v)
        numerator, denominator = self._rational_components(u_value, v_value)
        weighted_net = self.weighted_control_net
        weight_net = self.weights[:, :, np.newaxis]
        du_numerator = _evaluate_bspline_surface_net(
            control_net=_bspline_surface_derivative_net(weighted_net, self.degree_u, self.knots_u, axis=0),
            degree_u=self.degree_u - 1,
            degree_v=self.degree_v,
            knots_u=self.knots_u[1:-1],
            knots_v=self.knots_v,
            u=u_value,
            v=v_value,
        )
        dv_numerator = _evaluate_bspline_surface_net(
            control_net=_bspline_surface_derivative_net(weighted_net, self.degree_v, self.knots_v, axis=1),
            degree_u=self.degree_u,
            degree_v=self.degree_v - 1,
            knots_u=self.knots_u,
            knots_v=self.knots_v[1:-1],
            u=u_value,
            v=v_value,
        )
        du_weight = float(
            _evaluate_bspline_surface_net(
                control_net=_bspline_surface_derivative_net(weight_net, self.degree_u, self.knots_u, axis=0),
                degree_u=self.degree_u - 1,
                degree_v=self.degree_v,
                knots_u=self.knots_u[1:-1],
                knots_v=self.knots_v,
                u=u_value,
                v=v_value,
            )[0]
        )
        dv_weight = float(
            _evaluate_bspline_surface_net(
                control_net=_bspline_surface_derivative_net(weight_net, self.degree_v, self.knots_v, axis=1),
                degree_u=self.degree_u,
                degree_v=self.degree_v - 1,
                knots_u=self.knots_u,
                knots_v=self.knots_v[1:-1],
                u=u_value,
                v=v_value,
            )[0]
        )
        du = ((du_numerator * denominator) - (numerator * du_weight)) / (denominator * denominator)
        dv = ((dv_numerator * denominator) - (numerator * dv_weight)) / (denominator * denominator)
        return (_transform_vector(self.transform_matrix, du), _transform_vector(self.transform_matrix, dv))


FrameTransportPolicy = Literal["parallel_transport", "frenet", "fixed"]


@dataclass(frozen=True)
class SweepSurfacePatch(SurfacePatch):
    """A sweep surface payload over a 2D profile and a 3D path."""

    profile_points_uv: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float))
    path: Path3D = field(default_factory=lambda: Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]))
    frame_policy: FrameTransportPolicy = "parallel_transport"
    profile_reference: str | None = None
    path_reference: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.family != "sweep":
            raise ValueError("SweepSurfacePatch.family must be 'sweep'.")
        profile_points = _as_points2(self.profile_points_uv, name="profile_points_uv")
        if not isinstance(self.path, Path3D):
            raise ValueError("SweepSurfacePatch.path must be a Path3D.")
        path_points = self.path.sample()
        if path_points.shape[0] < 2:
            raise ValueError("SweepSurfacePatch.path must sample to at least two 3D points.")
        frame_policy = str(self.frame_policy)
        if frame_policy not in {"parallel_transport", "frenet", "fixed"}:
            raise ValueError("SweepSurfacePatch.frame_policy must be parallel_transport, frenet, or fixed.")
        profile_reference = None if self.profile_reference is None else str(self.profile_reference).strip()
        path_reference = None if self.path_reference is None else str(self.path_reference).strip()
        if profile_reference == "":
            raise ValueError("SweepSurfacePatch.profile_reference must be non-empty when provided.")
        if path_reference == "":
            raise ValueError("SweepSurfacePatch.path_reference must be non-empty when provided.")
        object.__setattr__(self, "profile_points_uv", profile_points)
        object.__setattr__(self, "frame_policy", frame_policy)
        object.__setattr__(self, "profile_reference", profile_reference)
        object.__setattr__(self, "path_reference", path_reference)

    def geometry_payload(self) -> dict[str, object]:
        return {
            "profile_points_uv": self.profile_points_uv,
            "path_points": self.path.sample(),
            "frame_policy": self.frame_policy,
            "profile_reference": self.profile_reference,
            "path_reference": self.path_reference,
        }

    def _normalized_parameters(self, u: float, v: float) -> tuple[float, float]:
        self.validate_parameters(u, v)
        u0, u1 = self.domain.u_range
        v0, v1 = self.domain.v_range
        u_norm = (float(u) - u0) / (u1 - u0)
        v_norm = (float(v) - v0) / (v1 - v0)
        return float(np.clip(u_norm, 0.0, 1.0)), float(np.clip(v_norm, 0.0, 1.0))

    def _path_point_and_frame(self, u_norm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        path_points = self.path.sample()
        point = _polyline_point_at(path_points, u_norm)
        previous_point = _polyline_point_at(path_points, max(0.0, u_norm - 1e-5))
        next_point = _polyline_point_at(path_points, min(1.0, u_norm + 1e-5))
        tangent = next_point - previous_point
        if float(np.linalg.norm(tangent)) == 0.0:
            tangent = path_points[-1] - path_points[0]
        u_axis, v_axis, w_axis = _frame_for_tangent(tangent)
        return point, u_axis, v_axis, w_axis

    def point_at(self, u: float, v: float) -> np.ndarray:
        u_norm, v_norm = self._normalized_parameters(u, v)
        path_point, u_axis, v_axis, _w_axis = self._path_point_and_frame(u_norm)
        profile_point = _polyline_point_at_2d(self.profile_points_uv, v_norm)
        local = path_point + (u_axis * profile_point[0]) + (v_axis * profile_point[1])
        return _transform_point(self.transform_matrix, local)

    def derivatives_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        self.validate_parameters(u, v)
        u_step = min(1e-5 * self.domain.u_span, self.domain.u_span * 0.25)
        v_step = min(1e-5 * self.domain.v_span, self.domain.v_span * 0.25)
        u_prev = max(self.domain.u_range[0], float(u) - u_step)
        u_next = min(self.domain.u_range[1], float(u) + u_step)
        v_prev = max(self.domain.v_range[0], float(v) - v_step)
        v_next = min(self.domain.v_range[1], float(v) + v_step)
        if np.isclose(u_prev, u_next):
            du = np.zeros(3, dtype=float)
        else:
            du = (self.point_at(u_next, v) - self.point_at(u_prev, v)) / (u_next - u_prev)
        if np.isclose(v_prev, v_next):
            dv = np.zeros(3, dtype=float)
        else:
            dv = (self.point_at(u, v_next) - self.point_at(u, v_prev)) / (v_next - v_prev)
        return du, dv


@dataclass(frozen=True)
class SubdivisionCrease:
    """Sharpness assigned to one authored control-cage edge."""

    edge: tuple[int, int]
    sharpness: float = 1.0

    def __post_init__(self) -> None:
        edge = tuple(int(index) for index in self.edge)
        if len(edge) != 2:
            raise ValueError("SubdivisionCrease.edge must contain exactly two vertex indices.")
        if edge[0] == edge[1]:
            raise ValueError("SubdivisionCrease.edge must reference two distinct vertices.")
        sharpness = float(self.sharpness)
        if not np.isfinite(sharpness) or sharpness < 0.0:
            raise ValueError("SubdivisionCrease.sharpness must be finite and non-negative.")
        normalized_edge = tuple(sorted(edge))
        object.__setattr__(self, "edge", (normalized_edge[0], normalized_edge[1]))
        object.__setattr__(self, "sharpness", sharpness)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "edge": self.edge,
            "sharpness": self.sharpness,
        }


SubdivisionScheme = Literal["catmull_clark"]


@dataclass(frozen=True)
class SubdivisionRefinementResult:
    """Finite subdivision approximation data produced from an authored control cage."""

    control_points: np.ndarray
    faces: tuple[tuple[int, ...], ...]
    level: int
    scheme: SubdivisionScheme
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        control_points = _as_control_points3(self.control_points, name="control_points")
        faces = _normalize_subdivision_faces(self.faces, control_point_count=control_points.shape[0])
        level = int(self.level)
        if level < 0:
            raise ValueError("SubdivisionRefinementResult.level must be >= 0.")
        scheme = str(self.scheme)
        if scheme != "catmull_clark":
            raise ValueError("SubdivisionRefinementResult.scheme must be 'catmull_clark'.")
        object.__setattr__(self, "control_points", control_points)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "control_points": self.control_points,
            "faces": self.faces,
            "level": self.level,
            "scheme": self.scheme,
            "metadata": self.metadata,
        }


def _catmull_clark_refine_once(
    control_points: np.ndarray,
    faces: tuple[tuple[int, ...], ...],
    crease_sharpness: dict[tuple[int, int], float],
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    face_points = np.asarray([control_points[list(face)].mean(axis=0) for face in faces], dtype=float)
    edge_faces = _subdivision_edge_faces(faces)
    edge_points: dict[tuple[int, int], np.ndarray] = {}
    for edge, adjacent_faces in edge_faces.items():
        start, end = edge
        is_sharp = crease_sharpness.get(edge, 0.0) > 0.0
        is_boundary = len(adjacent_faces) == 1
        if is_sharp or is_boundary:
            edge_points[edge] = (control_points[start] + control_points[end]) * 0.5
        else:
            edge_points[edge] = (
                control_points[start] + control_points[end] + face_points[adjacent_faces[0]] + face_points[adjacent_faces[1]]
            ) * 0.25

    adjacent_faces_by_vertex: dict[int, list[int]] = {index: [] for index in range(control_points.shape[0])}
    adjacent_edges_by_vertex: dict[int, list[tuple[int, int]]] = {index: [] for index in range(control_points.shape[0])}
    for face_index, face in enumerate(faces):
        for index in face:
            adjacent_faces_by_vertex[index].append(face_index)
    for edge in edge_faces:
        adjacent_edges_by_vertex[edge[0]].append(edge)
        adjacent_edges_by_vertex[edge[1]].append(edge)

    vertex_points = np.zeros_like(control_points)
    for index, point in enumerate(control_points):
        adjacent_edges = adjacent_edges_by_vertex[index]
        sharp_or_boundary_edges = [
            edge for edge in adjacent_edges if crease_sharpness.get(edge, 0.0) > 0.0 or len(edge_faces[edge]) == 1
        ]
        if len(sharp_or_boundary_edges) >= 2:
            neighbor_points = []
            for edge in sharp_or_boundary_edges[:2]:
                neighbor = edge[1] if edge[0] == index else edge[0]
                neighbor_points.append(control_points[neighbor])
            vertex_points[index] = ((6.0 * point) + neighbor_points[0] + neighbor_points[1]) * 0.125
            continue
        adjacent_faces = adjacent_faces_by_vertex[index]
        n = len(adjacent_faces)
        if n == 0:
            vertex_points[index] = point
            continue
        face_average = face_points[adjacent_faces].mean(axis=0)
        edge_midpoints = np.asarray(
            [(control_points[edge[0]] + control_points[edge[1]]) * 0.5 for edge in adjacent_edges],
            dtype=float,
        )
        edge_average = edge_midpoints.mean(axis=0)
        vertex_points[index] = (face_average + (2.0 * edge_average) + ((n - 3.0) * point)) / float(n)

    new_points: list[np.ndarray] = [point.copy() for point in vertex_points]
    edge_point_indices: dict[tuple[int, int], int] = {}
    for edge in sorted(edge_points):
        edge_point_indices[edge] = len(new_points)
        new_points.append(edge_points[edge])
    face_point_indices: list[int] = []
    for point in face_points:
        face_point_indices.append(len(new_points))
        new_points.append(point)

    new_faces: list[tuple[int, ...]] = []
    for face_index, face in enumerate(faces):
        face_point_index = face_point_indices[face_index]
        for local_index, vertex_index in enumerate(face):
            next_vertex = face[(local_index + 1) % len(face)]
            previous_vertex = face[(local_index - 1) % len(face)]
            next_edge = tuple(sorted((vertex_index, next_vertex)))
            previous_edge = tuple(sorted((previous_vertex, vertex_index)))
            new_faces.append(
                (
                    int(vertex_index),
                    edge_point_indices[(next_edge[0], next_edge[1])],
                    face_point_index,
                    edge_point_indices[(previous_edge[0], previous_edge[1])],
                )
            )
    return np.asarray(new_points, dtype=float), tuple(new_faces)


def refine_subdivision_control_cage(
    patch: "SubdivisionSurfacePatch",
    *,
    levels: int | None = None,
) -> SubdivisionRefinementResult:
    level_count = patch.subdivision_level if levels is None else int(levels)
    if level_count < 0:
        raise ValueError("Subdivision refinement levels must be >= 0.")
    control_points = patch.control_points.copy()
    faces = patch.faces
    crease_sharpness = {crease.edge: crease.sharpness for crease in patch.creases}
    for _level in range(level_count):
        control_points, faces = _catmull_clark_refine_once(control_points, faces, crease_sharpness)
        crease_sharpness = {}
    return SubdivisionRefinementResult(
        control_points=control_points,
        faces=faces,
        level=level_count,
        scheme=patch.scheme,
        metadata={
            "approximation": "finite_catmull_clark",
            "source_patch_id": patch.stable_identity,
        },
    )


@dataclass(frozen=True)
class SubdivisionSurfacePatch(SurfacePatch):
    """A subdivision surface payload with authored control cage and crease metadata."""

    control_points: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
    )
    faces: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3),)
    creases: tuple[SubdivisionCrease, ...] = field(default_factory=tuple)
    subdivision_level: int = 1
    scheme: SubdivisionScheme = "catmull_clark"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.family != "subdivision":
            raise ValueError("SubdivisionSurfacePatch.family must be 'subdivision'.")
        control_points = _as_control_points3(self.control_points, name="control_points")
        faces = _normalize_subdivision_faces(self.faces, control_point_count=control_points.shape[0])
        cage_edges = _subdivision_face_edges(faces)
        creases = tuple(crease if isinstance(crease, SubdivisionCrease) else SubdivisionCrease(**crease) for crease in self.creases)
        seen_creases: set[tuple[int, int]] = set()
        for crease in creases:
            if crease.edge not in cage_edges:
                raise ValueError("SubdivisionSurfacePatch creases must reference existing cage edges.")
            if crease.edge in seen_creases:
                raise ValueError("SubdivisionSurfacePatch creases must be unique per cage edge.")
            seen_creases.add(crease.edge)
        subdivision_level = int(self.subdivision_level)
        if subdivision_level < 0:
            raise ValueError("SubdivisionSurfacePatch.subdivision_level must be >= 0.")
        scheme = str(self.scheme)
        if scheme != "catmull_clark":
            raise ValueError("SubdivisionSurfacePatch.scheme must be 'catmull_clark'.")
        object.__setattr__(self, "control_points", control_points)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "creases", creases)
        object.__setattr__(self, "subdivision_level", subdivision_level)
        object.__setattr__(self, "scheme", scheme)

    def geometry_payload(self) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "subdivision_level": self.subdivision_level,
            "control_points": self.control_points,
            "faces": self.faces,
            "creases": [crease.canonical_payload() for crease in self.creases],
        }

    def refined_cage(self, *, levels: int | None = None) -> SubdivisionRefinementResult:
        return refine_subdivision_control_cage(self, levels=levels)

    def _normalized_parameters(self, u: float, v: float) -> tuple[float, float]:
        self.validate_parameters(u, v)
        u0, u1 = self.domain.u_range
        v0, v1 = self.domain.v_range
        u_norm = (float(u) - u0) / (u1 - u0)
        v_norm = (float(v) - v0) / (v1 - v0)
        return float(np.clip(u_norm, 0.0, 1.0)), float(np.clip(v_norm, 0.0, 1.0))

    def point_at(self, u: float, v: float) -> np.ndarray:
        u_norm, v_norm = self._normalized_parameters(u, v)
        result = self.refined_cage(levels=max(1, self.subdivision_level))
        face = result.faces[0]
        if len(face) < 4:
            raise ValueError("SubdivisionSurfacePatch finite evaluation requires a refined quad face.")
        a, b, c, d = (result.control_points[index] for index in face[:4])
        local = (
            ((1.0 - u_norm) * (1.0 - v_norm) * a)
            + (u_norm * (1.0 - v_norm) * b)
            + (u_norm * v_norm * c)
            + ((1.0 - u_norm) * v_norm * d)
        )
        return _transform_point(self.transform_matrix, local)

    def derivatives_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        self.validate_parameters(u, v)
        u_step = min(1e-5 * self.domain.u_span, self.domain.u_span * 0.25)
        v_step = min(1e-5 * self.domain.v_span, self.domain.v_span * 0.25)
        u_prev = max(self.domain.u_range[0], float(u) - u_step)
        u_next = min(self.domain.u_range[1], float(u) + u_step)
        v_prev = max(self.domain.v_range[0], float(v) - v_step)
        v_next = min(self.domain.v_range[1], float(v) + v_step)
        if np.isclose(u_prev, u_next):
            du = np.zeros(3, dtype=float)
        else:
            du = (self.point_at(u_next, v) - self.point_at(u_prev, v)) / (u_next - u_prev)
        if np.isclose(v_prev, v_next):
            dv = np.zeros(3, dtype=float)
        else:
            dv = (self.point_at(u, v_next) - self.point_at(u, v_prev)) / (v_next - v_prev)
        return du, dv

    def bounds_estimate(self, *, u_count: int = 3, v_count: int = 3) -> tuple[float, float, float, float, float, float]:
        del u_count, v_count
        transformed = np.asarray([_transform_point(self.transform_matrix, point) for point in self.control_points], dtype=float)
        mins = transformed.min(axis=0)
        maxs = transformed.max(axis=0)
        return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))


@dataclass(frozen=True)
class SurfaceShell:
    """An ordered collection of patches that form one shell."""

    patches: tuple[SurfacePatch, ...]
    connected: bool = True
    seams: tuple[SurfaceSeam, ...] = field(default_factory=tuple)
    adjacency: tuple[SurfaceAdjacencyRecord, ...] = field(default_factory=tuple)
    transform_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        patches = tuple(self.patches)
        seams = tuple(self.seams)
        adjacency = tuple(self.adjacency)
        if not patches:
            raise ValueError("SurfaceShell requires at least one patch.")
        if not all(isinstance(patch, SurfacePatch) for patch in patches):
            raise TypeError("SurfaceShell patches must all be SurfacePatch instances.")
        transform_matrix = _as_matrix4(self.transform_matrix, name="transform_matrix")
        metadata = _normalize_metadata(self.metadata)
        seam_ids = set()
        for seam in seams:
            if seam.seam_id in seam_ids:
                raise ValueError("SurfaceShell seam IDs must be unique.")
            seam_ids.add(seam.seam_id)
            for boundary in seam.boundaries:
                if boundary.patch_index >= len(patches):
                    raise ValueError("SurfaceSeam references a patch index outside the shell.")
        for record in adjacency:
            if record.source.patch_index >= len(patches):
                raise ValueError("SurfaceAdjacencyRecord source references a patch index outside the shell.")
            if record.target is not None and record.target.patch_index >= len(patches):
                raise ValueError("SurfaceAdjacencyRecord target references a patch index outside the shell.")
            if record.seam_id is not None and seam_ids and record.seam_id not in seam_ids:
                raise ValueError("SurfaceAdjacencyRecord references an unknown seam_id.")
        object.__setattr__(self, "patches", patches)
        object.__setattr__(self, "seams", seams)
        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(self, "transform_matrix", transform_matrix)
        object.__setattr__(self, "metadata", metadata)

    @property
    def patch_count(self) -> int:
        return len(self.patches)

    @property
    def stable_identity(self) -> str:
        return _stable_hash(self.canonical_payload())

    def kernel_metadata(self) -> dict[str, object]:
        kernel, _consumer = _split_metadata(self.metadata)
        return kernel

    def consumer_metadata(self) -> dict[str, object]:
        _kernel, consumer = _split_metadata(self.metadata)
        return consumer

    def merged_kernel_metadata(self, inherited: dict[str, object] | None = None) -> dict[str, object]:
        merged = {} if inherited is None else dict(inherited)
        merged.update(self.kernel_metadata())
        return merged

    def merged_consumer_metadata(self, inherited: dict[str, object] | None = None) -> dict[str, object]:
        merged = {} if inherited is None else dict(inherited)
        merged.update(self.consumer_metadata())
        return merged

    def canonical_payload(self) -> dict[str, object]:
        return {
            "patches": [patch.canonical_payload() for patch in self.patches],
            "connected": self.connected,
            "seams": [seam.canonical_payload() for seam in self.seams],
            "adjacency": [record.canonical_payload() for record in self.adjacency],
            "transform_matrix": self.transform_matrix,
            "metadata": self.metadata,
        }

    def with_transform(self, matrix: Sequence[Sequence[float]] | np.ndarray) -> "SurfaceShell":
        applied = _as_matrix4(matrix)
        return replace(self, transform_matrix=_compose_transform(self.transform_matrix, applied))

    def iter_patches(self, *, world: bool = True) -> tuple[SurfacePatch, ...]:
        if not world or _is_identity_matrix(self.transform_matrix):
            return self.patches
        return tuple(patch.with_transform(self.transform_matrix) for patch in self.patches)

    def adjacency_for_patch(self, patch_index: int) -> tuple[SurfaceAdjacencyRecord, ...]:
        patch_index = int(patch_index)
        return tuple(
            record
            for record in self.adjacency
            if record.source.patch_index == patch_index
            or (record.target is not None and record.target.patch_index == patch_index)
        )

    def bounds_estimate(self) -> tuple[float, float, float, float, float, float]:
        mins = np.array([np.inf, np.inf, np.inf], dtype=float)
        maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        for patch in self.iter_patches(world=True):
            bounds = patch.bounds_estimate()
            mins = np.minimum(mins, np.array([bounds[0], bounds[2], bounds[4]], dtype=float))
            maxs = np.maximum(maxs, np.array([bounds[1], bounds[3], bounds[5]], dtype=float))
        return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))


@dataclass(frozen=True)
class SurfaceBody:
    """An ordered collection of one or more shells."""

    shells: tuple[SurfaceShell, ...]
    transform_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shells = tuple(self.shells)
        if not shells:
            raise ValueError("SurfaceBody requires at least one shell.")
        if not all(isinstance(shell, SurfaceShell) for shell in shells):
            raise TypeError("SurfaceBody shells must all be SurfaceShell instances.")
        object.__setattr__(self, "shells", shells)
        object.__setattr__(self, "transform_matrix", _as_matrix4(self.transform_matrix, name="transform_matrix"))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    @property
    def shell_count(self) -> int:
        return len(self.shells)

    @property
    def patch_count(self) -> int:
        return sum(shell.patch_count for shell in self.shells)

    @property
    def stable_identity(self) -> str:
        return _stable_hash(self.canonical_payload())

    @property
    def cache_key(self) -> str:
        return self.stable_identity

    def kernel_metadata(self) -> dict[str, object]:
        kernel, _consumer = _split_metadata(self.metadata)
        return kernel

    def consumer_metadata(self) -> dict[str, object]:
        _kernel, consumer = _split_metadata(self.metadata)
        return consumer

    def merged_kernel_metadata(self, inherited: dict[str, object] | None = None) -> dict[str, object]:
        merged = {} if inherited is None else dict(inherited)
        merged.update(self.kernel_metadata())
        return merged

    def merged_consumer_metadata(self, inherited: dict[str, object] | None = None) -> dict[str, object]:
        merged = {} if inherited is None else dict(inherited)
        merged.update(self.consumer_metadata())
        return merged

    def canonical_payload(self) -> dict[str, object]:
        return {
            "shells": [shell.canonical_payload() for shell in self.shells],
            "transform_matrix": self.transform_matrix,
            "metadata": self.metadata,
        }

    def with_transform(self, matrix: Sequence[Sequence[float]] | np.ndarray) -> "SurfaceBody":
        applied = _as_matrix4(matrix)
        return replace(self, transform_matrix=_compose_transform(self.transform_matrix, applied))

    def iter_shells(self, *, world: bool = True) -> tuple[SurfaceShell, ...]:
        if not world or _is_identity_matrix(self.transform_matrix):
            return self.shells
        return tuple(shell.with_transform(self.transform_matrix) for shell in self.shells)

    def iter_patches(self, *, world: bool = True) -> tuple[SurfacePatch, ...]:
        return tuple(patch for shell in self.iter_shells(world=world) for patch in shell.iter_patches(world=world))

    def bounds_estimate(self) -> tuple[float, float, float, float, float, float]:
        mins = np.array([np.inf, np.inf, np.inf], dtype=float)
        maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        for shell in self.iter_shells(world=True):
            bounds = shell.bounds_estimate()
            mins = np.minimum(mins, np.array([bounds[0], bounds[2], bounds[4]], dtype=float))
            maxs = np.maximum(maxs, np.array([bounds[1], bounds[3], bounds[5]], dtype=float))
        return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))


def make_surface_shell(
    patches: Iterable[SurfacePatch],
    *,
    connected: bool = True,
    seams: Iterable[SurfaceSeam] = (),
    adjacency: Iterable[SurfaceAdjacencyRecord] = (),
    metadata: dict[str, object] | None = None,
) -> SurfaceShell:
    return SurfaceShell(
        tuple(patches),
        connected=connected,
        seams=tuple(seams),
        adjacency=tuple(adjacency),
        metadata=_normalize_metadata(metadata),
    )


def make_surface_body(
    shells: Iterable[SurfaceShell],
    *,
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    return SurfaceBody(tuple(shells), metadata=_normalize_metadata(metadata))


__all__ = [
    "PatchFamilyCapabilityRecord",
    "PATCH_FAMILY_CAPABILITY_MATRIX",
    "PATCH_FAMILY_FEATURE_COVERAGE",
    "REQUIRED_V1_PATCH_FAMILIES",
    "SUPPORTED_SURFACE_PATCH_FAMILIES",
    "SURFACE_SPEC_66_RETIREMENT_NOTE",
    "ParameterDomain",
    "TrimLoop",
    "SurfaceBoundaryRef",
    "SurfaceAdjacencyRecord",
    "SurfaceSeam",
    "SurfacePatch",
    "PlanarSurfacePatch",
    "RuledSurfacePatch",
    "RevolutionSurfacePatch",
    "BSplineSurfacePatch",
    "NURBSSurfacePatch",
    "SweepSurfacePatch",
    "SubdivisionCrease",
    "SubdivisionRefinementResult",
    "SubdivisionSurfacePatch",
    "refine_subdivision_control_cage",
    "SurfaceShell",
    "SurfaceBody",
    "make_surface_shell",
    "make_surface_body",
]
