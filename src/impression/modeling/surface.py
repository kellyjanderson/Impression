from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, Iterable, Literal, Sequence

import numpy as np


REQUIRED_V1_PATCH_FAMILIES: tuple[str, ...] = ("planar", "ruled", "revolution")
DEFERRED_V1_PATCH_FAMILIES: tuple[str, ...] = (
    "nurbs",
    "bspline",
    "subdivision",
    "implicit",
    "sweep",
)
PATCH_FAMILY_FEATURE_COVERAGE: dict[str, tuple[str, ...]] = {
    "planar": ("caps", "planar-primitives", "trimmed-faces"),
    "ruled": ("extrude", "loft", "linear-bridge-surfaces"),
    "revolution": ("rotate-extrude", "revolved-primitives"),
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
    "REQUIRED_V1_PATCH_FAMILIES",
    "DEFERRED_V1_PATCH_FAMILIES",
    "PATCH_FAMILY_FEATURE_COVERAGE",
    "ParameterDomain",
    "TrimLoop",
    "SurfaceBoundaryRef",
    "SurfaceAdjacencyRecord",
    "SurfaceSeam",
    "SurfacePatch",
    "PlanarSurfacePatch",
    "RuledSurfacePatch",
    "RevolutionSurfacePatch",
    "SurfaceShell",
    "SurfaceBody",
    "make_surface_shell",
    "make_surface_body",
]
