from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .surface import PATCH_FAMILY_CAPABILITY_MATRIX, SurfacePatch

SurfaceIntersectionSupportState = Literal["exact", "declared-tolerance", "adapter", "unsupported"]


@dataclass(frozen=True)
class SurfaceIntersectionTolerancePolicy:
    """Shared numeric policy for surface intersection dispatch."""

    position_tolerance: float = 1e-9
    parameter_tolerance: float = 1e-9
    degeneracy_tolerance: float = 1e-9
    max_iterations: int = 64

    def __post_init__(self) -> None:
        for name in ("position_tolerance", "parameter_tolerance", "degeneracy_tolerance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"SurfaceIntersectionTolerancePolicy.{name} must be a positive finite value.")
            object.__setattr__(self, name, value)
        max_iterations = int(self.max_iterations)
        if max_iterations <= 0:
            raise ValueError("SurfaceIntersectionTolerancePolicy.max_iterations must be positive.")
        object.__setattr__(self, "max_iterations", max_iterations)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "degeneracy_tolerance": self.degeneracy_tolerance,
            "max_iterations": self.max_iterations,
            "parameter_tolerance": self.parameter_tolerance,
            "position_tolerance": self.position_tolerance,
        }


DEFAULT_SURFACE_INTERSECTION_TOLERANCE_POLICY = SurfaceIntersectionTolerancePolicy()


@dataclass(frozen=True)
class SurfaceIntersectionRequest:
    """Canonical request shared by CSG, seam, loft, and future intersection consumers."""

    first_patch: SurfacePatch
    second_patch: SurfacePatch
    first_patch_ref: object | None = None
    second_patch_ref: object | None = None
    consumer: str = "surface-intersections"
    tolerance_policy: SurfaceIntersectionTolerancePolicy = DEFAULT_SURFACE_INTERSECTION_TOLERANCE_POLICY

    def __post_init__(self) -> None:
        if not isinstance(self.first_patch, SurfacePatch) or not isinstance(self.second_patch, SurfacePatch):
            raise TypeError("SurfaceIntersectionRequest patches must be SurfacePatch instances.")
        consumer = str(self.consumer).strip()
        if not consumer:
            raise ValueError("SurfaceIntersectionRequest.consumer must be non-empty.")
        object.__setattr__(self, "consumer", consumer)

    @property
    def family_pair(self) -> tuple[str, str]:
        return (self.first_patch.family, self.second_patch.family)

    @property
    def normalized_family_pair(self) -> tuple[str, str]:
        return tuple(sorted(self.family_pair))  # type: ignore[return-value]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "consumer": self.consumer,
            "family_pair": self.family_pair,
            "first_patch_family": self.first_patch.family,
            "second_patch_family": self.second_patch.family,
            "tolerance_policy": self.tolerance_policy.canonical_payload(),
        }


@dataclass(frozen=True)
class SurfaceIntersectionSolverRegistryRecord:
    """Registry entry for one normalized surface-family intersection solver."""

    first_family: str
    second_family: str
    solver_id: str
    support_state: SurfaceIntersectionSupportState
    operations: tuple[str, ...]
    diagnostic: str = ""

    def __post_init__(self) -> None:
        first_family = str(self.first_family).strip()
        second_family = str(self.second_family).strip()
        solver_id = str(self.solver_id).strip()
        support_state = str(self.support_state).strip()
        operations = tuple(str(operation).strip() for operation in self.operations)
        if not first_family or not second_family or not solver_id:
            raise ValueError("SurfaceIntersectionSolverRegistryRecord families and solver_id must be non-empty.")
        if support_state not in {"exact", "declared-tolerance", "adapter", "unsupported"}:
            raise ValueError("SurfaceIntersectionSolverRegistryRecord.support_state is invalid.")
        if not operations or not all(operations):
            raise ValueError("SurfaceIntersectionSolverRegistryRecord.operations must be non-empty.")
        diagnostic = str(self.diagnostic).strip()
        object.__setattr__(self, "first_family", first_family)
        object.__setattr__(self, "second_family", second_family)
        object.__setattr__(self, "solver_id", solver_id)
        object.__setattr__(self, "support_state", support_state)
        object.__setattr__(self, "operations", tuple(sorted(set(operations))))
        object.__setattr__(self, "diagnostic", diagnostic)

    @property
    def supported(self) -> bool:
        return self.support_state in {"exact", "declared-tolerance", "adapter"}

    @property
    def family_pair(self) -> tuple[str, str]:
        return (self.first_family, self.second_family)

    @property
    def normalized_family_pair(self) -> tuple[str, str]:
        return tuple(sorted(self.family_pair))  # type: ignore[return-value]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "diagnostic": self.diagnostic,
            "family_pair": self.family_pair,
            "operations": self.operations,
            "solver_id": self.solver_id,
            "support_state": self.support_state,
            "supported": self.supported,
        }


@dataclass(frozen=True)
class SurfaceIntersectionSupportDiagnostic:
    """Diagnostic for unsupported or unknown intersection solver dispatch."""

    code: Literal["unsupported-family-pair", "missing-registry-entry", "unknown-registry-entry"]
    message: str
    family_pair: tuple[str, str]
    consumer: str = "surface-intersections"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "consumer": self.consumer,
            "family_pair": self.family_pair,
            "message": self.message,
        }


@dataclass(frozen=True)
class SurfaceIntersectionSolverDispatchRecord:
    """Lookup result for a surface intersection request."""

    request: SurfaceIntersectionRequest
    solver: SurfaceIntersectionSolverRegistryRecord | None = None
    diagnostics: tuple[SurfaceIntersectionSupportDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return self.solver is not None and self.solver.supported and not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "request": self.request.canonical_payload(),
            "solver": None if self.solver is None else self.solver.canonical_payload(),
            "supported": self.supported,
        }


@dataclass(frozen=True)
class SurfaceIntersectionCurveRecord:
    """One normalized intersection curve or sampled curve segment."""

    curve_id: str
    kind: Literal["line", "arc", "spline", "sampled", "implicit-contour"]
    points_3d: tuple[tuple[float, float, float], ...]
    first_parameters: tuple[tuple[float, float], ...] = ()
    second_parameters: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        curve_id = str(self.curve_id).strip()
        kind = str(self.kind).strip()
        if not curve_id:
            raise ValueError("SurfaceIntersectionCurveRecord.curve_id must be non-empty.")
        if kind not in {"line", "arc", "spline", "sampled", "implicit-contour"}:
            raise ValueError("SurfaceIntersectionCurveRecord.kind is invalid.")
        points = tuple(_normalize_point3(point, name="points_3d") for point in self.points_3d)
        first_parameters = tuple(_normalize_point2(point, name="first_parameters") for point in self.first_parameters)
        second_parameters = tuple(_normalize_point2(point, name="second_parameters") for point in self.second_parameters)
        if len(points) < 2:
            raise ValueError("SurfaceIntersectionCurveRecord requires at least two 3D points.")
        if first_parameters and len(first_parameters) != len(points):
            raise ValueError("SurfaceIntersectionCurveRecord first_parameters must match points_3d length.")
        if second_parameters and len(second_parameters) != len(points):
            raise ValueError("SurfaceIntersectionCurveRecord second_parameters must match points_3d length.")
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "points_3d", points)
        object.__setattr__(self, "first_parameters", first_parameters)
        object.__setattr__(self, "second_parameters", second_parameters)

    @property
    def length_estimate(self) -> float:
        return _polyline_length(self.points_3d)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "curve_id": self.curve_id,
            "first_parameters": self.first_parameters,
            "kind": self.kind,
            "length_estimate": self.length_estimate,
            "points_3d": self.points_3d,
            "second_parameters": self.second_parameters,
        }


@dataclass(frozen=True)
class SurfaceIntersectionOverlapRegionRecord:
    """Parametric overlap region produced by coincident or tangent surface intersections."""

    region_id: str
    first_loop_uv: tuple[tuple[float, float], ...]
    second_loop_uv: tuple[tuple[float, float], ...]
    boundary_curve_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        region_id = str(self.region_id).strip()
        if not region_id:
            raise ValueError("SurfaceIntersectionOverlapRegionRecord.region_id must be non-empty.")
        first_loop = tuple(_normalize_point2(point, name="first_loop_uv") for point in self.first_loop_uv)
        second_loop = tuple(_normalize_point2(point, name="second_loop_uv") for point in self.second_loop_uv)
        if len(first_loop) < 3 or len(second_loop) < 3:
            raise ValueError("SurfaceIntersectionOverlapRegionRecord loops must contain at least three points.")
        object.__setattr__(self, "region_id", region_id)
        object.__setattr__(self, "first_loop_uv", first_loop)
        object.__setattr__(self, "second_loop_uv", second_loop)
        object.__setattr__(self, "boundary_curve_ids", tuple(sorted(str(item) for item in self.boundary_curve_ids)))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary_curve_ids": self.boundary_curve_ids,
            "first_loop_uv": self.first_loop_uv,
            "region_id": self.region_id,
            "second_loop_uv": self.second_loop_uv,
        }


@dataclass(frozen=True)
class SurfaceIntersectionDegeneracyRecord:
    """Classified degeneracy or quality issue for an intersection result."""

    code: Literal["none", "tangent", "point-contact", "overlap", "short-curve", "high-residual"]
    message: str
    curve_id: str | None = None
    residual: float | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "curve_id": self.curve_id,
            "message": self.message,
            "residual": self.residual,
        }


@dataclass(frozen=True)
class SurfaceIntersectionResultRecord:
    """Common normalized result for all surface-surface intersection solvers."""

    request: SurfaceIntersectionRequest
    classification: Literal["empty", "curves", "points", "overlap", "degenerate", "unsupported"]
    curves: tuple[SurfaceIntersectionCurveRecord, ...] = ()
    points: tuple[tuple[float, float, float], ...] = ()
    overlap_regions: tuple[SurfaceIntersectionOverlapRegionRecord, ...] = ()
    degeneracies: tuple[SurfaceIntersectionDegeneracyRecord, ...] = ()
    max_residual: float = 0.0
    quality: Literal["exact", "within-tolerance", "degenerate", "unsupported"] = "exact"

    @property
    def supported(self) -> bool:
        return self.classification != "unsupported" and self.quality != "unsupported"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "curves": [curve.canonical_payload() for curve in self.curves],
            "degeneracies": [record.canonical_payload() for record in self.degeneracies],
            "max_residual": self.max_residual,
            "overlap_regions": [region.canonical_payload() for region in self.overlap_regions],
            "points": self.points,
            "quality": self.quality,
            "request": self.request.canonical_payload(),
            "supported": self.supported,
        }


def _normalize_point2(value: Sequence[float], *, name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 2D point.")
    return (float(array[0]), float(array[1]))


def _normalize_point3(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3D point.")
    return (float(array[0]), float(array[1]), float(array[2]))


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    array = np.asarray(points, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(array, axis=0), axis=1)))


def classify_surface_intersection_degeneracy(
    result: SurfaceIntersectionResultRecord,
    *,
    tolerance_policy: SurfaceIntersectionTolerancePolicy | None = None,
) -> tuple[SurfaceIntersectionDegeneracyRecord, ...]:
    """Classify degeneracy records from normalized intersection result geometry."""

    policy = DEFAULT_SURFACE_INTERSECTION_TOLERANCE_POLICY if tolerance_policy is None else tolerance_policy
    degeneracies = list(result.degeneracies)
    if result.overlap_regions:
        degeneracies.append(
            SurfaceIntersectionDegeneracyRecord(
                code="overlap",
                message="Surface intersection produced one or more overlap regions.",
            )
        )
    if result.points and not result.curves and not result.overlap_regions:
        degeneracies.append(
            SurfaceIntersectionDegeneracyRecord(
                code="point-contact",
                message="Surface intersection produced isolated point contact.",
            )
        )
    for curve in result.curves:
        if curve.length_estimate <= policy.degeneracy_tolerance:
            degeneracies.append(
                SurfaceIntersectionDegeneracyRecord(
                    code="short-curve",
                    curve_id=curve.curve_id,
                    message="Surface intersection curve is shorter than degeneracy tolerance.",
                )
            )
    if result.max_residual > policy.position_tolerance:
        degeneracies.append(
            SurfaceIntersectionDegeneracyRecord(
                code="high-residual",
                message="Surface intersection residual exceeds position tolerance.",
                residual=result.max_residual,
            )
        )
    seen: set[tuple[object, ...]] = set()
    unique: list[SurfaceIntersectionDegeneracyRecord] = []
    for record in degeneracies:
        key = (record.code, record.curve_id, record.residual, record.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return tuple(unique)


def normalize_surface_intersection_result(
    request: SurfaceIntersectionRequest,
    *,
    curves: Sequence[SurfaceIntersectionCurveRecord] = (),
    points: Sequence[Sequence[float]] = (),
    overlap_regions: Sequence[SurfaceIntersectionOverlapRegionRecord] = (),
    max_residual: float = 0.0,
    quality: Literal["exact", "within-tolerance", "degenerate", "unsupported"] = "exact",
    degeneracies: Sequence[SurfaceIntersectionDegeneracyRecord] = (),
) -> SurfaceIntersectionResultRecord:
    """Normalize solver output into a deterministic surface intersection result record."""

    normalized_curves = tuple(sorted(curves, key=lambda curve: curve.curve_id))
    normalized_points = tuple(sorted(_normalize_point3(point, name="points") for point in points))
    normalized_regions = tuple(sorted(overlap_regions, key=lambda region: region.region_id))
    residual = float(max_residual)
    if not np.isfinite(residual) or residual < 0.0:
        raise ValueError("normalize_surface_intersection_result max_residual must be finite and non-negative.")
    if quality not in {"exact", "within-tolerance", "degenerate", "unsupported"}:
        raise ValueError("normalize_surface_intersection_result quality is invalid.")
    if quality == "unsupported":
        classification: Literal["empty", "curves", "points", "overlap", "degenerate", "unsupported"] = "unsupported"
    elif normalized_regions:
        classification = "overlap"
    elif normalized_curves:
        classification = "curves"
    elif normalized_points:
        classification = "points"
    else:
        classification = "empty"
    base = SurfaceIntersectionResultRecord(
        request=request,
        classification=classification,
        curves=normalized_curves,
        points=normalized_points,
        overlap_regions=normalized_regions,
        degeneracies=tuple(degeneracies),
        max_residual=residual,
        quality=quality,
    )
    classified_degeneracies = classify_surface_intersection_degeneracy(base)
    if classified_degeneracies and classification != "unsupported":
        classification = "degenerate" if not normalized_curves and not normalized_regions else classification
        quality = "degenerate" if quality == "exact" else quality
    return SurfaceIntersectionResultRecord(
        request=request,
        classification=classification,
        curves=normalized_curves,
        points=normalized_points,
        overlap_regions=normalized_regions,
        degeneracies=classified_degeneracies,
        max_residual=residual,
        quality=quality,
    )


def _promoted_surface_family_names() -> tuple[str, ...]:
    return tuple(sorted(PATCH_FAMILY_CAPABILITY_MATRIX))


def _default_surface_intersection_solver_records() -> tuple[SurfaceIntersectionSolverRegistryRecord, ...]:
    exact_pairs = {
        ("planar", "planar"): "planar-planar-analytic",
        ("planar", "ruled"): "planar-ruled-analytic",
        ("planar", "revolution"): "planar-revolution-analytic",
        ("revolution", "revolution"): "axis-compatible-revolution-analytic",
    }
    records: list[SurfaceIntersectionSolverRegistryRecord] = []
    families = _promoted_surface_family_names()
    for index, first_family in enumerate(families):
        for second_family in families[index:]:
            pair = tuple(sorted((first_family, second_family)))
            solver_id = exact_pairs.get(pair)
            if solver_id is None:
                records.append(
                    SurfaceIntersectionSolverRegistryRecord(
                        first_family=pair[0],
                        second_family=pair[1],
                        solver_id=f"{pair[0]}-{pair[1]}-unsupported",
                        support_state="unsupported",
                        operations=("classification", "csg", "seam"),
                        diagnostic=(
                            f"surface intersection solver for {pair[0]}/{pair[1]} "
                            "is not implemented in this registry"
                        ),
                    )
                )
            else:
                records.append(
                    SurfaceIntersectionSolverRegistryRecord(
                        first_family=pair[0],
                        second_family=pair[1],
                        solver_id=solver_id,
                        support_state="exact",
                        operations=("classification", "csg", "seam"),
                    )
                )
    return tuple(records)


def build_surface_intersection_solver_registry(
    records: Sequence[SurfaceIntersectionSolverRegistryRecord] | None = None,
) -> dict[tuple[str, str], SurfaceIntersectionSolverRegistryRecord]:
    """Build a deterministic normalized family-pair solver registry."""

    source = _default_surface_intersection_solver_records() if records is None else tuple(records)
    registry: dict[tuple[str, str], SurfaceIntersectionSolverRegistryRecord] = {}
    for record in source:
        registry[record.normalized_family_pair] = record
    return dict(sorted(registry.items()))


def assert_surface_intersection_solver_registry_complete(
    registry: dict[tuple[str, str], SurfaceIntersectionSolverRegistryRecord] | None = None,
) -> dict[tuple[str, str], SurfaceIntersectionSolverRegistryRecord]:
    """Assert every promoted surface family pair has an explicit registry entry."""

    checked = build_surface_intersection_solver_registry() if registry is None else registry
    families = _promoted_surface_family_names()
    expected = {
        tuple(sorted((first_family, second_family)))
        for index, first_family in enumerate(families)
        for second_family in families[index:]
    }
    actual = set(checked)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        messages = []
        if missing:
            messages.append(f"missing surface intersection registry entries: {missing!r}")
        if unknown:
            messages.append(f"unknown surface intersection registry entries: {unknown!r}")
        raise AssertionError("; ".join(messages))
    return checked


SURFACE_INTERSECTION_SOLVER_REGISTRY = assert_surface_intersection_solver_registry_complete()


def make_surface_intersection_request(
    first_patch: SurfacePatch,
    second_patch: SurfacePatch,
    *,
    first_patch_ref: object | None = None,
    second_patch_ref: object | None = None,
    consumer: str = "surface-intersections",
    tolerance_policy: SurfaceIntersectionTolerancePolicy | None = None,
) -> SurfaceIntersectionRequest:
    """Build a canonical surface intersection request."""

    return SurfaceIntersectionRequest(
        first_patch=first_patch,
        second_patch=second_patch,
        first_patch_ref=first_patch_ref,
        second_patch_ref=second_patch_ref,
        consumer=consumer,
        tolerance_policy=DEFAULT_SURFACE_INTERSECTION_TOLERANCE_POLICY
        if tolerance_policy is None
        else tolerance_policy,
    )


def lookup_surface_intersection_solver(
    request: SurfaceIntersectionRequest,
    registry: dict[tuple[str, str], SurfaceIntersectionSolverRegistryRecord] | None = None,
) -> SurfaceIntersectionSolverDispatchRecord:
    """Look up an intersection solver for a canonical request."""

    checked = SURFACE_INTERSECTION_SOLVER_REGISTRY if registry is None else registry
    key = request.normalized_family_pair
    solver = checked.get(key)
    if solver is None:
        return SurfaceIntersectionSolverDispatchRecord(
            request=request,
            diagnostics=(
                SurfaceIntersectionSupportDiagnostic(
                    code="missing-registry-entry",
                    consumer=request.consumer,
                    family_pair=key,
                    message=f"surface intersection registry has no entry for {key[0]}/{key[1]}",
                ),
            ),
        )
    if not solver.supported:
        return SurfaceIntersectionSolverDispatchRecord(
            request=request,
            solver=solver,
            diagnostics=(
                SurfaceIntersectionSupportDiagnostic(
                    code="unsupported-family-pair",
                    consumer=request.consumer,
                    family_pair=key,
                    message=solver.diagnostic
                    or f"surface intersection pair {key[0]}/{key[1]} is unsupported",
                ),
            ),
        )
    return SurfaceIntersectionSolverDispatchRecord(request=request, solver=solver)


__all__ = [
    "DEFAULT_SURFACE_INTERSECTION_TOLERANCE_POLICY",
    "SURFACE_INTERSECTION_SOLVER_REGISTRY",
    "SurfaceIntersectionCurveRecord",
    "SurfaceIntersectionDegeneracyRecord",
    "SurfaceIntersectionOverlapRegionRecord",
    "SurfaceIntersectionRequest",
    "SurfaceIntersectionResultRecord",
    "SurfaceIntersectionSolverDispatchRecord",
    "SurfaceIntersectionSolverRegistryRecord",
    "SurfaceIntersectionSupportDiagnostic",
    "SurfaceIntersectionSupportState",
    "SurfaceIntersectionTolerancePolicy",
    "assert_surface_intersection_solver_registry_complete",
    "build_surface_intersection_solver_registry",
    "classify_surface_intersection_degeneracy",
    "lookup_surface_intersection_solver",
    "make_surface_intersection_request",
    "normalize_surface_intersection_result",
]
