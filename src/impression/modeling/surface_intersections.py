from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .surface import (
    PATCH_FAMILY_CAPABILITY_MATRIX,
    BSplineSurfacePatch,
    NURBSSurfacePatch,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SubdivisionSurfacePatch,
    SurfacePatch,
)

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

    code: Literal[
        "unsupported-family-pair",
        "missing-registry-entry",
        "unknown-registry-entry",
        "budget-exhausted",
    ]
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
class SurfaceAnalyticSplineSolverIterationRecord:
    """Bounded sampling/refinement witness for an analytic-to-spline solve."""

    sample_count: int
    accepted_point_count: int
    max_residual: float
    converged: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "accepted_point_count": self.accepted_point_count,
            "converged": self.converged,
            "max_residual": self.max_residual,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class SurfaceAnalyticSplineResidualReport:
    """Residual and diagnostic report for an analytic-to-spline solve."""

    solver_id: str
    iterations: tuple[SurfaceAnalyticSplineSolverIterationRecord, ...]
    diagnostics: tuple[SurfaceIntersectionSupportDiagnostic, ...] = ()

    @property
    def converged(self) -> bool:
        return bool(self.iterations) and self.iterations[-1].converged and not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "iterations": [iteration.canonical_payload() for iteration in self.iterations],
            "solver_id": self.solver_id,
        }


@dataclass(frozen=True)
class SurfaceSplineSplineSolverIterationRecord:
    """Bounded pairing/refinement witness for spline-to-spline solves."""

    first_sample_count: int
    second_sample_count: int
    accepted_pair_count: int
    max_residual: float
    converged: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "accepted_pair_count": self.accepted_pair_count,
            "converged": self.converged,
            "first_sample_count": self.first_sample_count,
            "max_residual": self.max_residual,
            "second_sample_count": self.second_sample_count,
        }


@dataclass(frozen=True)
class SurfaceSplineSplineResidualReport:
    """Residual and diagnostic report for a spline-to-spline solve."""

    solver_id: str
    iterations: tuple[SurfaceSplineSplineSolverIterationRecord, ...]
    diagnostics: tuple[SurfaceIntersectionSupportDiagnostic, ...] = ()

    @property
    def converged(self) -> bool:
        return bool(self.iterations) and self.iterations[-1].converged and not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "iterations": [iteration.canonical_payload() for iteration in self.iterations],
            "solver_id": self.solver_id,
        }


@dataclass(frozen=True)
class SurfaceSubdivisionIntersectionBudget:
    """Bounded refinement and contour budget for subdivision intersection adapters."""

    max_refinement_level: int = 2
    max_sample_count: int = 81
    max_contour_count: int = 8

    def __post_init__(self) -> None:
        max_refinement_level = int(self.max_refinement_level)
        max_sample_count = int(self.max_sample_count)
        max_contour_count = int(self.max_contour_count)
        if max_refinement_level < 0:
            raise ValueError("SurfaceSubdivisionIntersectionBudget.max_refinement_level must be non-negative.")
        if max_sample_count <= 0:
            raise ValueError("SurfaceSubdivisionIntersectionBudget.max_sample_count must be positive.")
        if max_contour_count <= 0:
            raise ValueError("SurfaceSubdivisionIntersectionBudget.max_contour_count must be positive.")
        object.__setattr__(self, "max_refinement_level", max_refinement_level)
        object.__setattr__(self, "max_sample_count", max_sample_count)
        object.__setattr__(self, "max_contour_count", max_contour_count)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "max_contour_count": self.max_contour_count,
            "max_refinement_level": self.max_refinement_level,
            "max_sample_count": self.max_sample_count,
        }


DEFAULT_SURFACE_SUBDIVISION_INTERSECTION_BUDGET = SurfaceSubdivisionIntersectionBudget()


@dataclass(frozen=True)
class SurfaceSubdivisionRefinedContourRecord:
    """Subdivision adapter contour witness produced from refined surface evaluation."""

    contour_id: str
    refinement_level: int
    curve: SurfaceIntersectionCurveRecord
    sample_count: int
    max_residual: float

    def __post_init__(self) -> None:
        contour_id = str(self.contour_id).strip()
        refinement_level = int(self.refinement_level)
        sample_count = int(self.sample_count)
        max_residual = float(self.max_residual)
        if not contour_id:
            raise ValueError("SurfaceSubdivisionRefinedContourRecord.contour_id must be non-empty.")
        if refinement_level < 0:
            raise ValueError("SurfaceSubdivisionRefinedContourRecord.refinement_level must be non-negative.")
        if not isinstance(self.curve, SurfaceIntersectionCurveRecord):
            raise TypeError("SurfaceSubdivisionRefinedContourRecord.curve must be a SurfaceIntersectionCurveRecord.")
        if sample_count <= 0:
            raise ValueError("SurfaceSubdivisionRefinedContourRecord.sample_count must be positive.")
        if not np.isfinite(max_residual) or max_residual < 0.0:
            raise ValueError("SurfaceSubdivisionRefinedContourRecord.max_residual must be finite and non-negative.")
        object.__setattr__(self, "contour_id", contour_id)
        object.__setattr__(self, "refinement_level", refinement_level)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "max_residual", max_residual)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "contour_id": self.contour_id,
            "curve": self.curve.canonical_payload(),
            "max_residual": self.max_residual,
            "refinement_level": self.refinement_level,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class SurfaceSubdivisionIntersectionAdapterReport:
    """Report for bounded subdivision intersection adapter execution."""

    solver_id: str
    budget: SurfaceSubdivisionIntersectionBudget
    contours: tuple[SurfaceSubdivisionRefinedContourRecord, ...] = ()
    diagnostics: tuple[SurfaceIntersectionSupportDiagnostic, ...] = ()

    @property
    def converged(self) -> bool:
        return bool(self.contours) and not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "budget": self.budget.canonical_payload(),
            "contours": [contour.canonical_payload() for contour in self.contours],
            "converged": self.converged,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "solver_id": self.solver_id,
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


def _surface_intersection_is_spline_family(patch: SurfacePatch) -> bool:
    return isinstance(patch, (BSplineSurfacePatch, NURBSSurfacePatch))


def _surface_intersection_is_subdivision_family(patch: SurfacePatch) -> bool:
    return isinstance(patch, SubdivisionSurfacePatch)


def _surface_intersection_is_analytic_family(patch: SurfacePatch) -> bool:
    return isinstance(patch, (PlanarSurfacePatch, RuledSurfacePatch, RevolutionSurfacePatch))


def _analytic_surface_residual(analytic: SurfacePatch, point: np.ndarray) -> float:
    if isinstance(analytic, PlanarSurfacePatch):
        du, dv = analytic.derivatives_at(0.5, 0.5)
        normal = np.cross(du, dv)
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            return float("inf")
        normal = normal / norm
        origin = analytic.point_at(0.5, 0.5)
        return abs(float(np.dot(point - origin, normal)))
    # Declared-tolerance fallback for ruled/revolution analytic surfaces:
    # sample the analytic parameter space deterministically and use closest
    # surface distance as the residual witness.
    u0, u1 = analytic.domain.u_range
    v0, v1 = analytic.domain.v_range
    samples = (
        analytic.point_at(float(u), float(v))
        for u in np.linspace(u0, u1, 9)
        for v in np.linspace(v0, v1, 9)
    )
    return min(float(np.linalg.norm(point - sample)) for sample in samples)


def solve_analytic_spline_surface_intersection(
    request: SurfaceIntersectionRequest,
    *,
    sample_count: int = 17,
) -> tuple[SurfaceIntersectionResultRecord, SurfaceAnalyticSplineResidualReport]:
    """Solve analytic surface against B-spline/NURBS by bounded declared-tolerance sampling."""

    dispatch = lookup_surface_intersection_solver(request)
    if not dispatch.supported or dispatch.solver is None or dispatch.solver.solver_id != "analytic-spline-declared-tolerance":
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer=request.consumer,
            family_pair=request.normalized_family_pair,
            message="analytic-to-spline solver requires a declared-tolerance analytic/spline registry dispatch",
        )
        result = normalize_surface_intersection_result(
            request,
            quality="unsupported",
            degeneracies=(
                SurfaceIntersectionDegeneracyRecord(
                    code="high-residual",
                    message=diagnostic.message,
                ),
            ),
        )
        return (
            result,
            SurfaceAnalyticSplineResidualReport(
                solver_id="analytic-spline-declared-tolerance",
                iterations=(),
                diagnostics=(diagnostic,),
            ),
        )
    first_is_analytic = _surface_intersection_is_analytic_family(request.first_patch)
    second_is_analytic = _surface_intersection_is_analytic_family(request.second_patch)
    first_is_spline = _surface_intersection_is_spline_family(request.first_patch)
    second_is_spline = _surface_intersection_is_spline_family(request.second_patch)
    if first_is_analytic and second_is_spline:
        analytic = request.first_patch
        spline = request.second_patch
        analytic_is_first = True
    elif second_is_analytic and first_is_spline:
        analytic = request.second_patch
        spline = request.first_patch
        analytic_is_first = False
    else:
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer=request.consumer,
            family_pair=request.normalized_family_pair,
            message="analytic-to-spline solver requires exactly one analytic and one spline surface",
        )
        result = normalize_surface_intersection_result(request, quality="unsupported")
        return (
            result,
            SurfaceAnalyticSplineResidualReport(
                solver_id=dispatch.solver.solver_id,
                iterations=(),
                diagnostics=(diagnostic,),
            ),
        )

    sample_count = max(3, int(sample_count))
    u0, u1 = spline.domain.u_range
    v0, v1 = spline.domain.v_range
    accepted: list[tuple[tuple[float, float, float], tuple[float, float], float]] = []
    threshold = max(request.tolerance_policy.position_tolerance * 10.0, request.tolerance_policy.degeneracy_tolerance)
    max_residual = 0.0
    for u in np.linspace(u0, u1, sample_count):
        for v in np.linspace(v0, v1, sample_count):
            point = spline.point_at(float(u), float(v))
            residual = _analytic_surface_residual(analytic, point)
            max_residual = max(max_residual, residual if np.isfinite(residual) else float("inf"))
            if residual <= threshold:
                accepted.append((tuple(float(component) for component in point), (float(u), float(v)), residual))

    # Keep a deterministic curve-like ordering. This is a declared-tolerance
    # seed/refinement boundary; later solver specs can replace the sampler with
    # exact marching while preserving this result shape.
    accepted = sorted(set(accepted), key=lambda item: (item[1], item[0]))
    converged = len(accepted) >= 2 and max((item[2] for item in accepted), default=float("inf")) <= threshold
    iteration = SurfaceAnalyticSplineSolverIterationRecord(
        sample_count=sample_count * sample_count,
        accepted_point_count=len(accepted),
        max_residual=max((item[2] for item in accepted), default=max_residual),
        converged=converged,
    )
    diagnostics: tuple[SurfaceIntersectionSupportDiagnostic, ...] = ()
    if not converged:
        diagnostics = (
            SurfaceIntersectionSupportDiagnostic(
                code="unsupported-family-pair",
                consumer=request.consumer,
                family_pair=request.normalized_family_pair,
                message="analytic-to-spline solver did not converge within declared tolerance sampling budget",
            ),
        )
    if converged:
        points = tuple(item[0] for item in accepted)
        spline_parameters = tuple(item[1] for item in accepted)
        empty_parameters: tuple[tuple[float, float], ...] = ()
        curve = SurfaceIntersectionCurveRecord(
            curve_id="analytic-spline-curve-0",
            kind="sampled",
            points_3d=points,
            first_parameters=empty_parameters if analytic_is_first else spline_parameters,
            second_parameters=spline_parameters if analytic_is_first else empty_parameters,
        )
        result = normalize_surface_intersection_result(
            request,
            curves=(curve,),
            max_residual=iteration.max_residual,
            quality="within-tolerance",
        )
    else:
        result = normalize_surface_intersection_result(
            request,
            max_residual=max_residual if np.isfinite(max_residual) else 0.0,
            quality="unsupported",
        )
    return (
        result,
        SurfaceAnalyticSplineResidualReport(
            solver_id=dispatch.solver.solver_id,
            iterations=(iteration,),
            diagnostics=diagnostics,
        ),
    )


def _sample_spline_surface_points(
    patch: SurfacePatch,
    sample_count: int,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float]], ...]:
    return _sample_surface_points(patch, sample_count)


def _sample_surface_points(
    patch: SurfacePatch,
    sample_count: int,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float]], ...]:
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    return tuple(
        (
            tuple(float(component) for component in patch.point_at(float(u), float(v))),
            (float(u), float(v)),
        )
        for u in np.linspace(u0, u1, sample_count)
        for v in np.linspace(v0, v1, sample_count)
    )


def check_subdivision_intersection_budget(
    request: SurfaceIntersectionRequest,
    *,
    budget: SurfaceSubdivisionIntersectionBudget = DEFAULT_SURFACE_SUBDIVISION_INTERSECTION_BUDGET,
    sample_count: int,
    contour_count: int = 1,
) -> tuple[SurfaceIntersectionSupportDiagnostic, ...]:
    """Return deterministic diagnostics when a subdivision adapter would exceed budget."""

    diagnostics: list[SurfaceIntersectionSupportDiagnostic] = []
    if sample_count * sample_count > budget.max_sample_count:
        diagnostics.append(
            SurfaceIntersectionSupportDiagnostic(
                code="budget-exhausted",
                consumer=request.consumer,
                family_pair=request.normalized_family_pair,
                message=(
                    "subdivision intersection adapter sample budget exhausted: "
                    f"{sample_count * sample_count} samples exceeds {budget.max_sample_count}"
                ),
            )
        )
    if contour_count > budget.max_contour_count:
        diagnostics.append(
            SurfaceIntersectionSupportDiagnostic(
                code="budget-exhausted",
                consumer=request.consumer,
                family_pair=request.normalized_family_pair,
                message=(
                    "subdivision intersection adapter contour budget exhausted: "
                    f"{contour_count} contours exceeds {budget.max_contour_count}"
                ),
            )
        )
    for patch in (request.first_patch, request.second_patch):
        if isinstance(patch, SubdivisionSurfacePatch) and patch.subdivision_level > budget.max_refinement_level:
            diagnostics.append(
                SurfaceIntersectionSupportDiagnostic(
                    code="budget-exhausted",
                    consumer=request.consumer,
                    family_pair=request.normalized_family_pair,
                    message=(
                        "subdivision intersection adapter refinement budget exhausted: "
                        f"level {patch.subdivision_level} exceeds {budget.max_refinement_level}"
                    ),
                )
            )
    return tuple(diagnostics)


def solve_subdivision_surface_intersection_adapter(
    request: SurfaceIntersectionRequest,
    *,
    budget: SurfaceSubdivisionIntersectionBudget = DEFAULT_SURFACE_SUBDIVISION_INTERSECTION_BUDGET,
    sample_count: int | None = None,
) -> tuple[SurfaceIntersectionResultRecord, SurfaceSubdivisionIntersectionAdapterReport]:
    """Intersect subdivision pairs through bounded surface-native evaluation records."""

    dispatch = lookup_surface_intersection_solver(request)
    if not dispatch.supported or dispatch.solver is None or dispatch.solver.solver_id != "subdivision-adapter-declared-tolerance":
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer=request.consumer,
            family_pair=request.normalized_family_pair,
            message="subdivision intersection adapter requires an adapter-supported subdivision registry dispatch",
        )
        return (
            normalize_surface_intersection_result(request, quality="unsupported"),
            SurfaceSubdivisionIntersectionAdapterReport(
                solver_id="subdivision-adapter-declared-tolerance",
                budget=budget,
                diagnostics=(diagnostic,),
            ),
        )
    if not (_surface_intersection_is_subdivision_family(request.first_patch) or _surface_intersection_is_subdivision_family(request.second_patch)):
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer=request.consumer,
            family_pair=request.normalized_family_pair,
            message="subdivision intersection adapter requires at least one subdivision surface",
        )
        return (
            normalize_surface_intersection_result(request, quality="unsupported"),
            SurfaceSubdivisionIntersectionAdapterReport(
                solver_id=dispatch.solver.solver_id,
                budget=budget,
                diagnostics=(diagnostic,),
            ),
        )

    if sample_count is None:
        sample_count = max(3, min(9, int(np.floor(np.sqrt(budget.max_sample_count)))))
    sample_count = max(3, int(sample_count))
    budget_diagnostics = check_subdivision_intersection_budget(
        request,
        budget=budget,
        sample_count=sample_count,
        contour_count=1,
    )
    if budget_diagnostics:
        return (
            normalize_surface_intersection_result(request, quality="unsupported"),
            SurfaceSubdivisionIntersectionAdapterReport(
                solver_id=dispatch.solver.solver_id,
                budget=budget,
                diagnostics=budget_diagnostics,
            ),
        )

    first_samples = _sample_surface_points(request.first_patch, sample_count)
    second_samples = _sample_surface_points(request.second_patch, sample_count)
    threshold = max(request.tolerance_policy.position_tolerance * 10.0, request.tolerance_policy.degeneracy_tolerance)
    accepted: list[tuple[tuple[float, float, float], tuple[float, float], tuple[float, float], float]] = []
    for first_point, first_uv in first_samples:
        first_array = np.asarray(first_point, dtype=float)
        nearest_point, nearest_uv = min(
            second_samples,
            key=lambda sample: float(np.linalg.norm(first_array - np.asarray(sample[0], dtype=float))),
        )
        residual = float(np.linalg.norm(first_array - np.asarray(nearest_point, dtype=float)))
        if residual <= threshold:
            midpoint = tuple(float(component) for component in ((first_array + np.asarray(nearest_point, dtype=float)) * 0.5))
            accepted.append((midpoint, first_uv, nearest_uv, residual))

    accepted = sorted(set(accepted), key=lambda item: (item[1], item[2], item[0]))
    max_residual = max((item[3] for item in accepted), default=float("inf"))
    if len(accepted) < 2 or max_residual > threshold:
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer=request.consumer,
            family_pair=request.normalized_family_pair,
            message="subdivision intersection adapter did not converge within declared tolerance sampling budget",
        )
        return (
            normalize_surface_intersection_result(
                request,
                max_residual=max_residual if np.isfinite(max_residual) else 0.0,
                quality="unsupported",
            ),
            SurfaceSubdivisionIntersectionAdapterReport(
                solver_id=dispatch.solver.solver_id,
                budget=budget,
                diagnostics=(diagnostic,),
            ),
        )

    curve = SurfaceIntersectionCurveRecord(
        curve_id="subdivision-adapter-contour-0",
        kind="sampled",
        points_3d=tuple(item[0] for item in accepted),
        first_parameters=tuple(item[1] for item in accepted),
        second_parameters=tuple(item[2] for item in accepted),
    )
    refinement_level = max(
        (patch.subdivision_level for patch in (request.first_patch, request.second_patch) if isinstance(patch, SubdivisionSurfacePatch)),
        default=0,
    )
    contour = SurfaceSubdivisionRefinedContourRecord(
        contour_id="subdivision-adapter-contour-0",
        refinement_level=refinement_level,
        curve=curve,
        sample_count=sample_count * sample_count,
        max_residual=max_residual,
    )
    result = normalize_surface_intersection_result(
        request,
        curves=(curve,),
        max_residual=max_residual,
        quality="within-tolerance",
    )
    return (
        result,
        SurfaceSubdivisionIntersectionAdapterReport(
            solver_id=dispatch.solver.solver_id,
            budget=budget,
            contours=(contour,),
        ),
    )


def solve_spline_spline_surface_intersection(
    request: SurfaceIntersectionRequest,
    *,
    sample_count: int = 13,
) -> tuple[SurfaceIntersectionResultRecord, SurfaceSplineSplineResidualReport]:
    """Solve B-spline/NURBS pairs by bounded declared-tolerance closest sample pairing."""

    dispatch = lookup_surface_intersection_solver(request)
    if not dispatch.supported or dispatch.solver is None or dispatch.solver.solver_id != "spline-spline-declared-tolerance":
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer=request.consumer,
            family_pair=request.normalized_family_pair,
            message="spline-to-spline solver requires a declared-tolerance spline/spline registry dispatch",
        )
        return (
            normalize_surface_intersection_result(request, quality="unsupported"),
            SurfaceSplineSplineResidualReport(
                solver_id="spline-spline-declared-tolerance",
                iterations=(),
                diagnostics=(diagnostic,),
            ),
        )
    if not (_surface_intersection_is_spline_family(request.first_patch) and _surface_intersection_is_spline_family(request.second_patch)):
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer=request.consumer,
            family_pair=request.normalized_family_pair,
            message="spline-to-spline solver requires two B-spline or NURBS surfaces",
        )
        return (
            normalize_surface_intersection_result(request, quality="unsupported"),
            SurfaceSplineSplineResidualReport(
                solver_id=dispatch.solver.solver_id,
                iterations=(),
                diagnostics=(diagnostic,),
            ),
        )

    sample_count = max(3, int(sample_count))
    first_samples = _sample_spline_surface_points(request.first_patch, sample_count)
    second_samples = _sample_spline_surface_points(request.second_patch, sample_count)
    threshold = max(request.tolerance_policy.position_tolerance * 10.0, request.tolerance_policy.degeneracy_tolerance)
    accepted: list[tuple[tuple[float, float, float], tuple[float, float], tuple[float, float], float]] = []
    for first_point, first_uv in first_samples:
        first_array = np.asarray(first_point, dtype=float)
        nearest_point, nearest_uv = min(
            second_samples,
            key=lambda sample: float(np.linalg.norm(first_array - np.asarray(sample[0], dtype=float))),
        )
        residual = float(np.linalg.norm(first_array - np.asarray(nearest_point, dtype=float)))
        if residual <= threshold:
            midpoint = tuple(float(component) for component in ((first_array + np.asarray(nearest_point, dtype=float)) * 0.5))
            accepted.append((midpoint, first_uv, nearest_uv, residual))

    accepted = sorted(set(accepted), key=lambda item: (item[1], item[2], item[0]))
    max_residual = max((item[3] for item in accepted), default=float("inf"))
    converged = len(accepted) >= 2 and max_residual <= threshold
    iteration = SurfaceSplineSplineSolverIterationRecord(
        first_sample_count=len(first_samples),
        second_sample_count=len(second_samples),
        accepted_pair_count=len(accepted),
        max_residual=max_residual if np.isfinite(max_residual) else 0.0,
        converged=converged,
    )
    diagnostics: tuple[SurfaceIntersectionSupportDiagnostic, ...] = ()
    if not converged:
        diagnostics = (
            SurfaceIntersectionSupportDiagnostic(
                code="unsupported-family-pair",
                consumer=request.consumer,
                family_pair=request.normalized_family_pair,
                message="spline-to-spline solver did not converge within declared tolerance sampling budget",
            ),
        )
        result = normalize_surface_intersection_result(request, quality="unsupported")
    else:
        curve = SurfaceIntersectionCurveRecord(
            curve_id="spline-spline-curve-0",
            kind="sampled",
            points_3d=tuple(item[0] for item in accepted),
            first_parameters=tuple(item[1] for item in accepted),
            second_parameters=tuple(item[2] for item in accepted),
        )
        result = normalize_surface_intersection_result(
            request,
            curves=(curve,),
            max_residual=iteration.max_residual,
            quality="within-tolerance",
        )
    return (
        result,
        SurfaceSplineSplineResidualReport(
            solver_id=dispatch.solver.solver_id,
            iterations=(iteration,),
            diagnostics=diagnostics,
        ),
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
    declared_tolerance_pairs = {
        ("bspline", "planar"): "analytic-spline-declared-tolerance",
        ("bspline", "bspline"): "spline-spline-declared-tolerance",
        ("bspline", "nurbs"): "spline-spline-declared-tolerance",
        ("bspline", "revolution"): "analytic-spline-declared-tolerance",
        ("bspline", "ruled"): "analytic-spline-declared-tolerance",
        ("nurbs", "nurbs"): "spline-spline-declared-tolerance",
        ("nurbs", "planar"): "analytic-spline-declared-tolerance",
        ("nurbs", "revolution"): "analytic-spline-declared-tolerance",
        ("nurbs", "ruled"): "analytic-spline-declared-tolerance",
    }
    adapter_pairs = {
        tuple(sorted((family, "subdivision"))): "subdivision-adapter-declared-tolerance"
        for family in _promoted_surface_family_names()
    }
    records: list[SurfaceIntersectionSolverRegistryRecord] = []
    families = _promoted_surface_family_names()
    for index, first_family in enumerate(families):
        for second_family in families[index:]:
            pair = tuple(sorted((first_family, second_family)))
            solver_id = exact_pairs.get(pair)
            declared_solver_id = declared_tolerance_pairs.get(pair)
            adapter_solver_id = adapter_pairs.get(pair)
            if solver_id is None:
                if declared_solver_id is not None:
                    records.append(
                        SurfaceIntersectionSolverRegistryRecord(
                            first_family=pair[0],
                            second_family=pair[1],
                            solver_id=declared_solver_id,
                            support_state="declared-tolerance",
                            operations=("classification", "csg", "seam"),
                        )
                    )
                    continue
                if adapter_solver_id is not None:
                    records.append(
                        SurfaceIntersectionSolverRegistryRecord(
                            first_family=pair[0],
                            second_family=pair[1],
                            solver_id=adapter_solver_id,
                            support_state="adapter",
                            operations=("classification", "csg", "seam"),
                        )
                    )
                    continue
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
    "DEFAULT_SURFACE_SUBDIVISION_INTERSECTION_BUDGET",
    "SURFACE_INTERSECTION_SOLVER_REGISTRY",
    "SurfaceAnalyticSplineResidualReport",
    "SurfaceAnalyticSplineSolverIterationRecord",
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
    "SurfaceSplineSplineResidualReport",
    "SurfaceSplineSplineSolverIterationRecord",
    "SurfaceSubdivisionIntersectionAdapterReport",
    "SurfaceSubdivisionIntersectionBudget",
    "SurfaceSubdivisionRefinedContourRecord",
    "assert_surface_intersection_solver_registry_complete",
    "build_surface_intersection_solver_registry",
    "check_subdivision_intersection_budget",
    "classify_surface_intersection_degeneracy",
    "lookup_surface_intersection_solver",
    "make_surface_intersection_request",
    "normalize_surface_intersection_result",
    "solve_analytic_spline_surface_intersection",
    "solve_spline_spline_surface_intersection",
    "solve_subdivision_surface_intersection_adapter",
]
