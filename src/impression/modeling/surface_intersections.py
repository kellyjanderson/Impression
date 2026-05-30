from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .surface import (
    PATCH_FAMILY_CAPABILITY_MATRIX,
    BSplineSurfacePatch,
    ImplicitFieldSafetyPolicy,
    ImplicitFieldValidationDiagnostic,
    ImplicitSurfacePatch,
    NURBSSurfacePatch,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SubdivisionSurfacePatch,
    SweepSurfacePatch,
    SurfacePatch,
    assess_implicit_field_security,
    evaluate_path_frame,
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
class SurfaceSweepCSGEventSeedRecord:
    """Deterministic event seed emitted by a sweep patch for CSG routing."""

    seed_id: str
    kind: Literal["path-endpoint", "profile-vertex", "frame-sample"]
    parameter: tuple[float, float]
    point: tuple[float, float, float]
    source_reference: str | None = None

    def __post_init__(self) -> None:
        seed_id = str(self.seed_id).strip()
        if not seed_id:
            raise ValueError("SurfaceSweepCSGEventSeedRecord.seed_id must be non-empty.")
        parameter = tuple(float(value) for value in self.parameter)
        point = tuple(float(value) for value in self.point)
        if len(parameter) != 2 or any(not np.isfinite(value) for value in parameter):
            raise ValueError("SurfaceSweepCSGEventSeedRecord.parameter must be a finite UV pair.")
        if len(point) != 3 or any(not np.isfinite(value) for value in point):
            raise ValueError("SurfaceSweepCSGEventSeedRecord.point must be a finite 3D point.")
        object.__setattr__(self, "seed_id", seed_id)
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "point", point)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "seed_id": self.seed_id,
            "kind": self.kind,
            "parameter": self.parameter,
            "point": self.point,
            "source_reference": self.source_reference,
        }


@dataclass(frozen=True)
class SurfaceSweepCSGFrameEventDiagnostic:
    """Frame or authoring diagnostic emitted while adapting a sweep patch for CSG."""

    code: Literal["frame-singularity", "repeated-event", "invalid-sweep"]
    message: str
    parameter: tuple[float, float] | None = None
    blocking: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "blocking": self.blocking,
            "code": self.code,
            "message": self.message,
            "parameter": self.parameter,
        }


@dataclass(frozen=True)
class SurfaceSweepCSGEvaluatorAdapter:
    """CSG-ready wrapper around a sweep patch evaluator and deterministic event seeds."""

    patch: SweepSurfacePatch
    event_seeds: tuple[SurfaceSweepCSGEventSeedRecord, ...]
    diagnostics: tuple[SurfaceSweepCSGFrameEventDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return bool(self.event_seeds) and not any(diagnostic.blocking for diagnostic in self.diagnostics)

    def point_at(self, u: float, v: float) -> np.ndarray:
        return self.patch.point_at(float(u), float(v))

    def derivatives_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        return self.patch.derivatives_at(float(u), float(v))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family": self.patch.family,
            "supported": self.supported,
            "profile_reference": self.patch.profile_reference,
            "path_reference": self.patch.path_reference,
            "event_seeds": [seed.canonical_payload() for seed in self.event_seeds],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


def collect_sweep_csg_event_seeds(
    patch: SweepSurfacePatch,
    *,
    frame_sample_count: int = 5,
) -> tuple[SurfaceSweepCSGEventSeedRecord, ...]:
    """Collect deterministic path/profile/frame event seeds for a sweep CSG route."""

    if not isinstance(patch, SweepSurfacePatch):
        raise TypeError("collect_sweep_csg_event_seeds requires a SweepSurfacePatch.")
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    seeds: list[SurfaceSweepCSGEventSeedRecord] = []
    for index, u in enumerate((u0, u1)):
        for v in (v0, v1):
            point = _safe_sweep_event_point(patch, float(u), float(v))
            seeds.append(
                SurfaceSweepCSGEventSeedRecord(
                    seed_id=f"path-endpoint-{index}-{len(seeds)}",
                    kind="path-endpoint",
                    parameter=(float(u), float(v)),
                    point=tuple(float(component) for component in point),
                    source_reference=patch.path_reference,
                )
            )
    for index, profile_point in enumerate(patch.profile_points_uv):
        v_norm = 0.0 if len(patch.profile_points_uv) == 1 else index / max(1, len(patch.profile_points_uv) - 1)
        v = float(v0 + (v1 - v0) * v_norm)
        u = float(u0)
        point = _safe_sweep_event_point(patch, u, v)
        seeds.append(
            SurfaceSweepCSGEventSeedRecord(
                seed_id=f"profile-vertex-{index}",
                kind="profile-vertex",
                parameter=(u, v),
                point=tuple(float(component) for component in point),
                source_reference=patch.profile_reference,
            )
        )
    for index, u_norm in enumerate(np.linspace(0.0, 1.0, max(2, int(frame_sample_count)))):
        u = float(u0 + (u1 - u0) * float(u_norm))
        v = float((v0 + v1) * 0.5)
        point = _safe_sweep_event_point(patch, u, v)
        seeds.append(
            SurfaceSweepCSGEventSeedRecord(
                seed_id=f"frame-sample-{index}",
                kind="frame-sample",
                parameter=(u, v),
                point=tuple(float(component) for component in point),
                source_reference=patch.path_reference,
            )
        )
    return tuple(sorted(seeds, key=lambda seed: (seed.parameter, seed.kind, seed.seed_id)))


def _safe_sweep_event_point(patch: SweepSurfacePatch, u: float, v: float) -> tuple[float, float, float]:
    try:
        point = patch.point_at(float(u), float(v))
    except ValueError:
        point = np.zeros(3, dtype=float)
    return tuple(float(component) for component in point)


def make_sweep_csg_evaluator_adapter(
    patch: SweepSurfacePatch,
    *,
    frame_sample_count: int = 5,
) -> SurfaceSweepCSGEvaluatorAdapter:
    """Build a CSG-ready sweep evaluator adapter with frame-event diagnostics."""

    if not isinstance(patch, SweepSurfacePatch):
        raise TypeError("make_sweep_csg_evaluator_adapter requires a SweepSurfacePatch.")
    diagnostics: list[SurfaceSweepCSGFrameEventDiagnostic] = []
    for u_norm in np.linspace(0.0, 1.0, max(2, int(frame_sample_count))):
        frame = evaluate_path_frame(patch.path, float(u_norm), patch.frame_policy)
        for diagnostic in frame.diagnostics:
            diagnostics.append(
                SurfaceSweepCSGFrameEventDiagnostic(
                    code="frame-singularity",
                    message=diagnostic.message,
                    parameter=(float(u_norm), 0.0),
                )
            )
    seeds = collect_sweep_csg_event_seeds(patch, frame_sample_count=frame_sample_count)
    seen_parameters: set[tuple[float, float]] = set()
    for seed in seeds:
        if seed.parameter in seen_parameters:
            diagnostics.append(
                SurfaceSweepCSGFrameEventDiagnostic(
                    code="repeated-event",
                    message="Sweep CSG event seed repeats an authored parameter location.",
                    parameter=seed.parameter,
                    blocking=False,
                )
            )
        seen_parameters.add(seed.parameter)
    return SurfaceSweepCSGEvaluatorAdapter(
        patch=patch,
        event_seeds=seeds,
        diagnostics=tuple(diagnostics),
    )


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
        "unsafe-implicit-field",
        "non-convergent",
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
class SurfaceSweepPairSolverIterationRecord:
    """Bounded sampling witness for a sweep-participating surface intersection."""

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
class SurfaceSweepPairResidualReport:
    """Residual, event, and diagnostic report for sweep CSG intersection routes."""

    solver_id: str
    iterations: tuple[SurfaceSweepPairSolverIterationRecord, ...]
    sweep_adapters: tuple[SurfaceSweepCSGEvaluatorAdapter, ...] = ()
    diagnostics: tuple[SurfaceIntersectionSupportDiagnostic | SurfaceSweepCSGFrameEventDiagnostic, ...] = ()

    @property
    def converged(self) -> bool:
        return bool(self.iterations) and self.iterations[-1].converged and not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "iterations": [iteration.canonical_payload() for iteration in self.iterations],
            "solver_id": self.solver_id,
            "sweep_adapters": [adapter.canonical_payload() for adapter in self.sweep_adapters],
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
class SurfaceImplicitIntersectionBudget:
    """Strict budget gate for implicit surface intersection execution."""

    max_cells: int = 4096
    max_depth: int = 8
    max_iterations: int = 96

    def __post_init__(self) -> None:
        for name in ("max_cells", "max_depth", "max_iterations"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"SurfaceImplicitIntersectionBudget.{name} must be positive.")
            object.__setattr__(self, name, value)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "max_cells": self.max_cells,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
        }


DEFAULT_SURFACE_IMPLICIT_INTERSECTION_BUDGET = SurfaceImplicitIntersectionBudget()


@dataclass(frozen=True)
class SurfaceImplicitIntersectionSafetyDecision:
    """Serializable safety and budget decision for an implicit intersection request."""

    request: SurfaceIntersectionRequest
    budget: SurfaceImplicitIntersectionBudget
    implicit_diagnostics: tuple[ImplicitFieldValidationDiagnostic, ...] = ()
    diagnostics: tuple[SurfaceIntersectionSupportDiagnostic, ...] = ()

    @property
    def executable(self) -> bool:
        return not self.diagnostics and all(diagnostic.safe for diagnostic in self.implicit_diagnostics)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "budget": self.budget.canonical_payload(),
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "executable": self.executable,
            "implicit_diagnostics": [diagnostic.canonical_payload() for diagnostic in self.implicit_diagnostics],
            "request": self.request.canonical_payload(),
        }


@dataclass(frozen=True)
class SurfaceImplicitContourRecord:
    """Surface-native implicit contour extracted under a bounded adapter policy."""

    contour_id: str
    curve: SurfaceIntersectionCurveRecord
    evaluated_cell_count: int
    max_abs_field_residual: float

    def __post_init__(self) -> None:
        contour_id = str(self.contour_id).strip()
        evaluated_cell_count = int(self.evaluated_cell_count)
        max_abs_field_residual = float(self.max_abs_field_residual)
        if not contour_id:
            raise ValueError("SurfaceImplicitContourRecord.contour_id must be non-empty.")
        if not isinstance(self.curve, SurfaceIntersectionCurveRecord):
            raise TypeError("SurfaceImplicitContourRecord.curve must be a SurfaceIntersectionCurveRecord.")
        if evaluated_cell_count <= 0:
            raise ValueError("SurfaceImplicitContourRecord.evaluated_cell_count must be positive.")
        if not np.isfinite(max_abs_field_residual) or max_abs_field_residual < 0.0:
            raise ValueError("SurfaceImplicitContourRecord.max_abs_field_residual must be finite and non-negative.")
        object.__setattr__(self, "contour_id", contour_id)
        object.__setattr__(self, "evaluated_cell_count", evaluated_cell_count)
        object.__setattr__(self, "max_abs_field_residual", max_abs_field_residual)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "contour_id": self.contour_id,
            "curve": self.curve.canonical_payload(),
            "evaluated_cell_count": self.evaluated_cell_count,
            "max_abs_field_residual": self.max_abs_field_residual,
        }


@dataclass(frozen=True)
class SurfaceImplicitContourExtractionTrace:
    """Execution trace for a bounded implicit contour extraction adapter."""

    solver_id: str
    safety_decision: SurfaceImplicitIntersectionSafetyDecision
    contours: tuple[SurfaceImplicitContourRecord, ...] = ()
    diagnostics: tuple[SurfaceIntersectionSupportDiagnostic, ...] = ()

    @property
    def converged(self) -> bool:
        return self.safety_decision.executable and bool(self.contours) and not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "contours": [contour.canonical_payload() for contour in self.contours],
            "converged": self.converged,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "safety_decision": self.safety_decision.canonical_payload(),
            "solver_id": self.solver_id,
        }


@dataclass(frozen=True)
class SurfaceImplicitNonConvergenceDiagnostic:
    """Implicit result diagnostic that distinguishes refusal and residual failure causes."""

    reason: Literal["budget-exhausted", "unsafe-implicit-field", "residual-failure", "unsupported-family-pair"]
    message: str
    residual: float | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "message": self.message,
            "reason": self.reason,
            "residual": self.residual,
        }


@dataclass(frozen=True)
class SurfaceImplicitResidualReport:
    """Residual report for declared-tolerance implicit contour extraction output."""

    contour_count: int
    evaluated_cell_count: int
    max_abs_field_residual: float
    tolerance: float
    non_convergence: tuple[SurfaceImplicitNonConvergenceDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        contour_count = int(self.contour_count)
        evaluated_cell_count = int(self.evaluated_cell_count)
        max_abs_field_residual = float(self.max_abs_field_residual)
        tolerance = float(self.tolerance)
        if contour_count < 0:
            raise ValueError("SurfaceImplicitResidualReport.contour_count must be non-negative.")
        if evaluated_cell_count < 0:
            raise ValueError("SurfaceImplicitResidualReport.evaluated_cell_count must be non-negative.")
        if not np.isfinite(max_abs_field_residual) or max_abs_field_residual < 0.0:
            raise ValueError("SurfaceImplicitResidualReport.max_abs_field_residual must be finite and non-negative.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("SurfaceImplicitResidualReport.tolerance must be positive and finite.")
        object.__setattr__(self, "contour_count", contour_count)
        object.__setattr__(self, "evaluated_cell_count", evaluated_cell_count)
        object.__setattr__(self, "max_abs_field_residual", max_abs_field_residual)
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def within_tolerance(self) -> bool:
        return self.contour_count > 0 and self.max_abs_field_residual <= self.tolerance and not self.non_convergence

    def canonical_payload(self) -> dict[str, object]:
        return {
            "contour_count": self.contour_count,
            "evaluated_cell_count": self.evaluated_cell_count,
            "max_abs_field_residual": self.max_abs_field_residual,
            "non_convergence": [diagnostic.canonical_payload() for diagnostic in self.non_convergence],
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
        }


@dataclass(frozen=True)
class SurfaceImplicitResultClassificationRecord:
    """Declared-tolerance classification for implicit intersection results."""

    classification: Literal["declared-tolerance", "non-convergent", "refused"]
    residual_report: SurfaceImplicitResidualReport
    result_quality: Literal["within-tolerance", "unsupported"]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "residual_report": self.residual_report.canonical_payload(),
            "result_quality": self.result_quality,
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


def _surface_intersection_is_implicit_family(patch: SurfacePatch) -> bool:
    return isinstance(patch, ImplicitSurfacePatch)


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


def check_implicit_intersection_budget(
    request: SurfaceIntersectionRequest,
    *,
    budget: SurfaceImplicitIntersectionBudget = DEFAULT_SURFACE_IMPLICIT_INTERSECTION_BUDGET,
    cell_count: int,
    depth: int,
    iteration_count: int,
) -> tuple[SurfaceIntersectionSupportDiagnostic, ...]:
    """Return deterministic diagnostics when implicit intersection work exceeds budget."""

    diagnostics: list[SurfaceIntersectionSupportDiagnostic] = []
    cell_count = int(cell_count)
    depth = int(depth)
    iteration_count = int(iteration_count)
    if cell_count > budget.max_cells:
        diagnostics.append(
            SurfaceIntersectionSupportDiagnostic(
                code="budget-exhausted",
                consumer=request.consumer,
                family_pair=request.normalized_family_pair,
                message=f"implicit intersection cell budget exhausted: {cell_count} cells exceeds {budget.max_cells}",
            )
        )
    if depth > budget.max_depth:
        diagnostics.append(
            SurfaceIntersectionSupportDiagnostic(
                code="budget-exhausted",
                consumer=request.consumer,
                family_pair=request.normalized_family_pair,
                message=f"implicit intersection depth budget exhausted: {depth} exceeds {budget.max_depth}",
            )
        )
    if iteration_count > budget.max_iterations:
        diagnostics.append(
            SurfaceIntersectionSupportDiagnostic(
                code="budget-exhausted",
                consumer=request.consumer,
                family_pair=request.normalized_family_pair,
                message=(
                    "implicit intersection iteration budget exhausted: "
                    f"{iteration_count} iterations exceeds {budget.max_iterations}"
                ),
            )
        )
    return tuple(diagnostics)


def check_implicit_intersection_safety(
    request: SurfaceIntersectionRequest,
    *,
    budget: SurfaceImplicitIntersectionBudget = DEFAULT_SURFACE_IMPLICIT_INTERSECTION_BUDGET,
    field_safety_policy: ImplicitFieldSafetyPolicy | None = None,
    cell_count: int = 1,
    depth: int = 1,
    iteration_count: int = 1,
) -> SurfaceImplicitIntersectionSafetyDecision:
    """Gate implicit intersection execution before any solver samples field values."""

    implicit_patches = tuple(patch for patch in (request.first_patch, request.second_patch) if isinstance(patch, ImplicitSurfacePatch))
    diagnostics: list[SurfaceIntersectionSupportDiagnostic] = []
    implicit_diagnostics: list[ImplicitFieldValidationDiagnostic] = []
    if not implicit_patches:
        diagnostics.append(
            SurfaceIntersectionSupportDiagnostic(
                code="unsupported-family-pair",
                consumer=request.consumer,
                family_pair=request.normalized_family_pair,
                message="implicit intersection safety gate requires at least one implicit surface",
            )
        )
    for patch in implicit_patches:
        implicit_diagnostic = assess_implicit_field_security(patch.field, policy=field_safety_policy)
        implicit_diagnostics.append(implicit_diagnostic)
        if not implicit_diagnostic.safe:
            diagnostics.append(
                SurfaceIntersectionSupportDiagnostic(
                    code="unsafe-implicit-field",
                    consumer=request.consumer,
                    family_pair=request.normalized_family_pair,
                    message=f"implicit intersection refused unsafe field: {implicit_diagnostic.reason}",
                )
            )
    diagnostics.extend(
        check_implicit_intersection_budget(
            request,
            budget=budget,
            cell_count=cell_count,
            depth=depth,
            iteration_count=iteration_count,
        )
    )
    return SurfaceImplicitIntersectionSafetyDecision(
        request=request,
        budget=budget,
        implicit_diagnostics=tuple(implicit_diagnostics),
        diagnostics=tuple(diagnostics),
    )


def _implicit_field_residuals(
    implicit_patches: Sequence[ImplicitSurfacePatch],
    point: Sequence[float],
) -> tuple[float, ...]:
    return tuple(abs(float(patch.field_value_at(point).value)) for patch in implicit_patches)


def _sample_implicit_bounds_points(
    patch: ImplicitSurfacePatch,
    sample_count: int,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float]], ...]:
    xmin, xmax, ymin, ymax, zmin, zmax = patch.bounds_estimate()
    xs = np.linspace(xmin, xmax, sample_count)
    ys = np.linspace(ymin, ymax, sample_count)
    zs = np.linspace(zmin, zmax, sample_count)
    return tuple(
        ((float(x), float(y), float(z)), (float(index), 0.0))
        for index, (x, y, z) in enumerate((x, y, z) for z in zs for y in ys for x in xs)
    )


def solve_implicit_surface_intersection_adapter(
    request: SurfaceIntersectionRequest,
    *,
    budget: SurfaceImplicitIntersectionBudget = DEFAULT_SURFACE_IMPLICIT_INTERSECTION_BUDGET,
    field_safety_policy: ImplicitFieldSafetyPolicy | None = None,
    sample_count: int = 9,
) -> tuple[SurfaceIntersectionResultRecord, SurfaceImplicitContourExtractionTrace]:
    """Extract bounded implicit contour records after explicit safety approval."""

    dispatch = lookup_surface_intersection_solver(request)
    if not dispatch.supported or dispatch.solver is None or dispatch.solver.solver_id != "implicit-contour-declared-tolerance":
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer=request.consumer,
            family_pair=request.normalized_family_pair,
            message="implicit contour adapter requires an adapter-supported implicit registry dispatch",
        )
        safety_decision = check_implicit_intersection_safety(request, budget=budget, field_safety_policy=field_safety_policy)
        return (
            normalize_surface_intersection_result(request, quality="unsupported"),
            SurfaceImplicitContourExtractionTrace(
                solver_id="implicit-contour-declared-tolerance",
                safety_decision=safety_decision,
                diagnostics=(diagnostic,),
            ),
        )

    sample_count = max(3, int(sample_count))
    implicit_patches = tuple(patch for patch in (request.first_patch, request.second_patch) if isinstance(patch, ImplicitSurfacePatch))
    if len(implicit_patches) == 2:
        cell_count = sample_count * sample_count * sample_count
        depth = sample_count
    else:
        cell_count = sample_count * sample_count
        depth = 1
    safety_decision = check_implicit_intersection_safety(
        request,
        budget=budget,
        field_safety_policy=field_safety_policy,
        cell_count=cell_count,
        depth=depth,
        iteration_count=sample_count,
    )
    if not safety_decision.executable:
        return (
            normalize_surface_intersection_result(request, quality="unsupported"),
            SurfaceImplicitContourExtractionTrace(
                solver_id=dispatch.solver.solver_id,
                safety_decision=safety_decision,
                diagnostics=safety_decision.diagnostics,
            ),
        )

    non_implicit_patches = tuple(patch for patch in (request.first_patch, request.second_patch) if not isinstance(patch, ImplicitSurfacePatch))
    if non_implicit_patches:
        sampling_patch = non_implicit_patches[0]
        samples = _sample_surface_points(sampling_patch, sample_count)
    else:
        samples = _sample_implicit_bounds_points(implicit_patches[0], sample_count)

    threshold = max(request.tolerance_policy.position_tolerance * 10.0, request.tolerance_policy.degeneracy_tolerance)
    accepted: list[tuple[tuple[float, float, float], tuple[float, float], float]] = []
    for point, uv in samples:
        residuals = _implicit_field_residuals(implicit_patches, point)
        max_abs_residual = max(residuals, default=float("inf"))
        if max_abs_residual <= threshold:
            accepted.append((point, uv, max_abs_residual))

    accepted = sorted(set(accepted), key=lambda item: (item[1], item[0]))
    max_residual = max((item[2] for item in accepted), default=float("inf"))
    if len(accepted) < 2 or max_residual > threshold:
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer=request.consumer,
            family_pair=request.normalized_family_pair,
            message="implicit contour adapter did not extract a contour within declared tolerance sampling budget",
        )
        return (
            normalize_surface_intersection_result(
                request,
                max_residual=max_residual if np.isfinite(max_residual) else 0.0,
                quality="unsupported",
            ),
            SurfaceImplicitContourExtractionTrace(
                solver_id=dispatch.solver.solver_id,
                safety_decision=safety_decision,
                diagnostics=(diagnostic,),
            ),
        )

    first_parameters: tuple[tuple[float, float], ...] = ()
    second_parameters: tuple[tuple[float, float], ...] = ()
    if not isinstance(request.first_patch, ImplicitSurfacePatch):
        first_parameters = tuple(item[1] for item in accepted)
    if not isinstance(request.second_patch, ImplicitSurfacePatch):
        second_parameters = tuple(item[1] for item in accepted)
    curve = SurfaceIntersectionCurveRecord(
        curve_id="implicit-contour-0",
        kind="implicit-contour",
        points_3d=tuple(item[0] for item in accepted),
        first_parameters=first_parameters,
        second_parameters=second_parameters,
    )
    contour = SurfaceImplicitContourRecord(
        contour_id="implicit-contour-0",
        curve=curve,
        evaluated_cell_count=cell_count,
        max_abs_field_residual=max_residual,
    )
    result = normalize_surface_intersection_result(
        request,
        curves=(curve,),
        max_residual=max_residual,
        quality="within-tolerance",
    )
    return (
        result,
        SurfaceImplicitContourExtractionTrace(
            solver_id=dispatch.solver.solver_id,
            safety_decision=safety_decision,
            contours=(contour,),
        ),
    )


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


def build_implicit_non_convergence_diagnostic(
    trace: SurfaceImplicitContourExtractionTrace,
    *,
    residual: float | None = None,
) -> SurfaceImplicitNonConvergenceDiagnostic:
    """Build a stable implicit non-convergence diagnostic from extraction trace state."""

    reasons = {diagnostic.code for diagnostic in trace.diagnostics + trace.safety_decision.diagnostics}
    if "budget-exhausted" in reasons:
        return SurfaceImplicitNonConvergenceDiagnostic(
            reason="budget-exhausted",
            message="implicit contour extraction did not execute because the intersection budget was exhausted",
            residual=residual,
        )
    if "unsafe-implicit-field" in reasons:
        return SurfaceImplicitNonConvergenceDiagnostic(
            reason="unsafe-implicit-field",
            message="implicit contour extraction did not execute because an implicit field failed safety policy",
            residual=residual,
        )
    if "unsupported-family-pair" in reasons:
        return SurfaceImplicitNonConvergenceDiagnostic(
            reason="unsupported-family-pair",
            message="implicit contour extraction did not execute because the family pair is unsupported",
            residual=residual,
        )
    return SurfaceImplicitNonConvergenceDiagnostic(
        reason="residual-failure",
        message="implicit contour extraction residual exceeded declared tolerance or produced no contours",
        residual=residual,
    )


def classify_implicit_intersection_residuals(
    trace: SurfaceImplicitContourExtractionTrace,
    *,
    tolerance_policy: SurfaceIntersectionTolerancePolicy | None = None,
) -> SurfaceImplicitResultClassificationRecord:
    """Classify implicit contour trace residuals into declared-tolerance result state."""

    policy = DEFAULT_SURFACE_INTERSECTION_TOLERANCE_POLICY if tolerance_policy is None else tolerance_policy
    tolerance = max(policy.position_tolerance * 10.0, policy.degeneracy_tolerance)
    contour_count = len(trace.contours)
    evaluated_cell_count = sum(contour.evaluated_cell_count for contour in trace.contours)
    max_residual = max((contour.max_abs_field_residual for contour in trace.contours), default=0.0)
    non_convergence: tuple[SurfaceImplicitNonConvergenceDiagnostic, ...] = ()
    if not trace.safety_decision.executable or trace.diagnostics:
        non_convergence = (build_implicit_non_convergence_diagnostic(trace, residual=max_residual),)
        classification: Literal["declared-tolerance", "non-convergent", "refused"] = "refused"
    elif contour_count == 0 or max_residual > tolerance:
        non_convergence = (build_implicit_non_convergence_diagnostic(trace, residual=max_residual),)
        classification = "non-convergent"
    else:
        classification = "declared-tolerance"
    residual_report = SurfaceImplicitResidualReport(
        contour_count=contour_count,
        evaluated_cell_count=evaluated_cell_count,
        max_abs_field_residual=max_residual,
        tolerance=tolerance,
        non_convergence=non_convergence,
    )
    return SurfaceImplicitResultClassificationRecord(
        classification=classification,
        residual_report=residual_report,
        result_quality="within-tolerance" if residual_report.within_tolerance else "unsupported",
    )


def assemble_implicit_intersection_result(
    request: SurfaceIntersectionRequest,
    trace: SurfaceImplicitContourExtractionTrace,
    *,
    tolerance_policy: SurfaceIntersectionTolerancePolicy | None = None,
) -> tuple[SurfaceIntersectionResultRecord, SurfaceImplicitResultClassificationRecord]:
    """Assemble normalized intersection result records from implicit contour classification."""

    classification = classify_implicit_intersection_residuals(trace, tolerance_policy=tolerance_policy)
    if classification.residual_report.within_tolerance:
        result = normalize_surface_intersection_result(
            request,
            curves=tuple(contour.curve for contour in trace.contours),
            max_residual=classification.residual_report.max_abs_field_residual,
            quality="within-tolerance",
        )
    else:
        degeneracies = tuple(
            SurfaceIntersectionDegeneracyRecord(
                code="high-residual",
                message=diagnostic.message,
                residual=diagnostic.residual,
            )
            for diagnostic in classification.residual_report.non_convergence
        )
        result = normalize_surface_intersection_result(
            request,
            max_residual=classification.residual_report.max_abs_field_residual,
            quality="unsupported",
            degeneracies=degeneracies,
        )
    return result, classification


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


def solve_sweep_surface_intersection_adapter(
    request: SurfaceIntersectionRequest,
    *,
    sample_count: int = 13,
) -> tuple[SurfaceIntersectionResultRecord, SurfaceSweepPairResidualReport]:
    """Solve sweep-participating surface pairs by bounded declared-tolerance sampling."""

    dispatch = lookup_surface_intersection_solver(request)
    if not dispatch.supported or dispatch.solver is None or dispatch.solver.solver_id != "sweep-pair-declared-tolerance":
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer=request.consumer,
            family_pair=request.normalized_family_pair,
            message="sweep pair solver requires a declared-tolerance sweep registry dispatch",
        )
        return (
            normalize_surface_intersection_result(request, quality="unsupported"),
            SurfaceSweepPairResidualReport(
                solver_id="sweep-pair-declared-tolerance",
                iterations=(),
                diagnostics=(diagnostic,),
            ),
        )
    if not (isinstance(request.first_patch, SweepSurfacePatch) or isinstance(request.second_patch, SweepSurfacePatch)):
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer=request.consumer,
            family_pair=request.normalized_family_pair,
            message="sweep pair solver requires at least one sweep surface",
        )
        return (
            normalize_surface_intersection_result(request, quality="unsupported"),
            SurfaceSweepPairResidualReport(
                solver_id=dispatch.solver.solver_id,
                iterations=(),
                diagnostics=(diagnostic,),
            ),
        )

    sweep_adapters = tuple(
        make_sweep_csg_evaluator_adapter(patch)
        for patch in (request.first_patch, request.second_patch)
        if isinstance(patch, SweepSurfacePatch)
    )
    adapter_diagnostics = tuple(diagnostic for adapter in sweep_adapters for diagnostic in adapter.diagnostics if diagnostic.blocking)
    if adapter_diagnostics:
        return (
            normalize_surface_intersection_result(request, quality="unsupported"),
            SurfaceSweepPairResidualReport(
                solver_id=dispatch.solver.solver_id,
                iterations=(),
                sweep_adapters=sweep_adapters,
                diagnostics=adapter_diagnostics,
            ),
        )

    sample_count = max(3, int(sample_count))
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
    converged = len(accepted) >= 2 and max_residual <= threshold
    iteration = SurfaceSweepPairSolverIterationRecord(
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
                message="sweep pair solver did not converge within declared tolerance sampling budget",
            ),
        )
        result = normalize_surface_intersection_result(request, quality="unsupported")
    else:
        curve = SurfaceIntersectionCurveRecord(
            curve_id="sweep-pair-curve-0",
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
        SurfaceSweepPairResidualReport(
            solver_id=dispatch.solver.solver_id,
            iterations=(iteration,),
            sweep_adapters=sweep_adapters,
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
        ("planar", "sweep"): "sweep-pair-declared-tolerance",
        ("bspline", "sweep"): "sweep-pair-declared-tolerance",
        ("nurbs", "sweep"): "sweep-pair-declared-tolerance",
        ("revolution", "sweep"): "sweep-pair-declared-tolerance",
        ("ruled", "sweep"): "sweep-pair-declared-tolerance",
        ("subdivision", "sweep"): "sweep-pair-declared-tolerance",
        ("sweep", "sweep"): "sweep-pair-declared-tolerance",
    }
    adapter_pairs = {
        tuple(sorted((family, "subdivision"))): "subdivision-adapter-declared-tolerance"
        for family in _promoted_surface_family_names()
    }
    adapter_pairs.update(
        {
            tuple(sorted((family, "implicit"))): "implicit-contour-declared-tolerance"
            for family in _promoted_surface_family_names()
        }
    )
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
    "DEFAULT_SURFACE_IMPLICIT_INTERSECTION_BUDGET",
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
    "SurfaceImplicitContourExtractionTrace",
    "SurfaceImplicitContourRecord",
    "SurfaceImplicitIntersectionBudget",
    "SurfaceImplicitIntersectionSafetyDecision",
    "SurfaceImplicitNonConvergenceDiagnostic",
    "SurfaceImplicitResidualReport",
    "SurfaceImplicitResultClassificationRecord",
    "SurfaceSplineSplineResidualReport",
    "SurfaceSplineSplineSolverIterationRecord",
    "SurfaceSubdivisionIntersectionAdapterReport",
    "SurfaceSubdivisionIntersectionBudget",
    "SurfaceSubdivisionRefinedContourRecord",
    "assert_surface_intersection_solver_registry_complete",
    "assemble_implicit_intersection_result",
    "build_surface_intersection_solver_registry",
    "build_implicit_non_convergence_diagnostic",
    "check_implicit_intersection_budget",
    "check_implicit_intersection_safety",
    "check_subdivision_intersection_budget",
    "classify_surface_intersection_degeneracy",
    "classify_implicit_intersection_residuals",
    "lookup_surface_intersection_solver",
    "make_surface_intersection_request",
    "normalize_surface_intersection_result",
    "solve_analytic_spline_surface_intersection",
    "solve_implicit_surface_intersection_adapter",
    "solve_spline_spline_surface_intersection",
    "solve_subdivision_surface_intersection_adapter",
]
