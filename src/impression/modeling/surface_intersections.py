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
    "SurfaceIntersectionRequest",
    "SurfaceIntersectionSolverDispatchRecord",
    "SurfaceIntersectionSolverRegistryRecord",
    "SurfaceIntersectionSupportDiagnostic",
    "SurfaceIntersectionSupportState",
    "SurfaceIntersectionTolerancePolicy",
    "assert_surface_intersection_solver_registry_complete",
    "build_surface_intersection_solver_registry",
    "lookup_surface_intersection_solver",
    "make_surface_intersection_request",
]
