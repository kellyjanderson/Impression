from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Iterable, Literal, Mapping, Sequence, Union

import warnings

import numpy as np

from impression.mesh import Mesh, analyze_mesh
from impression.modeling.group import MeshGroup

from ._color import get_mesh_color, set_mesh_color
from ._legacy_mesh_deprecation import warn_mesh_primary_api
from .surface import (
    PATCH_FAMILY_CAPABILITY_MATRIX,
    SurfaceBoundaryRef,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SurfaceBody,
    SurfaceSeam,
    TrimLoop,
    make_surface_body,
    make_surface_shell,
)

BooleanBackend = Literal["manifold", "surface"]
SurfaceBooleanOperation = Literal["union", "difference", "intersection"]
SURFACE_BOOLEAN_OPERATIONS: tuple[SurfaceBooleanOperation, ...] = ("union", "difference", "intersection")
SurfaceBooleanStatus = Literal["succeeded", "invalid", "unsupported"]
SurfaceBooleanClassification = Literal["open", "closed", "empty"]
SurfaceBooleanBodyRelation = Literal["disjoint", "touching", "overlap", "containment", "equal"]
SurfaceBooleanPatchRelation = Literal["inside", "outside", "on"]
SurfaceBooleanSplitRole = Literal["survive", "cut_cap", "discard"]
SurfaceBooleanUnsupportedPhase = Literal["operand-family-eligibility", "intersection-kernel"]
SurfaceBooleanSupportState = Literal["exact", "declared-tolerance", "adapter", "unsupported", "not-yet-implemented"]
SurfaceCSGCurveKind = Literal["line", "arc", "conic", "sampled"]
SurfaceCSGToleranceDiagnosticCode = Literal[
    "invalid-tolerance",
    "degenerate-curve",
    "ambiguous-curve",
    "outside-domain",
]
SurfaceCSGPlanarRelation = Literal["crossing", "parallel", "coincident", "disjoint", "touching", "unsupported-linear"]
SurfaceCSGCallerCategory = Literal["public-api", "primitive", "feature", "compatibility"]


class BooleanOperationError(RuntimeError):
    """Raised when a boolean operation cannot produce a valid solid."""


class SurfaceBooleanEligibilityError(BooleanOperationError):
    """Raised when surfaced boolean inputs violate the current v1 contract."""


class SurfaceBooleanExecutionUnavailableError(BooleanOperationError):
    """Raised when a caller explicitly requires execution that is not implemented yet."""

    def __init__(self, operation: SurfaceBooleanOperation, operand_ids: tuple[str, ...]) -> None:
        self.operation = operation
        self.operand_ids = operand_ids
        super().__init__(
            f"Surface boolean {operation} execution is not implemented yet after canonical input preparation."
        )


@dataclass(frozen=True)
class SurfaceCSGCallerInventoryRecord:
    """One known authored route that depends on surface CSG readiness."""

    caller_id: str
    module: str
    category: SurfaceCSGCallerCategory
    operation: SurfaceBooleanOperation | None
    surface_route: str
    mesh_route: str | None = None
    explicit_mesh_route: bool = False

    def canonical_payload(self) -> dict[str, object]:
        return {
            "caller_id": self.caller_id,
            "category": self.category,
            "explicit_mesh_route": self.explicit_mesh_route,
            "mesh_route": self.mesh_route,
            "module": self.module,
            "operation": self.operation,
            "surface_route": self.surface_route,
        }


@dataclass(frozen=True)
class SurfaceCSGFeatureGateDiagnostic:
    """Shared diagnostic for primitive and feature surface CSG readiness."""

    caller_id: str
    operation: SurfaceBooleanOperation
    supported: bool
    reason: str
    operand_ids: tuple[str, ...] = ()
    boundary: str = "surface-boolean"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary": self.boundary,
            "caller_id": self.caller_id,
            "operand_ids": self.operand_ids,
            "operation": self.operation,
            "reason": self.reason,
            "supported": self.supported,
        }


@dataclass(frozen=True)
class SurfaceCSGPlanDiagnostic:
    """Diagnostic accumulated by a surface CSG operation plan before execution."""

    code: Literal["invalid-operand", "invalid-operand-count", "unsupported-family-pair"]
    message: str
    operation: SurfaceBooleanOperation
    operand_index: int | None = None
    left_family: str | None = None
    right_family: str | None = None
    phase: str | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "operation": self.operation,
            "operand_index": self.operand_index,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class SurfaceCSGPairDispatchRecord:
    """Planner dispatch decision for one operation and pair of patch families."""

    operation: SurfaceBooleanOperation
    left_operand_index: int
    right_operand_index: int
    left_family: str
    right_family: str
    supported: bool
    phase: str
    support_state: SurfaceBooleanSupportState
    required_future_capability: str | None = None

    @classmethod
    def from_support(
        cls,
        *,
        left_operand_index: int,
        right_operand_index: int,
        support: SurfaceBooleanFamilyPairSupport,
    ) -> "SurfaceCSGPairDispatchRecord":
        return cls(
            operation=support.operation,
            left_operand_index=left_operand_index,
            right_operand_index=right_operand_index,
            left_family=support.left_family,
            right_family=support.right_family,
            supported=support.supported,
            phase=support.phase,
            support_state=support.support_state,
            required_future_capability=support.required_future_capability,
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "left_operand_index": self.left_operand_index,
            "right_operand_index": self.right_operand_index,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "supported": self.supported,
            "phase": self.phase,
            "support_state": self.support_state,
            "required_future_capability": self.required_future_capability,
        }


@dataclass(frozen=True)
class SurfaceCSGFeatureDependencyRecord:
    """One non-primitive feature builder dependency on surface CSG policy."""

    caller_id: str
    module: str
    operation: SurfaceBooleanOperation | None
    surface_builder: str
    explicit_mesh_route: str | None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "caller_id": self.caller_id,
            "explicit_mesh_route": self.explicit_mesh_route,
            "module": self.module,
            "operation": self.operation,
            "surface_builder": self.surface_builder,
        }


@dataclass(frozen=True)
class SurfaceBooleanOperands:
    """Canonical surfaced boolean operands ready for execution."""

    operation: SurfaceBooleanOperation
    bodies: tuple[SurfaceBody, ...]

    @property
    def operand_count(self) -> int:
        return len(self.bodies)

    @property
    def body_ids(self) -> tuple[str, ...]:
        return tuple(body.stable_identity for body in self.bodies)


@dataclass(frozen=True)
class SurfaceCSGOperationPlan:
    """Pre-execution surface CSG plan with accumulated diagnostics."""

    operation: SurfaceBooleanOperation
    operands: SurfaceBooleanOperands | None
    pair_dispatch: tuple[SurfaceCSGPairDispatchRecord, ...] = ()
    diagnostics: tuple[SurfaceCSGPlanDiagnostic, ...] = ()

    @property
    def executable(self) -> bool:
        return self.operands is not None and not self.diagnostics

    @property
    def body_ids(self) -> tuple[str, ...]:
        return () if self.operands is None else self.operands.body_ids

    def assert_executable(self) -> "SurfaceCSGOperationPlan":
        if not self.executable:
            message = "; ".join(diagnostic.message for diagnostic in self.diagnostics)
            raise SurfaceBooleanEligibilityError(message or f"Surface CSG {self.operation} plan is not executable.")
        return self

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "body_ids": self.body_ids,
            "executable": self.executable,
            "pair_dispatch": [record.canonical_payload() for record in self.pair_dispatch],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceBooleanResult:
    """Structured surfaced boolean result contract."""

    operation: SurfaceBooleanOperation
    operands: SurfaceBooleanOperands
    status: SurfaceBooleanStatus
    body: SurfaceBody | None = None
    classification: SurfaceBooleanClassification | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status == "succeeded":
            if self.classification is None:
                raise ValueError("Succeeded surface boolean results require classification.")
            if self.classification == "empty":
                if self.body is not None:
                    raise ValueError("Empty succeeded surface boolean results may not carry body.")
            elif self.body is None:
                raise ValueError("Non-empty succeeded surface boolean results require body.")
            if self.failure_reason is not None:
                raise ValueError("Succeeded surface boolean results may not carry failure_reason.")
        else:
            if self.body is not None or self.classification is not None:
                raise ValueError("Invalid or unsupported surface boolean results may not carry body or classification.")
            if not self.failure_reason:
                raise ValueError("Invalid or unsupported surface boolean results require failure_reason.")

    @property
    def body_id(self) -> str | None:
        return None if self.body is None else self.body.stable_identity


@dataclass(frozen=True)
class SurfaceBooleanPatchRef:
    """Stable reference to one patch in one boolean operand."""

    operand_index: int
    patch_index: int


@dataclass(frozen=True)
class SurfaceBooleanTrimFragment:
    """One cut fragment expressed in patch-local UV coordinates."""

    patch: SurfaceBooleanPatchRef
    points_uv: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SurfaceCSGToleranceDiagnostic:
    """Explicit tolerance diagnostic emitted by CSG curve helpers."""

    code: SurfaceCSGToleranceDiagnosticCode
    message: str
    curve_id: str | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "curve_id": self.curve_id,
            "message": self.message,
        }


@dataclass(frozen=True)
class SurfaceCSGTolerancePolicy:
    """Shared CSG numeric policy for snapping, equality, degeneracy, and domains."""

    snap_tolerance: float = 1e-9
    equality_tolerance: float = 1e-9
    degeneracy_tolerance: float = 1e-9
    domain_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        values = {
            "snap_tolerance": self.snap_tolerance,
            "equality_tolerance": self.equality_tolerance,
            "degeneracy_tolerance": self.degeneracy_tolerance,
            "domain_tolerance": self.domain_tolerance,
        }
        for name, value in values.items():
            normalized = float(value)
            if not np.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(f"SurfaceCSGTolerancePolicy.{name} must be a positive finite value.")
            object.__setattr__(self, name, normalized)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "snap_tolerance": self.snap_tolerance,
            "equality_tolerance": self.equality_tolerance,
            "degeneracy_tolerance": self.degeneracy_tolerance,
            "domain_tolerance": self.domain_tolerance,
        }


DEFAULT_SURFACE_CSG_TOLERANCE_POLICY = SurfaceCSGTolerancePolicy()


@dataclass(frozen=True)
class SurfaceCSGCurvePrimitive:
    """Surface-native CSG curve record shared by intersection and trim stages."""

    kind: SurfaceCSGCurveKind
    points_3d: tuple[tuple[float, float, float], ...]
    parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"line", "arc", "conic", "sampled"}:
            raise ValueError("SurfaceCSGCurvePrimitive.kind must be line, arc, conic, or sampled.")
        points = tuple(_normalize_curve_point(point) for point in self.points_3d)
        if len(points) < 2:
            raise ValueError("SurfaceCSGCurvePrimitive requires at least two points.")
        parameters = tuple((str(name), float(value)) for name, value in self.parameters)
        if any(not name for name, _ in parameters):
            raise ValueError("SurfaceCSGCurvePrimitive parameter names must be non-empty.")
        if any(not np.isfinite(value) for _, value in parameters):
            raise ValueError("SurfaceCSGCurvePrimitive parameter values must be finite.")
        object.__setattr__(self, "points_3d", points)
        object.__setattr__(self, "parameters", tuple(sorted(parameters)))

    def canonical_payload(self, policy: SurfaceCSGTolerancePolicy | None = None) -> dict[str, object]:
        normalized_policy = normalize_surface_csg_tolerance_policy(policy)
        return {
            "kind": self.kind,
            "parameters": self.parameters,
            "points_3d": tuple(
                tuple(_snap_scalar(component, normalized_policy.snap_tolerance) for component in point)
                for point in self.points_3d
            ),
        }


@dataclass(frozen=True)
class SurfaceCSGPatchLocalCurve:
    """One CSG curve mapped into a participating patch parameter domain."""

    source_curve_digest: str
    patch: SurfaceBooleanPatchRef
    points_uv: tuple[tuple[float, float], ...]
    domain_bounds: tuple[float, float, float, float]
    orientation: Literal["forward", "reversed"] = "forward"

    def __post_init__(self) -> None:
        points_uv = tuple(_normalize_curve_point_uv(point) for point in self.points_uv)
        if len(points_uv) < 2:
            raise ValueError("SurfaceCSGPatchLocalCurve requires at least two UV points.")
        if self.orientation not in {"forward", "reversed"}:
            raise ValueError("SurfaceCSGPatchLocalCurve.orientation must be forward or reversed.")
        object.__setattr__(self, "points_uv", points_uv)

    def canonical_payload(self, policy: SurfaceCSGTolerancePolicy | None = None) -> dict[str, object]:
        normalized_policy = normalize_surface_csg_tolerance_policy(policy)
        return {
            "domain_bounds": self.domain_bounds,
            "orientation": self.orientation,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "points_uv": tuple(
                tuple(_snap_scalar(component, normalized_policy.snap_tolerance) for component in point)
                for point in self.points_uv
            ),
            "source_curve_digest": self.source_curve_digest,
        }


@dataclass(frozen=True)
class SurfaceCSGCurveMappingDiagnostic:
    """Explicit diagnostic for failed 3D-curve to patch-local mapping."""

    code: SurfaceCSGToleranceDiagnosticCode
    message: str
    patch: SurfaceBooleanPatchRef
    source_curve_digest: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "source_curve_digest": self.source_curve_digest,
        }


@dataclass(frozen=True)
class SurfaceCSGPatchLocalCurveMappingResult:
    """Result of mapping one CSG curve into one patch-local parameter domain."""

    source_curve: SurfaceCSGCurvePrimitive
    patch: SurfaceBooleanPatchRef
    curve: SurfaceCSGPatchLocalCurve | None = None
    diagnostics: tuple[SurfaceCSGCurveMappingDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return self.curve is not None and not self.diagnostics


@dataclass(frozen=True)
class SurfaceCSGPlanarRelationDiagnostic:
    """Explicit diagnostic for a low-order analytic CSG relation."""

    relation: SurfaceCSGPlanarRelation
    message: str
    first_patch: SurfaceBooleanPatchRef
    second_patch: SurfaceBooleanPatchRef

    def canonical_payload(self) -> dict[str, object]:
        return {
            "first_patch": {
                "operand_index": self.first_patch.operand_index,
                "patch_index": self.first_patch.patch_index,
            },
            "message": self.message,
            "relation": self.relation,
            "second_patch": {
                "operand_index": self.second_patch.operand_index,
                "patch_index": self.second_patch.patch_index,
            },
        }


@dataclass(frozen=True)
class SurfaceCSGConicDiagnostic:
    """Explicit diagnostic for revolution/conic analytic intersections."""

    code: str
    message: str
    first_patch: SurfaceBooleanPatchRef
    second_patch: SurfaceBooleanPatchRef

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "first_patch": {
                "operand_index": self.first_patch.operand_index,
                "patch_index": self.first_patch.patch_index,
            },
            "message": self.message,
            "second_patch": {
                "operand_index": self.second_patch.operand_index,
                "patch_index": self.second_patch.patch_index,
            },
        }


@dataclass(frozen=True)
class SurfaceCSGRevolutionIntersectionRecord:
    """Analytic revolution/conic patch intersection result."""

    first_patch: SurfaceBooleanPatchRef
    second_patch: SurfaceBooleanPatchRef
    conic_kind: Literal["circle", "ellipse", "parabola", "hyperbola", "line", "unsupported"]
    curve: SurfaceCSGCurvePrimitive | None = None
    patch_local_curves: tuple[SurfaceCSGPatchLocalCurve, ...] = ()
    diagnostics: tuple[SurfaceCSGConicDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return self.curve is not None and self.conic_kind != "unsupported" and not self.diagnostics


@dataclass(frozen=True)
class SurfaceCSGHigherOrderSupportRecord:
    """Explicit solver-boundary decision for higher-order CSG family pairs."""

    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    supported: bool
    solver_boundary: str
    support_state: SurfaceBooleanSupportState = "not-yet-implemented"
    required_future_capability: str | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "left_family": self.left_family,
            "operation": self.operation,
            "required_future_capability": self.required_future_capability,
            "right_family": self.right_family,
            "solver_boundary": self.solver_boundary,
            "support_state": self.support_state,
            "supported": self.supported,
        }


@dataclass(frozen=True)
class SurfaceCSGHigherOrderRefusalDiagnostic:
    """Explicit refusal diagnostic for unsupported higher-order CSG pairs."""

    support: SurfaceCSGHigherOrderSupportRecord

    @property
    def message(self) -> str:
        return (
            f"unsupported higher-order surface boolean pair for {self.support.operation}: "
            f"{self.support.left_family}/{self.support.right_family} at {self.support.solver_boundary}; "
            f"requires {self.support.required_future_capability}"
        )

    def canonical_payload(self) -> dict[str, object]:
        payload = self.support.canonical_payload()
        payload["message"] = self.message
        return payload


@dataclass(frozen=True)
class SurfaceCSGAnalyticIntersectionRecord:
    """Analytic low-order patch intersection result."""

    first_patch: SurfaceBooleanPatchRef
    second_patch: SurfaceBooleanPatchRef
    relation: SurfaceCSGPlanarRelation
    curve: SurfaceCSGCurvePrimitive | None = None
    patch_local_curves: tuple[SurfaceCSGPatchLocalCurve, ...] = ()
    diagnostics: tuple[SurfaceCSGPlanarRelationDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return self.curve is not None and self.relation == "crossing" and not self.diagnostics


@dataclass(frozen=True)
class SurfaceCSGArrangementDiagnostic:
    """Explicit diagnostic for invalid patch-local curve arrangements."""

    code: Literal["ambiguous-overlap", "self-intersection", "zero-length-fragment", "outside-domain"]
    message: str
    patch: SurfaceBooleanPatchRef
    cut_curve_ids: tuple[str, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "cut_curve_ids": self.cut_curve_ids,
            "message": self.message,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
        }


@dataclass(frozen=True)
class SurfaceCSGSplitTrimLoopRecord:
    """One deterministic trim-loop fragment produced by a CSG arrangement."""

    patch: SurfaceBooleanPatchRef
    loop: TrimLoop
    source_category: str
    cut_curve_ids: tuple[str, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "cut_curve_ids": self.cut_curve_ids,
            "loop": {
                "category": self.loop.category,
                "clockwise": self.loop.is_clockwise,
                "points_uv": tuple(tuple(point) for point in self.loop.points_uv),
            },
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "source_category": self.source_category,
        }


@dataclass(frozen=True)
class SurfaceCSGPatchLocalArrangementGraph:
    """Patch-local arrangement of intersection curves and split trim loops."""

    patch: SurfaceBooleanPatchRef
    patch_local_curves: tuple[SurfaceCSGPatchLocalCurve, ...]
    split_loops: tuple[SurfaceCSGSplitTrimLoopRecord, ...]
    diagnostics: tuple[SurfaceCSGArrangementDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.diagnostics


@dataclass(frozen=True)
class SurfaceBooleanCutCurve:
    """One surfaced cut curve shared by two operand patches."""

    cut_curve_id: str
    points_3d: tuple[tuple[float, float, float], ...]
    patches: tuple[SurfaceBooleanPatchRef, SurfaceBooleanPatchRef]
    trim_fragments: tuple[SurfaceBooleanTrimFragment, SurfaceBooleanTrimFragment]
    curve: SurfaceCSGCurvePrimitive | None = None
    patch_local_curves: tuple[SurfaceCSGPatchLocalCurve, ...] = ()


@dataclass(frozen=True)
class SurfaceBooleanPatchClassification:
    """One deterministic patch classification relative to the opposing operand."""

    patch: SurfaceBooleanPatchRef
    relation: SurfaceBooleanPatchRelation
    cut_curve_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SurfaceCSGFragmentClassificationDiagnostic:
    """Explicit diagnostic for ambiguous fragment classification."""

    code: Literal["open-body", "ambiguous-boundary", "outside-domain", "unstable-sample"]
    message: str
    patch: SurfaceBooleanPatchRef
    sample_point: tuple[float, float, float] | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "sample_point": self.sample_point,
        }


@dataclass(frozen=True)
class SurfaceCSGFragmentClassificationRecord:
    """Surface-native inside/outside/on classification for one split fragment."""

    patch: SurfaceBooleanPatchRef
    relation: SurfaceBooleanPatchRelation
    sample_uv: tuple[float, float]
    sample_point: tuple[float, float, float]
    cut_curve_ids: tuple[str, ...] = ()
    diagnostics: tuple[SurfaceCSGFragmentClassificationDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "cut_curve_ids": self.cut_curve_ids,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "relation": self.relation,
            "sample_point": self.sample_point,
            "sample_uv": self.sample_uv,
        }


@dataclass(frozen=True)
class SurfaceCSGCutCapRequirementRecord:
    """Operation-specific indication that a classified fragment contributes a cut cap."""

    patch: SurfaceBooleanPatchRef
    required: bool
    reason: str
    cut_curve_ids: tuple[str, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "cut_curve_ids": self.cut_curve_ids,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "reason": self.reason,
            "required": self.required,
        }


@dataclass(frozen=True)
class SurfaceCSGOperationSelectionRecord:
    """Deterministic operation-selection result for one classified fragment."""

    operation: SurfaceBooleanOperation
    patch: SurfaceBooleanPatchRef
    relation: SurfaceBooleanPatchRelation
    role: SurfaceBooleanSplitRole
    cut_cap: SurfaceCSGCutCapRequirementRecord
    cut_curve_ids: tuple[str, ...] = ()

    @property
    def survives(self) -> bool:
        return self.role in {"survive", "cut_cap"}

    def canonical_payload(self) -> dict[str, object]:
        return {
            "cut_cap": self.cut_cap.canonical_payload(),
            "cut_curve_ids": self.cut_curve_ids,
            "operation": self.operation,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "relation": self.relation,
            "role": self.role,
            "survives": self.survives,
        }


@dataclass(frozen=True)
class SurfaceBooleanSplitRecord:
    """One deterministic surfaced split-selection record for a source patch fragment."""

    patch: SurfaceBooleanPatchRef
    relation: SurfaceBooleanPatchRelation
    role: SurfaceBooleanSplitRole
    cut_curve_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SurfaceBooleanTrimmedPatchFragment:
    """One reconstructed surfaced patch fragment before shell assembly."""

    source_patch: SurfaceBooleanPatchRef
    patch: PlanarSurfacePatch
    cut_curve_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SurfaceCSGFragmentProvenanceRecord:
    """Provenance for one assembled CSG fragment."""

    source_patch: SurfaceBooleanPatchRef
    result_shell_index: int
    result_patch_index: int
    cut_curve_ids: tuple[str, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "cut_curve_ids": self.cut_curve_ids,
            "result_patch_index": self.result_patch_index,
            "result_shell_index": self.result_shell_index,
            "source_patch": {
                "operand_index": self.source_patch.operand_index,
                "patch_index": self.source_patch.patch_index,
            },
        }


@dataclass(frozen=True)
class SurfaceCSGReconstructionDiagnostic:
    """Diagnostic emitted while converting transient CSG records into durable shells."""

    code: Literal[
        "invalid-fragment-graph",
        "invalid-cut-boundary",
        "missing-source-patch",
        "missing-cap-payload",
        "empty-shell",
        "assembly-error",
    ]
    message: str
    source_patch: SurfaceBooleanPatchRef | None = None
    cap_payload_index: int | None = None

    def canonical_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "cap_payload_index": self.cap_payload_index,
            "code": self.code,
            "message": self.message,
            "source_patch": None,
        }
        if self.source_patch is not None:
            payload["source_patch"] = {
                "operand_index": self.source_patch.operand_index,
                "patch_index": self.source_patch.patch_index,
            }
        return payload


@dataclass(frozen=True)
class SurfaceCSGShellOrderingRecord:
    """Stable shell ordering witness for a reconstructed CSG result."""

    result_shell_index: int
    patch_count: int
    source_patches: tuple[SurfaceBooleanPatchRef, ...]
    sort_key: tuple[object, ...]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "patch_count": self.patch_count,
            "result_shell_index": self.result_shell_index,
            "sort_key": self.sort_key,
            "source_patches": tuple(
                {
                    "operand_index": patch.operand_index,
                    "patch_index": patch.patch_index,
                }
                for patch in self.source_patches
            ),
        }


@dataclass(frozen=True)
class SurfaceCSGShellAssemblyRecord:
    """Provisional CSG shell assembly result before seam rebuild and validity cleanup."""

    operation: SurfaceBooleanOperation
    classification: SurfaceBooleanClassification
    shells: tuple[SurfaceShell, ...] = ()
    provenance: tuple[SurfaceCSGFragmentProvenanceRecord, ...] = ()
    shell_ordering: tuple[SurfaceCSGShellOrderingRecord, ...] = ()
    diagnostics: tuple[SurfaceCSGReconstructionDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.diagnostics

    def to_body(self, *, metadata: dict[str, object] | None = None) -> SurfaceBody | None:
        if not self.supported:
            raise SurfaceBooleanEligibilityError("; ".join(diagnostic.message for diagnostic in self.diagnostics))
        if self.classification == "empty":
            return None
        return make_surface_body(self.shells, metadata=metadata)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "operation": self.operation,
            "provenance": [record.canonical_payload() for record in self.provenance],
            "shell_ordering": [record.canonical_payload() for record in self.shell_ordering],
            "shell_count": len(self.shells),
        }


@dataclass(frozen=True)
class SurfaceCSGBoundaryUseProvenanceRecord:
    """Provenance for one result boundary use during CSG seam rebuild."""

    boundary: SurfaceBoundaryRef
    use_count: int
    seam_ids: tuple[str, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary": {
                "boundary_id": self.boundary.boundary_id,
                "patch_index": self.boundary.patch_index,
            },
            "seam_ids": self.seam_ids,
            "use_count": self.use_count,
        }


@dataclass(frozen=True)
class SurfaceCSGSeamRebuildRecord:
    """Rebuilt seam and adjacency truth for one provisional CSG shell."""

    shell: SurfaceShell
    boundary_uses: tuple[SurfaceCSGBoundaryUseProvenanceRecord, ...]
    diagnostics: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary_uses": [record.canonical_payload() for record in self.boundary_uses],
            "diagnostics": self.diagnostics,
            "patch_count": self.shell.patch_count,
            "seam_count": len(self.shell.seams),
        }


@dataclass(frozen=True)
class SurfaceCSGValidityDiagnostic:
    """Diagnostic emitted by the final CSG validity/provenance gate."""

    code: Literal["invalid-shell", "non-closed-result", "healing-failed"]
    message: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class SurfaceCSGProvenanceMetadataRecord:
    """Deterministic CSG operation provenance attached to accepted results."""

    operation: SurfaceBooleanOperation
    operand_ids: tuple[str, ...]
    backend: str = "surface"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "operand_ids": self.operand_ids,
            "operation": self.operation,
        }


@dataclass(frozen=True)
class SurfaceCSGValidityGateRecord:
    """Final CSG acceptance gate output."""

    status: SurfaceBooleanStatus
    body: SurfaceBody | None = None
    diagnostics: tuple[SurfaceCSGValidityDiagnostic, ...] = ()
    provenance: SurfaceCSGProvenanceMetadataRecord | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "succeeded" and self.body is not None and not self.diagnostics


@dataclass(frozen=True)
class SurfaceCSGPostReconstructionValidityDiagnostic:
    """Diagnostic emitted while handing reconstructed CSG shells to validity gates."""

    code: Literal["invalid-assembly", "seam-rebuild-failed", "validity-gate-rejected"]
    message: str
    result_shell_index: int | None = None
    underlying_code: str | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "result_shell_index": self.result_shell_index,
            "underlying_code": self.underlying_code,
        }


@dataclass(frozen=True)
class SurfaceCSGValidityHandoffRecord:
    """Post-assembly handoff from durable CSG shell candidates to final validity."""

    operation: SurfaceBooleanOperation
    classification: SurfaceBooleanClassification
    status: SurfaceBooleanStatus
    assembly: SurfaceCSGShellAssemblyRecord
    body: SurfaceBody | None = None
    seam_rebuilds: tuple[SurfaceCSGSeamRebuildRecord, ...] = ()
    validity_gate: SurfaceCSGValidityGateRecord | None = None
    diagnostics: tuple[SurfaceCSGPostReconstructionValidityDiagnostic, ...] = ()

    @property
    def accepted(self) -> bool:
        if self.classification == "empty":
            return self.status == "succeeded" and not self.diagnostics
        return self.status == "succeeded" and self.body is not None and not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "assembly": self.assembly.canonical_payload(),
            "body_id": None if self.body is None else self.body.stable_identity,
            "classification": self.classification,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "operation": self.operation,
            "seam_rebuilds": [record.canonical_payload() for record in self.seam_rebuilds],
            "status": self.status,
            "validity_gate": None
            if self.validity_gate is None
            else {
                "accepted": self.validity_gate.accepted,
                "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.validity_gate.diagnostics],
                "status": self.validity_gate.status,
            },
        }


@dataclass(frozen=True)
class SurfaceCSGOperandOrderingNormalizationRecord:
    """Deterministic operand ordering witness for CSG provenance maps."""

    operation: SurfaceBooleanOperation
    operand_ids: tuple[str, ...]
    normalized_operand_ids: tuple[str, ...]
    normalized_to_original_indices: tuple[int, ...]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "normalized_operand_ids": self.normalized_operand_ids,
            "normalized_to_original_indices": self.normalized_to_original_indices,
            "operand_ids": self.operand_ids,
            "operation": self.operation,
        }


@dataclass(frozen=True)
class SurfaceCSGProvenanceDiagnostic:
    """Diagnostic emitted while building result provenance maps."""

    code: Literal["invalid-assembly", "missing-result-patch", "missing-boundary-attachment"]
    message: str
    result_shell_index: int | None = None
    result_patch_index: int | None = None
    source_patch: SurfaceBooleanPatchRef | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "result_patch_index": self.result_patch_index,
            "result_shell_index": self.result_shell_index,
            "source_patch": None
            if self.source_patch is None
            else {
                "operand_index": self.source_patch.operand_index,
                "patch_index": self.source_patch.patch_index,
            },
        }


@dataclass(frozen=True)
class SurfaceCSGResultPatchProvenanceRecord:
    """Stable provenance for one patch in a reconstructed CSG result shell."""

    result_shell_index: int
    result_patch_index: int
    source_patch: SurfaceBooleanPatchRef
    source_role: Literal["surviving-fragment", "generated-cap"]
    cut_curve_ids: tuple[str, ...] = ()
    cap_payload_index: int | None = None
    boundary_attachment_index: int | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary_attachment_index": self.boundary_attachment_index,
            "cap_payload_index": self.cap_payload_index,
            "cut_curve_ids": self.cut_curve_ids,
            "result_patch_index": self.result_patch_index,
            "result_shell_index": self.result_shell_index,
            "source_patch": {
                "operand_index": self.source_patch.operand_index,
                "patch_index": self.source_patch.patch_index,
            },
            "source_role": self.source_role,
        }


@dataclass(frozen=True)
class SurfaceCSGResultProvenanceMap:
    """Complete traceability map for a CSG result candidate."""

    operation: SurfaceBooleanOperation
    operand_ordering: SurfaceCSGOperandOrderingNormalizationRecord
    result_patches: tuple[SurfaceCSGResultPatchProvenanceRecord, ...] = ()
    diagnostics: tuple[SurfaceCSGProvenanceDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "operand_ordering": self.operand_ordering.canonical_payload(),
            "operation": self.operation,
            "result_patches": [record.canonical_payload() for record in self.result_patches],
            "supported": self.supported,
        }


@dataclass(frozen=True)
class SurfaceBooleanIntersectionStage:
    """Deterministic surfaced intersection/classification stage output."""

    operation: SurfaceBooleanOperation
    operands: SurfaceBooleanOperands
    supported: bool
    body_relation: SurfaceBooleanBodyRelation
    cut_curves: tuple[SurfaceBooleanCutCurve, ...] = ()
    patch_classifications: tuple[SurfaceBooleanPatchClassification, ...] = ()
    split_records: tuple[SurfaceBooleanSplitRecord, ...] = ()
    support_reason: str | None = None

    def __post_init__(self) -> None:
        if self.supported and self.support_reason is not None:
            raise ValueError("Supported intersection stages may not carry support_reason.")
        if not self.supported and not self.support_reason:
            raise ValueError("Unsupported intersection stages require support_reason.")


@dataclass(frozen=True)
class SurfaceCSGFragmentClassificationEdgeRecord:
    """One classified fragment edge in the transient CSG fragment graph."""

    patch: SurfaceBooleanPatchRef
    relation: SurfaceBooleanPatchRelation
    role: SurfaceBooleanSplitRole
    cut_curve_ids: tuple[str, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "relation": self.relation,
            "role": self.role,
            "cut_curve_ids": self.cut_curve_ids,
        }


@dataclass(frozen=True)
class SurfaceCSGFragmentGraphDiagnostic:
    """Diagnostic emitted while building the transient CSG fragment graph."""

    code: Literal["non-executable-plan", "unsupported-intersection-stage", "missing-selection"]
    message: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class SurfaceCSGFragmentGraphRecord:
    """Transient classified CSG fragment graph for a planned operation."""

    operation: SurfaceBooleanOperation
    plan: SurfaceCSGOperationPlan
    intersection_stage: SurfaceBooleanIntersectionStage | None = None
    classification_edges: tuple[SurfaceCSGFragmentClassificationEdgeRecord, ...] = ()
    diagnostics: tuple[SurfaceCSGFragmentGraphDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "supported": self.supported,
            "plan": self.plan.canonical_payload(),
            "intersection_stage_supported": None
            if self.intersection_stage is None
            else self.intersection_stage.supported,
            "classification_edges": [edge.canonical_payload() for edge in self.classification_edges],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceCSGUnsupportedCapDiagnostic:
    """Diagnostic for a cut-cap request that cannot produce a surface-native cap patch."""

    code: Literal["unsupported-cap-family", "missing-source-patch", "invalid-fragment-graph"]
    message: str
    patch: SurfaceBooleanPatchRef | None = None
    cap_family: str | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "patch": None
            if self.patch is None
            else {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "cap_family": self.cap_family,
        }


@dataclass(frozen=True)
class SurfaceCSGGeneratedCapPatchPayloadRecord:
    """Generated cap patch payload for one cut-cap fragment."""

    source_patch: SurfaceBooleanPatchRef
    cap_family: str
    patch: PlanarSurfacePatch
    cut_curve_ids: tuple[str, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "source_patch": {
                "operand_index": self.source_patch.operand_index,
                "patch_index": self.source_patch.patch_index,
            },
            "cap_family": self.cap_family,
            "patch_id": self.patch.stable_identity,
            "cut_curve_ids": self.cut_curve_ids,
        }


@dataclass(frozen=True)
class SurfaceCSGCapConstructionRecord:
    """Generated cap payloads and diagnostics for a classified fragment graph."""

    operation: SurfaceBooleanOperation
    cap_payloads: tuple[SurfaceCSGGeneratedCapPatchPayloadRecord, ...] = ()
    diagnostics: tuple[SurfaceCSGUnsupportedCapDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "supported": self.supported,
            "cap_payloads": [payload.canonical_payload() for payload in self.cap_payloads],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceCSGTrimAttachmentRecord:
    """Trim attachment for a generated CSG cap payload."""

    source_patch: SurfaceBooleanPatchRef
    cap_payload_index: int
    trim_loop: TrimLoop
    cut_curve_ids: tuple[str, ...] = ()
    exposure: Literal["shared", "open"] = "shared"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "source_patch": {
                "operand_index": self.source_patch.operand_index,
                "patch_index": self.source_patch.patch_index,
            },
            "cap_payload_index": self.cap_payload_index,
            "trim_loop": self.trim_loop.normalized().canonical_payload(),
            "cut_curve_ids": self.cut_curve_ids,
            "exposure": self.exposure,
        }


@dataclass(frozen=True)
class SurfaceCSGBoundaryExposureDiagnostic:
    """Diagnostic for exposed generated CSG cap boundaries."""

    code: Literal["invalid-cap-construction", "open-boundary"]
    message: str
    source_patch: SurfaceBooleanPatchRef | None = None
    cap_payload_index: int | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "source_patch": None
            if self.source_patch is None
            else {
                "operand_index": self.source_patch.operand_index,
                "patch_index": self.source_patch.patch_index,
            },
            "cap_payload_index": self.cap_payload_index,
        }


@dataclass(frozen=True)
class SurfaceCSGCutBoundaryRecord:
    """Cut-boundary trim attachments for generated CSG cap payloads."""

    operation: SurfaceBooleanOperation
    trim_attachments: tuple[SurfaceCSGTrimAttachmentRecord, ...] = ()
    diagnostics: tuple[SurfaceCSGBoundaryExposureDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "supported": self.supported,
            "trim_attachments": [attachment.canonical_payload() for attachment in self.trim_attachments],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceBooleanFamilyPairSupport:
    """Boolean support declaration for one pair of patch families."""

    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    supported: bool
    phase: SurfaceBooleanUnsupportedPhase | str
    support_state: SurfaceBooleanSupportState = "unsupported"
    required_future_capability: str | None = None

    def __post_init__(self) -> None:
        operation = str(self.operation)
        if operation not in {"union", "difference", "intersection"}:
            raise ValueError("SurfaceBooleanFamilyPairSupport.operation is not supported.")
        left_family = str(self.left_family).strip()
        right_family = str(self.right_family).strip()
        phase = str(self.phase).strip()
        support_state = str(self.support_state).strip()
        if not left_family or not right_family or not phase:
            raise ValueError("SurfaceBooleanFamilyPairSupport families and phase must be non-empty.")
        if support_state not in {"exact", "declared-tolerance", "adapter", "unsupported", "not-yet-implemented"}:
            raise ValueError("SurfaceBooleanFamilyPairSupport.support_state is not supported.")
        if self.supported and support_state in {"unsupported", "not-yet-implemented"}:
            raise ValueError("Supported surface boolean family pairs require an executable support_state.")
        if not self.supported and support_state in {"exact", "declared-tolerance", "adapter"}:
            raise ValueError("Unsupported surface boolean family pairs may not use an executable support_state.")
        future = None if self.required_future_capability is None else str(self.required_future_capability).strip()
        if future == "":
            raise ValueError("SurfaceBooleanFamilyPairSupport.required_future_capability must be non-empty when provided.")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "left_family", left_family)
        object.__setattr__(self, "right_family", right_family)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "support_state", support_state)
        object.__setattr__(self, "required_future_capability", future)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "supported": self.supported,
            "phase": self.phase,
            "support_state": self.support_state,
            "required_future_capability": self.required_future_capability,
        }


@dataclass(frozen=True)
class SurfaceBooleanUnsupportedFamilyDiagnostic:
    """Explicit diagnostic for a boolean family pair that is outside current support."""

    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    phase: str
    required_future_capability: str

    @property
    def message(self) -> str:
        return (
            f"unsupported surface boolean family pair for {self.operation}: "
            f"{self.left_family}/{self.right_family} at phase {self.phase}; "
            f"requires {self.required_future_capability}"
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "phase": self.phase,
            "required_future_capability": self.required_future_capability,
            "message": self.message,
        }


@dataclass(frozen=True)
class SurfaceCSGSolverRegistryDiagnostic:
    """Coverage diagnostic for the surface CSG family-pair solver registry."""

    code: Literal["missing-pair", "unknown-pair", "record-key-mismatch", "missing-future-capability"]
    message: str
    operation: SurfaceBooleanOperation | None = None
    left_family: str | None = None
    right_family: str | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
        }


@dataclass(frozen=True)
class SurfaceCSGSolverRegistryRecord:
    """Auditable registry for every promoted surface CSG operation/family pair."""

    operations: tuple[SurfaceBooleanOperation, ...]
    families: tuple[str, ...]
    support_records: tuple[SurfaceBooleanFamilyPairSupport, ...]
    diagnostics: tuple[SurfaceCSGSolverRegistryDiagnostic, ...] = ()
    _support_lookup: Mapping[tuple[SurfaceBooleanOperation, str, str], SurfaceBooleanFamilyPairSupport] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        operations = tuple(str(operation) for operation in self.operations)
        families = tuple(str(family).strip() for family in self.families)
        if not operations or any(operation not in SURFACE_BOOLEAN_OPERATIONS for operation in operations):
            raise ValueError("SurfaceCSGSolverRegistryRecord.operations must be supported CSG operations.")
        if not families or any(not family for family in families):
            raise ValueError("SurfaceCSGSolverRegistryRecord.families must be non-empty.")
        if len(set(families)) != len(families):
            raise ValueError("SurfaceCSGSolverRegistryRecord.families must be unique.")
        lookup = {
            (record.operation, record.left_family, record.right_family): record
            for record in self.support_records
        }
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "families", families)
        object.__setattr__(self, "_support_lookup", lookup)

    @property
    def passed(self) -> bool:
        return not self.diagnostics

    def support_for(
        self,
        operation: SurfaceBooleanOperation,
        left_family: str,
        right_family: str,
    ) -> SurfaceBooleanFamilyPairSupport:
        """Return registry support for one operation/family pair."""

        key = (operation, left_family, right_family)
        support = self._support_lookup.get(key)
        if support is not None:
            return support
        reverse = self._support_lookup.get((operation, right_family, left_family))
        if reverse is not None:
            return SurfaceBooleanFamilyPairSupport(
                operation=operation,
                left_family=left_family,
                right_family=right_family,
                supported=reverse.supported,
                phase=reverse.phase,
                support_state=reverse.support_state,
                required_future_capability=reverse.required_future_capability,
            )
        return SurfaceBooleanFamilyPairSupport(
            operation=operation,
            left_family=left_family,
            right_family=right_family,
            supported=False,
            phase="operand-family-eligibility",
            support_state="unsupported",
            required_future_capability=_surface_boolean_required_future_capability(operation, left_family, right_family),
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operations": self.operations,
            "families": self.families,
            "passed": self.passed,
            "support_records": [record.canonical_payload() for record in self.support_records],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceCSGPrimitiveAnalyticPairRecord:
    """Analytic primitive CSG support declaration for one operation/family pair."""

    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    supported: bool
    support_state: SurfaceBooleanSupportState
    tolerance_policy: SurfaceCSGTolerancePolicy = DEFAULT_SURFACE_CSG_TOLERANCE_POLICY
    diagnostic: str = ""

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "supported": self.supported,
            "support_state": self.support_state,
            "tolerance_policy": self.tolerance_policy.canonical_payload(),
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True)
class SurfaceBooleanFamilyEligibilityResult:
    """Structural family eligibility result for a surfaced boolean request."""

    operation: SurfaceBooleanOperation
    supported: bool
    family_pairs: tuple[SurfaceBooleanFamilyPairSupport, ...]
    diagnostics: tuple[SurfaceBooleanUnsupportedFamilyDiagnostic, ...] = ()

    @property
    def required_future_capabilities(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(diagnostic.required_future_capability for diagnostic in self.diagnostics))

    @property
    def failure_reason(self) -> str | None:
        if self.supported:
            return None
        return "; ".join(diagnostic.message for diagnostic in self.diagnostics)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "supported": self.supported,
            "family_pairs": [item.canonical_payload() for item in self.family_pairs],
            "diagnostics": [item.canonical_payload() for item in self.diagnostics],
            "required_future_capabilities": self.required_future_capabilities,
        }


_SURFACE_BOOLEAN_EXECUTABLE_FAMILY_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        tuple(sorted((left_family, right_family)))
        for left_family, left_capability in PATCH_FAMILY_CAPABILITY_MATRIX.items()
        for right_family, right_capability in PATCH_FAMILY_CAPABILITY_MATRIX.items()
        if left_capability.support_phase == "available" and right_capability.support_phase == "available"
    }
)
ANALYTIC_SURFACE_CSG_FAMILIES: frozenset[str] = frozenset({"planar", "ruled", "revolution"})
SAMPLED_SURFACE_CSG_FAMILIES: frozenset[str] = frozenset({"implicit", "heightmap", "displacement"})
HIGHER_ORDER_SURFACE_CSG_FAMILIES: frozenset[str] = frozenset(
    {"bspline", "nurbs", "sweep", "subdivision", "implicit", "heightmap", "displacement", "torus"}
)


def _surface_boolean_required_future_capability(
    operation: SurfaceBooleanOperation,
    left_family: str,
    right_family: str,
) -> str:
    left_capability = PATCH_FAMILY_CAPABILITY_MATRIX.get(left_family)
    right_capability = PATCH_FAMILY_CAPABILITY_MATRIX.get(right_family)
    left_phase = "unknown" if left_capability is None else left_capability.support_phase
    right_phase = "unknown" if right_capability is None else right_capability.support_phase
    return (
        f"surface boolean {operation} support for {left_family}/{right_family} "
        f"families ({left_phase}/{right_phase})"
    )


def _surface_boolean_support_record(
    operation: SurfaceBooleanOperation,
    left_family: str,
    right_family: str,
) -> SurfaceBooleanFamilyPairSupport:
    canonical_pair = tuple(sorted((left_family, right_family)))
    supported = canonical_pair in _SURFACE_BOOLEAN_EXECUTABLE_FAMILY_PAIRS
    left_capability = PATCH_FAMILY_CAPABILITY_MATRIX.get(left_family)
    right_capability = PATCH_FAMILY_CAPABILITY_MATRIX.get(right_family)
    if supported:
        phase: SurfaceBooleanUnsupportedPhase | str = "intersection-kernel"
        support_state: SurfaceBooleanSupportState = (
            "exact"
            if left_family in ANALYTIC_SURFACE_CSG_FAMILIES and right_family in ANALYTIC_SURFACE_CSG_FAMILIES
            else "declared-tolerance"
        )
        required_future_capability = None
    elif left_capability is None or right_capability is None:
        phase = "operand-family-eligibility"
        support_state = "unsupported"
        required_future_capability = _surface_boolean_required_future_capability(
            operation, left_family, right_family
        )
    elif left_capability.support_phase == "planned" or right_capability.support_phase == "planned":
        phase = "operand-family-eligibility"
        support_state = (
            "unsupported"
            if left_family in SAMPLED_SURFACE_CSG_FAMILIES or right_family in SAMPLED_SURFACE_CSG_FAMILIES
            else "not-yet-implemented"
        )
        required_future_capability = _surface_boolean_required_future_capability(
            operation, left_family, right_family
        )
    else:
        phase = "intersection-kernel"
        support_state = "not-yet-implemented"
        required_future_capability = _surface_boolean_required_future_capability(
            operation, left_family, right_family
        )
    return SurfaceBooleanFamilyPairSupport(
        operation=operation,
        left_family=left_family,
        right_family=right_family,
        supported=supported,
        phase=phase,
        support_state=support_state,
        required_future_capability=required_future_capability,
    )


SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX: dict[tuple[SurfaceBooleanOperation, str, str], SurfaceBooleanFamilyPairSupport] = {
    (operation, left_family, right_family): _surface_boolean_support_record(operation, left_family, right_family)
    for operation in SURFACE_BOOLEAN_OPERATIONS
    for left_family in PATCH_FAMILY_CAPABILITY_MATRIX
    for right_family in PATCH_FAMILY_CAPABILITY_MATRIX
}


def build_surface_csg_solver_registry(
    support_matrix: Mapping[
        tuple[SurfaceBooleanOperation, str, str],
        SurfaceBooleanFamilyPairSupport,
    ] | None = None,
    *,
    family_capability_matrix: Mapping[str, object] | None = None,
    operations: Sequence[SurfaceBooleanOperation] = SURFACE_BOOLEAN_OPERATIONS,
) -> SurfaceCSGSolverRegistryRecord:
    """Build the auditable surface CSG family-pair solver registry."""

    matrix = support_matrix if support_matrix is not None else SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX
    families = tuple(sorted((family_capability_matrix or PATCH_FAMILY_CAPABILITY_MATRIX).keys()))
    normalized_operations = tuple(operations)
    diagnostics: list[SurfaceCSGSolverRegistryDiagnostic] = []
    records: list[SurfaceBooleanFamilyPairSupport] = []
    expected_keys = {
        (operation, left_family, right_family)
        for operation in normalized_operations
        for left_family in families
        for right_family in families
    }
    for key in sorted(expected_keys):
        support = matrix.get(key)
        if support is None:
            diagnostics.append(
                SurfaceCSGSolverRegistryDiagnostic(
                    code="missing-pair",
                    operation=key[0],
                    left_family=key[1],
                    right_family=key[2],
                    message=(
                        f"Surface CSG solver registry is missing support classification for "
                        f"{key[0]} {key[1]}/{key[2]}."
                    ),
                )
            )
            continue
        if (support.operation, support.left_family, support.right_family) != key:
            diagnostics.append(
                SurfaceCSGSolverRegistryDiagnostic(
                    code="record-key-mismatch",
                    operation=key[0],
                    left_family=key[1],
                    right_family=key[2],
                    message=(
                        f"Surface CSG solver registry key {key!r} does not match record "
                        f"{(support.operation, support.left_family, support.right_family)!r}."
                    ),
                )
            )
        if not support.supported and not support.required_future_capability:
            diagnostics.append(
                SurfaceCSGSolverRegistryDiagnostic(
                    code="missing-future-capability",
                    operation=support.operation,
                    left_family=support.left_family,
                    right_family=support.right_family,
                    message=(
                        f"Unsupported surface CSG pair {support.operation} "
                        f"{support.left_family}/{support.right_family} must name required future capability."
                    ),
                )
            )
        records.append(support)
    for operation, left_family, right_family in sorted(set(matrix).difference(expected_keys)):
        diagnostics.append(
            SurfaceCSGSolverRegistryDiagnostic(
                code="unknown-pair",
                operation=operation,
                left_family=left_family,
                right_family=right_family,
                message=(
                    f"Surface CSG solver registry contains unknown support classification for "
                    f"{operation} {left_family}/{right_family}."
                ),
            )
        )
    return SurfaceCSGSolverRegistryRecord(
        operations=normalized_operations,
        families=families,
        support_records=tuple(records),
        diagnostics=tuple(diagnostics),
    )


def assert_surface_csg_solver_registry_complete(
    registry: SurfaceCSGSolverRegistryRecord | None = None,
) -> SurfaceCSGSolverRegistryRecord:
    """Return the registry or raise when operation/family-pair coverage is incomplete."""

    checked = registry if registry is not None else build_surface_csg_solver_registry()
    if checked.diagnostics:
        joined = "; ".join(
            f"{diagnostic.code}:{diagnostic.operation}:{diagnostic.left_family}/{diagnostic.right_family}"
            for diagnostic in checked.diagnostics
        )
        raise ValueError(f"Surface CSG solver registry failed coverage checks: {joined}")
    return checked


SURFACE_CSG_SOLVER_REGISTRY = assert_surface_csg_solver_registry_complete()


def _normalize_curve_point(point: Sequence[float]) -> tuple[float, float, float]:
    if len(point) != 3:
        raise ValueError("CSG curve points must be 3D coordinates.")
    normalized = tuple(float(component) for component in point)
    if any(not np.isfinite(component) for component in normalized):
        raise ValueError("CSG curve points must contain finite coordinates.")
    return normalized


def _normalize_curve_point_uv(point: Sequence[float]) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError("CSG patch-local curve points must be UV coordinates.")
    normalized = (float(point[0]), float(point[1]))
    if any(not np.isfinite(component) for component in normalized):
        raise ValueError("CSG patch-local curve points must contain finite coordinates.")
    return normalized


def _snap_scalar(value: float, tolerance: float) -> float:
    return round(float(value) / tolerance) * tolerance


def normalize_surface_csg_tolerance_policy(
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> SurfaceCSGTolerancePolicy:
    """Normalize a CSG tolerance policy or mapping to the shared policy record."""

    if policy is None:
        return DEFAULT_SURFACE_CSG_TOLERANCE_POLICY
    if isinstance(policy, SurfaceCSGTolerancePolicy):
        return policy
    return SurfaceCSGTolerancePolicy(
        snap_tolerance=policy.get("snap_tolerance", DEFAULT_SURFACE_CSG_TOLERANCE_POLICY.snap_tolerance),
        equality_tolerance=policy.get("equality_tolerance", DEFAULT_SURFACE_CSG_TOLERANCE_POLICY.equality_tolerance),
        degeneracy_tolerance=policy.get("degeneracy_tolerance", DEFAULT_SURFACE_CSG_TOLERANCE_POLICY.degeneracy_tolerance),
        domain_tolerance=policy.get("domain_tolerance", DEFAULT_SURFACE_CSG_TOLERANCE_POLICY.domain_tolerance),
    )


def make_surface_csg_curve(
    kind: SurfaceCSGCurveKind,
    points_3d: Sequence[Sequence[float]],
    *,
    parameters: Sequence[tuple[str, float]] = (),
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> SurfaceCSGCurvePrimitive:
    """Construct a deterministic surface-native CSG curve primitive."""

    curve = SurfaceCSGCurvePrimitive(kind=kind, points_3d=tuple(tuple(point) for point in points_3d), parameters=tuple(parameters))
    diagnostics = validate_surface_csg_curve(curve, policy=policy)
    if diagnostics:
        raise ValueError("; ".join(diagnostic.message for diagnostic in diagnostics))
    return curve


def make_surface_csg_line_curve(
    start: Sequence[float],
    end: Sequence[float],
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> SurfaceCSGCurvePrimitive:
    """Construct a bounded line curve for CSG intersections."""

    return make_surface_csg_curve("line", (start, end), policy=policy)


def validate_surface_csg_curve(
    curve: SurfaceCSGCurvePrimitive,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> tuple[SurfaceCSGToleranceDiagnostic, ...]:
    """Return explicit tolerance diagnostics for one CSG curve primitive."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    points = np.asarray(curve.points_3d, dtype=float)
    diagnostics: list[SurfaceCSGToleranceDiagnostic] = []
    if curve.kind == "line":
        span = float(np.linalg.norm(points[-1] - points[0]))
    else:
        span = float(np.max(np.linalg.norm(points - points[0], axis=1)))
    if span <= normalized_policy.degeneracy_tolerance:
        diagnostics.append(
            SurfaceCSGToleranceDiagnostic(
                code="degenerate-curve",
                curve_id=surface_csg_curve_digest(curve, policy=normalized_policy),
                message="CSG curve length is at or below the degeneracy tolerance.",
            )
        )
    if curve.kind in {"arc", "conic"} and not curve.parameters:
        diagnostics.append(
            SurfaceCSGToleranceDiagnostic(
                code="ambiguous-curve",
                curve_id=surface_csg_curve_digest(curve, policy=normalized_policy),
                message=f"CSG {curve.kind} curves require parameters to disambiguate their analytic form.",
            )
        )
    return tuple(diagnostics)


def surface_csg_curve_key(
    curve: SurfaceCSGCurvePrimitive,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> tuple[object, ...]:
    """Return a deterministic equality/deduplication key for a CSG curve."""

    payload = curve.canonical_payload(normalize_surface_csg_tolerance_policy(policy))
    return (
        payload["kind"],
        tuple(payload["parameters"]),
        tuple(tuple(point) for point in payload["points_3d"]),
    )


def surface_csg_curve_digest(
    curve: SurfaceCSGCurvePrimitive,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> str:
    """Return a stable digest for deterministic curve hashing."""

    payload = curve.canonical_payload(normalize_surface_csg_tolerance_policy(policy))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def surface_csg_curves_equal(
    left: SurfaceCSGCurvePrimitive,
    right: SurfaceCSGCurvePrimitive,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> bool:
    """Return whether two CSG curves are equal under the shared tolerance policy."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    return surface_csg_curve_key(left, policy=normalized_policy) == surface_csg_curve_key(right, policy=normalized_policy)


def sort_surface_csg_curves(
    curves: Iterable[SurfaceCSGCurvePrimitive],
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> tuple[SurfaceCSGCurvePrimitive, ...]:
    """Return CSG curves in deterministic canonical-key order."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    return tuple(sorted(curves, key=lambda curve: surface_csg_curve_key(curve, policy=normalized_policy)))


def _surface_csg_mapping_diagnostic(
    code: SurfaceCSGToleranceDiagnosticCode,
    message: str,
    *,
    patch_ref: SurfaceBooleanPatchRef,
    curve: SurfaceCSGCurvePrimitive,
    policy: SurfaceCSGTolerancePolicy,
) -> SurfaceCSGCurveMappingDiagnostic:
    return SurfaceCSGCurveMappingDiagnostic(
        code=code,
        message=message,
        patch=patch_ref,
        source_curve_digest=surface_csg_curve_digest(curve, policy=policy),
    )


def validate_surface_csg_patch_local_curve_domain(
    curve: SurfaceCSGPatchLocalCurve,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> tuple[SurfaceCSGCurveMappingDiagnostic, ...]:
    """Validate that a patch-local CSG curve stays inside its declared domain."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    u0, u1, v0, v1 = curve.domain_bounds
    diagnostics: list[SurfaceCSGCurveMappingDiagnostic] = []
    for u, v in curve.points_uv:
        if (
            u < u0 - normalized_policy.domain_tolerance
            or u > u1 + normalized_policy.domain_tolerance
            or v < v0 - normalized_policy.domain_tolerance
            or v > v1 + normalized_policy.domain_tolerance
        ):
            diagnostics.append(
                SurfaceCSGCurveMappingDiagnostic(
                    code="outside-domain",
                    message="Mapped CSG curve point is outside the patch parameter domain.",
                    patch=curve.patch,
                    source_curve_digest=curve.source_curve_digest,
                )
            )
            break
    return tuple(diagnostics)


def _inverse_transform_point(matrix: np.ndarray, point: Sequence[float]) -> np.ndarray:
    hom = np.ones(4, dtype=float)
    hom[:3] = np.asarray(point, dtype=float).reshape(3)
    local = np.linalg.inv(matrix) @ hom
    return local[:3]


def _axis_frame(axis_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(reference, axis_direction))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    x_axis = reference - axis_direction * float(np.dot(reference, axis_direction))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(axis_direction, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return x_axis, y_axis


def _rotate_point_about_axis(point: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    rel = point - origin
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    return origin + rel * cos_a + np.cross(axis, rel) * sin_a + axis * float(np.dot(axis, rel)) * (1.0 - cos_a)


def _polyline_closest_normalized_parameter(points: np.ndarray, target: np.ndarray) -> float:
    if len(points) == 1:
        return 0.0
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(np.sum(segment_lengths))
    if total <= 0.0:
        return 0.0
    best_distance = float("inf")
    best_length = 0.0
    traversed = 0.0
    for index, length in enumerate(segment_lengths):
        if length <= 0.0:
            continue
        start = points[index]
        end = points[index + 1]
        direction = end - start
        t = float(np.clip(np.dot(target - start, direction) / (length * length), 0.0, 1.0))
        candidate = start + direction * t
        distance = float(np.linalg.norm(candidate - target))
        if distance < best_distance:
            best_distance = distance
            best_length = traversed + length * t
        traversed += length
    return best_length / total


def _map_revolution_point_to_uv(
    patch: RevolutionSurfacePatch,
    point: Sequence[float],
    *,
    policy: SurfaceCSGTolerancePolicy,
) -> tuple[float, float] | SurfaceCSGToleranceDiagnosticCode:
    local_point = _inverse_transform_point(patch.transform_matrix, point)
    axis = patch.axis_direction
    origin = patch.axis_origin
    radial = local_point - origin
    height = float(np.dot(radial, axis))
    radial -= axis * height
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm <= policy.degeneracy_tolerance:
        return "ambiguous-curve"
    x_axis, y_axis = _axis_frame(axis)
    angle = float(np.arctan2(np.dot(radial, y_axis), np.dot(radial, x_axis)))
    start = float(np.deg2rad(patch.start_angle_deg))
    sweep = float(np.deg2rad(patch.sweep_angle_deg))
    u_norm = (angle - start) / sweep
    if sweep > 0.0:
        while u_norm < -policy.domain_tolerance:
            u_norm += (2.0 * np.pi) / sweep
        while u_norm > 1.0 + policy.domain_tolerance:
            u_norm -= (2.0 * np.pi) / sweep
    unrotated = _rotate_point_about_axis(local_point, origin, axis, -(start + sweep * u_norm))
    v_norm = _polyline_closest_normalized_parameter(patch.profile_curve, unrotated)
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    return (float(u0 + u_norm * (u1 - u0)), float(v0 + v_norm * (v1 - v0)))


def map_surface_csg_curve_to_patch_local(
    curve: SurfaceCSGCurvePrimitive,
    patch_ref: SurfaceBooleanPatchRef,
    patch: PlanarSurfacePatch | RevolutionSurfacePatch,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> SurfaceCSGPatchLocalCurveMappingResult:
    """Map a 3D CSG curve into a deterministic patch-local UV curve."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    source_digest = surface_csg_curve_digest(curve, policy=normalized_policy)
    diagnostics: list[SurfaceCSGCurveMappingDiagnostic] = []
    points_uv: list[tuple[float, float]] = []
    for point in curve.points_3d:
        if isinstance(patch, PlanarSurfacePatch):
            points_uv.append(_planar_patch_point_to_uv(patch, point))
        elif isinstance(patch, RevolutionSurfacePatch):
            mapped = _map_revolution_point_to_uv(patch, point, policy=normalized_policy)
            if isinstance(mapped, str):
                diagnostics.append(
                    _surface_csg_mapping_diagnostic(
                        mapped,
                        "Revolution CSG curve mapping is singular or periodic-ambiguous at the axis.",
                        patch_ref=patch_ref,
                        curve=curve,
                        policy=normalized_policy,
                    )
                )
            else:
                points_uv.append(mapped)
        else:
            diagnostics.append(
                _surface_csg_mapping_diagnostic(
                    "ambiguous-curve",
                    f"Patch family {patch.family!r} does not expose deterministic CSG curve mapping yet.",
                    patch_ref=patch_ref,
                    curve=curve,
                    policy=normalized_policy,
                )
            )
    if diagnostics:
        return SurfaceCSGPatchLocalCurveMappingResult(source_curve=curve, patch=patch_ref, diagnostics=tuple(diagnostics))
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    local_curve = SurfaceCSGPatchLocalCurve(
        source_curve_digest=source_digest,
        patch=patch_ref,
        points_uv=tuple(points_uv),
        domain_bounds=(float(u0), float(u1), float(v0), float(v1)),
    )
    domain_diagnostics = validate_surface_csg_patch_local_curve_domain(local_curve, policy=normalized_policy)
    if domain_diagnostics:
        return SurfaceCSGPatchLocalCurveMappingResult(
            source_curve=curve,
            patch=patch_ref,
            curve=None,
            diagnostics=domain_diagnostics,
        )
    return SurfaceCSGPatchLocalCurveMappingResult(source_curve=curve, patch=patch_ref, curve=local_curve)


def _surface_csg_planar_relation_diagnostic(
    relation: SurfaceCSGPlanarRelation,
    message: str,
    first_ref: SurfaceBooleanPatchRef,
    second_ref: SurfaceBooleanPatchRef,
) -> SurfaceCSGPlanarRelationDiagnostic:
    return SurfaceCSGPlanarRelationDiagnostic(
        relation=relation,
        message=message,
        first_patch=first_ref,
        second_patch=second_ref,
    )


def _planar_patch_frame(patch: PlanarSurfacePatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u0, v0 = patch.domain.u_range[0], patch.domain.v_range[0]
    origin = patch.point_at(u0, v0)
    du, dv = patch.derivatives_at(u0, v0)
    normal = np.cross(du, dv)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("Planar patch has degenerate parameter frame.")
    return origin, du, dv


def _planar_patch_normal(patch: PlanarSurfacePatch) -> np.ndarray:
    _origin, du, dv = _planar_patch_frame(patch)
    normal = np.cross(du, dv)
    return normal / float(np.linalg.norm(normal))


def _line_parameter_interval_for_planar_patch(
    patch: PlanarSurfacePatch,
    line_point: np.ndarray,
    line_direction: np.ndarray,
    *,
    policy: SurfaceCSGTolerancePolicy,
) -> tuple[float, float] | None:
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    uv_point = np.asarray(_planar_patch_point_to_uv(patch, line_point), dtype=float)
    uv_direction = np.asarray(_planar_patch_point_to_uv(patch, line_point + line_direction), dtype=float) - uv_point
    low = -float("inf")
    high = float("inf")
    for value, direction, minimum, maximum in (
        (uv_point[0], uv_direction[0], float(u0), float(u1)),
        (uv_point[1], uv_direction[1], float(v0), float(v1)),
    ):
        if abs(direction) <= policy.degeneracy_tolerance:
            if value < minimum - policy.domain_tolerance or value > maximum + policy.domain_tolerance:
                return None
            continue
        t0 = (minimum - value) / direction
        t1 = (maximum - value) / direction
        low = max(low, min(t0, t1))
        high = min(high, max(t0, t1))
    if high < low - policy.domain_tolerance:
        return None
    return (low, high)


def intersect_planar_linear_patch_pair(
    first_ref: SurfaceBooleanPatchRef,
    first_patch: PlanarSurfacePatch | RuledSurfacePatch,
    second_ref: SurfaceBooleanPatchRef,
    second_patch: PlanarSurfacePatch | RuledSurfacePatch,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> SurfaceCSGAnalyticIntersectionRecord:
    """Intersect a low-order planar/linear patch pair without mesh fallback."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    if isinstance(first_patch, RuledSurfacePatch) or isinstance(second_patch, RuledSurfacePatch):
        return SurfaceCSGAnalyticIntersectionRecord(
            first_patch=first_ref,
            second_patch=second_ref,
            relation="unsupported-linear",
            diagnostics=(
                _surface_csg_planar_relation_diagnostic(
                    "unsupported-linear",
                    "Ruled patch analytic intersection is gated for the later linear intersection implementation.",
                    first_ref,
                    second_ref,
                ),
            ),
        )
    first_normal = _planar_patch_normal(first_patch)
    second_normal = _planar_patch_normal(second_patch)
    first_origin = first_patch.point_at(first_patch.domain.u_range[0], first_patch.domain.v_range[0])
    second_origin = second_patch.point_at(second_patch.domain.u_range[0], second_patch.domain.v_range[0])
    direction = np.cross(first_normal, second_normal)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= normalized_policy.degeneracy_tolerance:
        offset = abs(float(np.dot(first_normal, second_origin - first_origin)))
        relation: SurfaceCSGPlanarRelation = "coincident" if offset <= normalized_policy.equality_tolerance else "parallel"
        return SurfaceCSGAnalyticIntersectionRecord(
            first_patch=first_ref,
            second_patch=second_ref,
            relation=relation,
            diagnostics=(
                _surface_csg_planar_relation_diagnostic(
                    relation,
                    f"Planar patches are {relation}; no unique intersection curve is emitted.",
                    first_ref,
                    second_ref,
                ),
            ),
        )
    direction /= direction_norm
    system = np.vstack((first_normal, second_normal, direction))
    rhs = np.array(
        [
            float(np.dot(first_normal, first_origin)),
            float(np.dot(second_normal, second_origin)),
            0.0,
        ],
        dtype=float,
    )
    try:
        line_point = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return SurfaceCSGAnalyticIntersectionRecord(
            first_patch=first_ref,
            second_patch=second_ref,
            relation="parallel",
            diagnostics=(
                _surface_csg_planar_relation_diagnostic(
                    "parallel",
                    "Planar patch intersection system is singular.",
                    first_ref,
                    second_ref,
                ),
            ),
        )
    first_interval = _line_parameter_interval_for_planar_patch(
        first_patch, line_point, direction, policy=normalized_policy
    )
    second_interval = _line_parameter_interval_for_planar_patch(
        second_patch, line_point, direction, policy=normalized_policy
    )
    if first_interval is None or second_interval is None:
        return SurfaceCSGAnalyticIntersectionRecord(
            first_patch=first_ref,
            second_patch=second_ref,
            relation="disjoint",
            diagnostics=(
                _surface_csg_planar_relation_diagnostic(
                    "disjoint",
                    "Finite planar patch domains do not overlap along the plane intersection line.",
                    first_ref,
                    second_ref,
                ),
            ),
        )
    seg_min = max(first_interval[0], second_interval[0])
    seg_max = min(first_interval[1], second_interval[1])
    if seg_max - seg_min <= normalized_policy.degeneracy_tolerance:
        return SurfaceCSGAnalyticIntersectionRecord(
            first_patch=first_ref,
            second_patch=second_ref,
            relation="touching",
            diagnostics=(
                _surface_csg_planar_relation_diagnostic(
                    "touching",
                    "Finite planar patch domains touch without a non-degenerate intersection segment.",
                    first_ref,
                    second_ref,
                ),
            ),
        )
    start = line_point + direction * seg_min
    end = line_point + direction * seg_max
    curve = make_surface_csg_line_curve(start, end, policy=normalized_policy)
    first_mapping = map_surface_csg_curve_to_patch_local(curve, first_ref, first_patch, policy=normalized_policy)
    second_mapping = map_surface_csg_curve_to_patch_local(curve, second_ref, second_patch, policy=normalized_policy)
    if not first_mapping.supported or first_mapping.curve is None or not second_mapping.supported or second_mapping.curve is None:
        return SurfaceCSGAnalyticIntersectionRecord(
            first_patch=first_ref,
            second_patch=second_ref,
            relation="disjoint",
            diagnostics=(
                _surface_csg_planar_relation_diagnostic(
                    "disjoint",
                    "Analytic line segment could not be mapped into both patch domains.",
                    first_ref,
                    second_ref,
                ),
            ),
        )
    return SurfaceCSGAnalyticIntersectionRecord(
        first_patch=first_ref,
        second_patch=second_ref,
        relation="crossing",
        curve=curve,
        patch_local_curves=(first_mapping.curve, second_mapping.curve),
    )


def _surface_csg_conic_diagnostic(
    code: str,
    message: str,
    first_ref: SurfaceBooleanPatchRef,
    second_ref: SurfaceBooleanPatchRef,
) -> SurfaceCSGConicDiagnostic:
    return SurfaceCSGConicDiagnostic(
        code=code,
        message=message,
        first_patch=first_ref,
        second_patch=second_ref,
    )


def _revolution_profile_radius_height(patch: RevolutionSurfacePatch) -> tuple[np.ndarray, np.ndarray]:
    axis = patch.axis_direction
    rel = patch.profile_curve - patch.axis_origin
    heights = rel @ axis
    radial = rel - np.outer(heights, axis)
    radii = np.linalg.norm(radial, axis=1)
    return radii.astype(float), heights.astype(float)


def _revolution_radius_at_height(
    patch: RevolutionSurfacePatch,
    height: float,
    *,
    policy: SurfaceCSGTolerancePolicy,
) -> tuple[str, float] | None:
    radii, heights = _revolution_profile_radius_height(patch)
    h_min = float(np.min(heights))
    h_max = float(np.max(heights))
    if height < h_min - policy.domain_tolerance or height > h_max + policy.domain_tolerance:
        return None
    if np.allclose(radii, radii[0], atol=policy.equality_tolerance):
        return ("cylinder", float(radii[0]))
    if len(radii) >= 3 and abs(float(radii[0])) <= policy.equality_tolerance and abs(float(radii[-1])) <= policy.equality_tolerance:
        center_height = (h_min + h_max) * 0.5
        sphere_radius = (h_max - h_min) * 0.5
        offset = height - center_height
        remaining = sphere_radius * sphere_radius - offset * offset
        if remaining < -policy.equality_tolerance:
            return None
        return ("sphere", float(np.sqrt(max(0.0, remaining))))
    if len(radii) >= 2 and (
        abs(float(radii[0])) <= policy.equality_tolerance
        or abs(float(radii[-1])) <= policy.equality_tolerance
    ):
        order = np.argsort(heights)
        radius = float(np.interp(height, heights[order], radii[order]))
        return ("cone", radius)
    return None


def _circle_points(center: np.ndarray, axis: np.ndarray, radius: float) -> tuple[tuple[float, float, float], ...]:
    x_axis, y_axis = _axis_frame(axis)
    points = []
    for angle in (0.0, np.pi * 0.5, np.pi, np.pi * 1.5, np.pi * 2.0):
        point = center + radius * (np.cos(angle) * x_axis + np.sin(angle) * y_axis)
        points.append((float(point[0]), float(point[1]), float(point[2])))
    return tuple(points)


def intersect_planar_revolution_patch_pair(
    plane_ref: SurfaceBooleanPatchRef,
    plane_patch: PlanarSurfacePatch,
    revolution_ref: SurfaceBooleanPatchRef,
    revolution_patch: RevolutionSurfacePatch,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> SurfaceCSGRevolutionIntersectionRecord:
    """Intersect a plane with a supported revolution/conic patch without mesh fallback."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    plane_normal = _planar_patch_normal(plane_patch)
    plane_origin = plane_patch.point_at(plane_patch.domain.u_range[0], plane_patch.domain.v_range[0])
    axis = revolution_patch.axis_direction
    parallel = abs(float(np.dot(plane_normal, axis)))
    if 1.0 - parallel > normalized_policy.equality_tolerance:
        return SurfaceCSGRevolutionIntersectionRecord(
            first_patch=plane_ref,
            second_patch=revolution_ref,
            conic_kind="unsupported",
            diagnostics=(
                _surface_csg_conic_diagnostic(
                    "unsupported-oblique-plane",
                    "Only axis-perpendicular plane/revolution conic intersections are supported in this slice.",
                    plane_ref,
                    revolution_ref,
                ),
            ),
        )
    height = float(np.dot(plane_origin - revolution_patch.axis_origin, axis))
    radius_record = _revolution_radius_at_height(revolution_patch, height, policy=normalized_policy)
    if radius_record is None:
        return SurfaceCSGRevolutionIntersectionRecord(
            first_patch=plane_ref,
            second_patch=revolution_ref,
            conic_kind="unsupported",
            diagnostics=(
                _surface_csg_conic_diagnostic(
                    "outside-profile-domain",
                    "Plane lies outside the revolution profile height domain.",
                    plane_ref,
                    revolution_ref,
                ),
            ),
        )
    profile_kind, radius = radius_record
    if radius <= normalized_policy.degeneracy_tolerance:
        return SurfaceCSGRevolutionIntersectionRecord(
            first_patch=plane_ref,
            second_patch=revolution_ref,
            conic_kind="unsupported",
            diagnostics=(
                _surface_csg_conic_diagnostic(
                    "tangent-or-singular-axis",
                    "Plane/revolution intersection degenerates at the revolution axis.",
                    plane_ref,
                    revolution_ref,
                ),
            ),
        )
    center = revolution_patch.axis_origin + axis * height
    points = _circle_points(center, axis, radius)
    curve = make_surface_csg_curve(
        "arc",
        points,
        parameters=(
            ("axis_x", float(axis[0])),
            ("axis_y", float(axis[1])),
            ("axis_z", float(axis[2])),
            ("center_x", float(center[0])),
            ("center_y", float(center[1])),
            ("center_z", float(center[2])),
            ("radius", radius),
        ),
        policy=normalized_policy,
    )
    plane_mapping = map_surface_csg_curve_to_patch_local(curve, plane_ref, plane_patch, policy=normalized_policy)
    revolution_mapping = map_surface_csg_curve_to_patch_local(
        curve, revolution_ref, revolution_patch, policy=normalized_policy
    )
    if not plane_mapping.supported or plane_mapping.curve is None or not revolution_mapping.supported or revolution_mapping.curve is None:
        return SurfaceCSGRevolutionIntersectionRecord(
            first_patch=plane_ref,
            second_patch=revolution_ref,
            conic_kind="unsupported",
            diagnostics=(
                _surface_csg_conic_diagnostic(
                    "mapping-failed",
                    "Plane/revolution conic could not be mapped into both patch-local domains.",
                    plane_ref,
                    revolution_ref,
                ),
            ),
        )
    return SurfaceCSGRevolutionIntersectionRecord(
        first_patch=plane_ref,
        second_patch=revolution_ref,
        conic_kind="circle" if profile_kind in {"cylinder", "sphere", "cone"} else "unsupported",
        curve=curve,
        patch_local_curves=(plane_mapping.curve, revolution_mapping.curve),
    )


def intersect_axis_compatible_revolution_pair(
    first_ref: SurfaceBooleanPatchRef,
    first_patch: RevolutionSurfacePatch,
    second_ref: SurfaceBooleanPatchRef,
    second_patch: RevolutionSurfacePatch,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> SurfaceCSGRevolutionIntersectionRecord:
    """Gate axis-compatible revolution/revolution pairs before general algebra."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    same_axis = np.allclose(first_patch.axis_direction, second_patch.axis_direction, atol=normalized_policy.equality_tolerance)
    same_origin = np.linalg.norm(first_patch.axis_origin - second_patch.axis_origin) <= normalized_policy.equality_tolerance
    if same_axis and same_origin:
        code = "axis-compatible-revolution-gate"
        message = "Axis-compatible revolution/revolution intersection is recognized but awaits operation-specific curve selection."
    else:
        code = "unsupported-general-revolution"
        message = "General revolution/revolution intersection is outside the supported solver boundary."
    return SurfaceCSGRevolutionIntersectionRecord(
        first_patch=first_ref,
        second_patch=second_ref,
        conic_kind="unsupported",
        diagnostics=(_surface_csg_conic_diagnostic(code, message, first_ref, second_ref),),
    )


def _ensure_backend(backend: BooleanBackend) -> None:
    if backend not in {"manifold", "surface"}:
        raise ValueError(f"Unsupported backend '{backend}'. Only 'manifold' and 'surface' are available right now.")


def _load_manifold():
    try:
        from manifold3d import Manifold, Mesh as ManifoldMesh
    except ImportError as exc:  # pragma: no cover - runtime dep
        raise BooleanOperationError(
            "manifold3d is required for boolean operations. Install it with `pip install manifold3d`."
        ) from exc
    return Manifold, ManifoldMesh


def _mesh_from_manifold(manifold_mesh) -> Mesh:
    vertices = None
    faces = None
    for attr in ("vertices", "vert_properties", "verts"):
        if hasattr(manifold_mesh, attr):
            vertices = np.asarray(getattr(manifold_mesh, attr), dtype=float)
            break
    for attr in ("triangles", "tri_verts", "faces"):
        if hasattr(manifold_mesh, attr):
            faces = np.asarray(getattr(manifold_mesh, attr), dtype=int)
            break
    if vertices is None or faces is None:
        raise BooleanOperationError("manifold3d returned an unexpected mesh format.")
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise BooleanOperationError("manifold3d returned invalid vertex data.")
    if vertices.shape[1] > 3:
        vertices = vertices[:, :3]
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise BooleanOperationError("manifold3d returned non-triangular faces.")
    return Mesh(vertices, faces)


def _manifold_from_mesh(mesh: Mesh, face_id: int | None = None):
    Manifold, ManifoldMesh = _load_manifold()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    face_ids = None
    if face_id is not None:
        face_ids = np.full(mesh.n_faces, face_id, dtype=np.uint32)
    try:
        manifold_mesh = ManifoldMesh(vertices, faces, face_id=face_ids)
    except TypeError:
        try:
            manifold_mesh = ManifoldMesh(vertices=vertices, triangles=faces, face_id=face_ids)
        except TypeError as exc:  # pragma: no cover - defensive
            raise BooleanOperationError("Unable to build manifold mesh from input data.") from exc
    try:
        return Manifold(manifold_mesh)
    except Exception as exc:
        raise BooleanOperationError("manifold3d failed to create a solid from the provided mesh.") from exc


def _flatten_meshes(meshes: Iterable[Mesh | MeshGroup]) -> list[Mesh]:
    flattened: list[Mesh] = []
    for mesh in meshes:
        if isinstance(mesh, MeshGroup):
            flattened.extend(mesh.to_meshes())
        elif isinstance(mesh, Mesh):
            flattened.append(mesh)
        else:
            raise TypeError("Boolean operations require Mesh or MeshGroup inputs.")
    return flattened


def _check_mesh(mesh: Mesh) -> Mesh:
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("Mesh faces must be a (N, 3) triangle array.")
    if mesh.n_faces == 0:
        raise ValueError("Mesh has no faces.")
    if mesh.faces.min() < 0 or mesh.faces.max() >= mesh.n_vertices:
        raise ValueError("Mesh faces contain out-of-range vertex indices.")
    analyze_mesh(mesh)
    issues = mesh.analysis.issues() if mesh.analysis else []
    if issues:
        warnings.warn(
            f"Mesh analysis warnings: {', '.join(issues)}",
            RuntimeWarning,
        )
    return mesh


def _combine_color(result: Mesh, sources: list[Mesh]) -> None:
    for mesh in sources:
        color = get_mesh_color(mesh)
        if color is not None:
            set_mesh_color(result, (*color[0], color[1]))
            return


def _resolve_mesh_rgba(mesh: Mesh) -> tuple[float, float, float, float] | None:
    if mesh.color is not None:
        return mesh.color
    if mesh.face_colors is None or mesh.face_colors.size == 0:
        return None
    colors = np.asarray(mesh.face_colors, dtype=float)
    if colors.ndim != 2 or colors.shape[1] < 3:
        return None
    rgb = colors[:, :3].mean(axis=0)
    alpha = colors[:, 3].mean() if colors.shape[1] >= 4 else 1.0
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))


def _apply_face_colors(result: Mesh, face_ids: np.ndarray, color_map: dict[int, tuple[float, float, float, float] | None]) -> None:
    if face_ids.size == 0:
        return
    face_colors = np.zeros((face_ids.size, 4), dtype=float)
    for idx, face_id in enumerate(face_ids):
        rgba = color_map.get(int(face_id))
        if rgba is None:
            rgba = (0.8, 0.8, 0.8, 1.0)
        face_colors[idx] = rgba
    result.face_colors = face_colors


def _apply_boolean(
    meshes: Iterable[Mesh],
    operation: str,
) -> Mesh:
    meshes_list = [_check_mesh(mesh) for mesh in meshes]
    if not meshes_list:
        raise ValueError(f"boolean_{operation} requires at least one mesh.")

    color_map: dict[int, tuple[float, float, float, float] | None] = {}
    explicit_color = False
    manifold_meshes = []
    for idx, mesh in enumerate(meshes_list, start=1):
        color = _resolve_mesh_rgba(mesh)
        if color is not None:
            explicit_color = True
        color_map[idx] = color
        manifold_meshes.append(_manifold_from_mesh(mesh, face_id=idx))

    base = manifold_meshes[0]
    for other in manifold_meshes[1:]:
        if hasattr(base, operation):
            base = getattr(base, operation)(other)
        elif operation == "union" and hasattr(base, "__add__"):
            base = base + other
        elif operation == "difference" and hasattr(base, "__sub__"):
            base = base - other
        elif operation == "intersection":
            if hasattr(base, "__and__"):
                base = base & other
            elif hasattr(base, "__sub__"):
                # Intersection = A - (A - B) when direct op is unavailable.
                base = base - (base - other)
            else:
                raise BooleanOperationError(
                    "manifold3d does not support intersection on this version."
                )
        else:
            raise BooleanOperationError(f"manifold3d does not support '{operation}' on this version.")

    if hasattr(base, "to_mesh"):
        result_mesh = base.to_mesh()
    elif hasattr(base, "mesh"):
        result_mesh = base.mesh
    else:
        raise BooleanOperationError("manifold3d returned an unexpected result type.")

    result = _mesh_from_manifold(result_mesh)
    if explicit_color and hasattr(result_mesh, "face_id"):
        face_ids = np.asarray(result_mesh.face_id, dtype=int)
        if face_ids.shape[0] == result.n_faces:
            _apply_face_colors(result, face_ids, color_map)
        _combine_color(result, meshes_list)
    else:
        _combine_color(result, meshes_list)
    return result


def _classify_surface_body(body: SurfaceBody) -> Literal["open", "closed"]:
    # Local import keeps the surfaced preparation layer decoupled from the
    # mesh-primary boolean path while still using the canonical shell truth.
    from .tessellation import _classify_body

    return _classify_body(body)


def _aabb_overlap(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    return (
        max(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        min(left[3], right[3]),
        max(left[4], right[4]),
        min(left[5], right[5]),
    )


def _surface_boolean_body_relation(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
    *,
    epsilon: float = 1e-9,
) -> SurfaceBooleanBodyRelation:
    overlap = _aabb_overlap(left, right)
    spans = (
        overlap[1] - overlap[0],
        overlap[3] - overlap[2],
        overlap[5] - overlap[4],
    )
    if any(span < -epsilon for span in spans):
        return "disjoint"
    if all(abs(left[idx] - right[idx]) <= epsilon for idx in range(6)):
        return "equal"
    if any(abs(span) <= epsilon for span in spans):
        return "touching"

    def _contains(container: tuple[float, ...], candidate: tuple[float, ...]) -> bool:
        return (
            container[0] <= candidate[0] + epsilon
            and container[1] >= candidate[1] - epsilon
            and container[2] <= candidate[2] + epsilon
            and container[3] >= candidate[3] - epsilon
            and container[4] <= candidate[4] + epsilon
            and container[5] >= candidate[5] - epsilon
        )

    if _contains(left, right) or _contains(right, left):
        return "containment"
    return "overlap"


def _surface_body_patch_families(body: SurfaceBody) -> tuple[str, ...]:
    return tuple(sorted({patch.family for patch in body.iter_patches(world=True)}))


def surface_boolean_family_pair_support(
    operation: SurfaceBooleanOperation,
    left_family: str,
    right_family: str,
) -> SurfaceBooleanFamilyPairSupport:
    """Return the declared CSG support decision for one operation/family pair."""

    return SURFACE_CSG_SOLVER_REGISTRY.support_for(operation, left_family, right_family)


def surface_csg_solver_support_state(
    operation: SurfaceBooleanOperation,
    left_family: str,
    right_family: str,
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
) -> SurfaceBooleanSupportState:
    """Return only the support-state classification from a CSG solver registry."""

    source = registry if registry is not None else SURFACE_CSG_SOLVER_REGISTRY
    return source.support_for(operation, left_family, right_family).support_state


def surface_csg_completion_support_matrix() -> tuple[SurfaceBooleanFamilyPairSupport, ...]:
    """Return the authoritative operation/family CSG support matrix."""

    return SURFACE_CSG_SOLVER_REGISTRY.support_records


def surface_csg_refusal_record(
    operation: SurfaceBooleanOperation,
    left_family: str,
    right_family: str,
) -> SurfaceBooleanUnsupportedFamilyDiagnostic:
    """Build the structured refusal record for an unsupported CSG family pair."""

    support = surface_boolean_family_pair_support(operation, left_family, right_family)
    return build_surface_boolean_unsupported_family_diagnostic(support)


def surface_csg_analytic_primitive_pair_support(
    operation: SurfaceBooleanOperation,
    left_family: str,
    right_family: str,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> SurfaceCSGPrimitiveAnalyticPairRecord:
    """Return analytic primitive-pair support without selecting a mesh substitute."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    support = surface_boolean_family_pair_support(operation, left_family, right_family)
    analytic_pair = left_family in ANALYTIC_SURFACE_CSG_FAMILIES and right_family in ANALYTIC_SURFACE_CSG_FAMILIES
    supported = support.supported and analytic_pair and support.support_state in {"exact", "declared-tolerance"}
    if supported:
        diagnostic = ""
    elif support.supported:
        diagnostic = (
            f"surface boolean {operation} support for {left_family}/{right_family} is outside "
            "the analytic primitive CSG boundary."
        )
    else:
        diagnostic = surface_csg_refusal_record(operation, left_family, right_family).message
    return SurfaceCSGPrimitiveAnalyticPairRecord(
        operation=operation,
        left_family=left_family,
        right_family=right_family,
        supported=supported,
        support_state=support.support_state,
        tolerance_policy=normalized_policy,
        diagnostic=diagnostic,
    )


def _surface_boolean_family_pair_support(
    operation: SurfaceBooleanOperation,
    left_family: str,
    right_family: str,
) -> SurfaceBooleanFamilyPairSupport:
    return surface_boolean_family_pair_support(operation, left_family, right_family)


HIGHER_ORDER_CSG_FAMILIES = HIGHER_ORDER_SURFACE_CSG_FAMILIES


def classify_higher_order_csg_pair(
    operation: SurfaceBooleanOperation,
    left_family: str,
    right_family: str,
) -> SurfaceCSGHigherOrderSupportRecord:
    """Return the exact-solver boundary decision for higher-order CSG pairs."""

    pair_support = surface_boolean_family_pair_support(operation, left_family, right_family)
    higher_order = left_family in HIGHER_ORDER_CSG_FAMILIES or right_family in HIGHER_ORDER_CSG_FAMILIES
    if not higher_order:
        return SurfaceCSGHigherOrderSupportRecord(
            operation=operation,
            left_family=left_family,
            right_family=right_family,
            supported=pair_support.supported,
            solver_boundary="low-order-analytic",
            support_state=pair_support.support_state,
            required_future_capability=pair_support.required_future_capability,
        )
    future = pair_support.required_future_capability or (
        f"exact higher-order surface boolean {operation} solver for {left_family}/{right_family}"
    )
    return SurfaceCSGHigherOrderSupportRecord(
        operation=operation,
        left_family=left_family,
        right_family=right_family,
        supported=False,
        solver_boundary=(
            "sampled-tessellation-boundary"
            if left_family in SAMPLED_SURFACE_CSG_FAMILIES or right_family in SAMPLED_SURFACE_CSG_FAMILIES
            else "higher-order-exact-solver"
        ),
        support_state=pair_support.support_state,
        required_future_capability=future,
    )


def build_higher_order_csg_refusal_diagnostic(
    support: SurfaceCSGHigherOrderSupportRecord,
) -> SurfaceCSGHigherOrderRefusalDiagnostic:
    """Build the explicit refusal diagnostic for a higher-order solver-boundary record."""

    if support.supported:
        raise ValueError("Supported higher-order CSG records do not need refusal diagnostics.")
    if not support.required_future_capability:
        raise ValueError("Unsupported higher-order CSG records require a future capability.")
    return SurfaceCSGHigherOrderRefusalDiagnostic(support=support)


def build_surface_boolean_unsupported_family_diagnostic(
    support: SurfaceBooleanFamilyPairSupport,
) -> SurfaceBooleanUnsupportedFamilyDiagnostic:
    """Build the explicit unsupported-family diagnostic for a family support record."""

    if support.supported:
        raise ValueError("Supported surface boolean family pairs do not need unsupported diagnostics.")
    higher_order = classify_higher_order_csg_pair(support.operation, support.left_family, support.right_family)
    if higher_order.solver_boundary in {"higher-order-exact-solver", "sampled-tessellation-boundary"}:
        higher_order_diagnostic = build_higher_order_csg_refusal_diagnostic(higher_order)
        return SurfaceBooleanUnsupportedFamilyDiagnostic(
            operation=support.operation,
            left_family=support.left_family,
            right_family=support.right_family,
            phase=higher_order.solver_boundary,
            required_future_capability=higher_order_diagnostic.message,
        )
    if support.required_future_capability is None:
        raise ValueError("Unsupported surface boolean family pairs require a future capability.")
    return SurfaceBooleanUnsupportedFamilyDiagnostic(
        operation=support.operation,
        left_family=support.left_family,
        right_family=support.right_family,
        phase=support.phase,
        required_future_capability=support.required_future_capability,
    )


def surface_boolean_family_eligibility(operands: SurfaceBooleanOperands) -> SurfaceBooleanFamilyEligibilityResult:
    """Return structural patch-family support for a surfaced boolean request."""

    family_pairs: list[SurfaceBooleanFamilyPairSupport] = []
    for left_index, left_body in enumerate(operands.bodies):
        for right_body in operands.bodies[left_index + 1 :]:
            for left_family in _surface_body_patch_families(left_body):
                for right_family in _surface_body_patch_families(right_body):
                    family_pairs.append(_surface_boolean_family_pair_support(operands.operation, left_family, right_family))
    diagnostics = tuple(
        build_surface_boolean_unsupported_family_diagnostic(pair)
        for pair in family_pairs
        if not pair.supported
    )
    return SurfaceBooleanFamilyEligibilityResult(
        operation=operands.operation,
        supported=not diagnostics,
        family_pairs=tuple(family_pairs),
        diagnostics=diagnostics,
    )


def plan_prepared_surface_csg_operation(
    operands: SurfaceBooleanOperands,
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
) -> SurfaceCSGOperationPlan:
    """Plan an already prepared surface CSG operation without executing it."""

    source = registry if registry is not None else SURFACE_CSG_SOLVER_REGISTRY
    pair_dispatch: list[SurfaceCSGPairDispatchRecord] = []
    diagnostics: list[SurfaceCSGPlanDiagnostic] = []
    for left_index, left_body in enumerate(operands.bodies):
        for right_index, right_body in enumerate(operands.bodies[left_index + 1 :], start=left_index + 1):
            for left_family in _surface_body_patch_families(left_body):
                for right_family in _surface_body_patch_families(right_body):
                    support = source.support_for(operands.operation, left_family, right_family)
                    dispatch = SurfaceCSGPairDispatchRecord.from_support(
                        left_operand_index=left_index,
                        right_operand_index=right_index,
                        support=support,
                    )
                    pair_dispatch.append(dispatch)
                    if not support.supported:
                        unsupported = build_surface_boolean_unsupported_family_diagnostic(support)
                        diagnostics.append(
                            SurfaceCSGPlanDiagnostic(
                                code="unsupported-family-pair",
                                operation=operands.operation,
                                left_family=left_family,
                                right_family=right_family,
                                phase=unsupported.phase,
                                message=unsupported.message,
                            )
                        )
    return SurfaceCSGOperationPlan(
        operation=operands.operation,
        operands=operands,
        pair_dispatch=tuple(pair_dispatch),
        diagnostics=tuple(diagnostics),
    )


def plan_surface_csg_operation(
    operation: SurfaceBooleanOperation,
    bodies: Iterable[object],
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
) -> SurfaceCSGOperationPlan:
    """Prepare and plan a surface CSG operation, accumulating all pre-execution diagnostics."""

    body_tuple = tuple(bodies)
    diagnostics: list[SurfaceCSGPlanDiagnostic] = []
    if operation not in SURFACE_BOOLEAN_OPERATIONS:
        raise ValueError(f"Unsupported surface CSG operation {operation!r}.")
    if operation in {"union", "intersection"} and len(body_tuple) < 2:
        diagnostics.append(
            SurfaceCSGPlanDiagnostic(
                code="invalid-operand-count",
                operation=operation,
                message=f"Surface boolean {operation} requires at least two SurfaceBody operands.",
            )
        )
    if operation == "difference" and len(body_tuple) < 2:
        diagnostics.append(
            SurfaceCSGPlanDiagnostic(
                code="invalid-operand-count",
                operation=operation,
                message="Surface boolean difference requires a base and at least one cutter SurfaceBody.",
            )
        )

    canonical: list[SurfaceBody] = []
    for index, body in enumerate(body_tuple):
        if not isinstance(body, SurfaceBody):
            diagnostics.append(
                SurfaceCSGPlanDiagnostic(
                    code="invalid-operand",
                    operation=operation,
                    operand_index=index,
                    message=f"Surface CSG operand {index} must be a SurfaceBody; no mesh fallback was attempted.",
                )
            )
            continue
        role = "difference base" if operation == "difference" and index == 0 else f"{operation} operand {index}"
        if operation == "difference" and index > 0:
            role = f"difference cutter {index - 1}"
        try:
            canonical.append(_canonicalize_surface_boolean_body(body, role=role))
        except (TypeError, ValueError, SurfaceBooleanEligibilityError) as exc:
            diagnostics.append(
                SurfaceCSGPlanDiagnostic(
                    code="invalid-operand",
                    operation=operation,
                    operand_index=index,
                    message=str(exc),
                )
            )

    if diagnostics:
        return SurfaceCSGOperationPlan(operation=operation, operands=None, diagnostics=tuple(diagnostics))

    operands = SurfaceBooleanOperands(operation=operation, bodies=tuple(canonical))
    return plan_prepared_surface_csg_operation(operands, registry=registry)


SURFACE_CSG_CALLER_INVENTORY: tuple[SurfaceCSGCallerInventoryRecord, ...] = (
    SurfaceCSGCallerInventoryRecord(
        caller_id="csg.boolean_union",
        module="impression.modeling.csg",
        category="public-api",
        operation="union",
        surface_route="surface_boolean_result",
        mesh_route="_apply_boolean",
        explicit_mesh_route=True,
    ),
    SurfaceCSGCallerInventoryRecord(
        caller_id="csg.boolean_difference",
        module="impression.modeling.csg",
        category="public-api",
        operation="difference",
        surface_route="surface_boolean_result",
        mesh_route="_apply_boolean",
        explicit_mesh_route=True,
    ),
    SurfaceCSGCallerInventoryRecord(
        caller_id="csg.boolean_intersection",
        module="impression.modeling.csg",
        category="public-api",
        operation="intersection",
        surface_route="surface_boolean_result",
        mesh_route="_apply_boolean",
        explicit_mesh_route=True,
    ),
    SurfaceCSGCallerInventoryRecord(
        caller_id="hinges.make_traditional_hinge_pair",
        module="impression.modeling.hinges",
        category="feature",
        operation="union",
        surface_route="make_traditional_hinge_pair",
        mesh_route="_call_with_legacy_mesh_primitives",
        explicit_mesh_route=True,
    ),
    SurfaceCSGCallerInventoryRecord(
        caller_id="primitive.boolean_dependent_surface_builders",
        module="impression.modeling.primitives",
        category="primitive",
        operation=None,
        surface_route="surface primitive constructors",
        mesh_route="explicit backend='mesh' primitive constructors",
        explicit_mesh_route=True,
    ),
)


def surface_csg_caller_inventory() -> tuple[SurfaceCSGCallerInventoryRecord, ...]:
    """Return the durable inventory of authored routes that depend on CSG readiness."""

    return SURFACE_CSG_CALLER_INVENTORY


def surface_csg_feature_gate(
    caller_id: str,
    operation: SurfaceBooleanOperation,
    bodies: Iterable[object],
) -> SurfaceCSGFeatureGateDiagnostic:
    """Return the shared CSG readiness diagnostic for a primitive or feature caller."""

    body_tuple = tuple(bodies)
    if any(not isinstance(body, SurfaceBody) for body in body_tuple):
        return SurfaceCSGFeatureGateDiagnostic(
            caller_id=caller_id,
            operation=operation,
            supported=False,
            reason="Surface CSG requires SurfaceBody operands; no mesh fallback was attempted.",
        )
    if operation in {"union", "intersection"} and len(body_tuple) < 2:
        return SurfaceCSGFeatureGateDiagnostic(
            caller_id=caller_id,
            operation=operation,
            supported=False,
            operand_ids=tuple(body.stable_identity for body in body_tuple),
            reason=f"Surface boolean {operation} requires at least two SurfaceBody operands.",
        )
    if operation == "difference" and len(body_tuple) < 2:
        return SurfaceCSGFeatureGateDiagnostic(
            caller_id=caller_id,
            operation=operation,
            supported=False,
            operand_ids=tuple(body.stable_identity for body in body_tuple),
            reason="Surface boolean difference requires a base and at least one cutter SurfaceBody.",
        )

    plan = plan_surface_csg_operation(operation, body_tuple)
    if not plan.executable:
        return SurfaceCSGFeatureGateDiagnostic(
            caller_id=caller_id,
            operation=operation,
            supported=False,
            operand_ids=tuple(body.stable_identity for body in body_tuple),
            reason="; ".join(diagnostic.message for diagnostic in plan.diagnostics)
            or "Surface boolean planner rejected the request.",
        )
    return SurfaceCSGFeatureGateDiagnostic(
        caller_id=caller_id,
        operation=operation,
        supported=True,
        operand_ids=plan.body_ids,
        reason="Surface boolean request passed the shared CSG readiness gate.",
    )


def assert_no_hidden_surface_csg_mesh_fallback(caller_id: str, result: object) -> object:
    """Reject hidden mesh results from authored surface CSG routes."""

    if isinstance(result, (Mesh, MeshGroup)):
        raise BooleanOperationError(
            f"{caller_id} produced a mesh result across the surface CSG route; explicit mesh compatibility is required."
        )
    return result


def _contains_bounds(
    container: tuple[float, float, float, float, float, float],
    candidate: tuple[float, float, float, float, float, float],
    *,
    epsilon: float = 1e-9,
) -> bool:
    return (
        container[0] <= candidate[0] + epsilon
        and container[1] >= candidate[1] - epsilon
        and container[2] <= candidate[2] + epsilon
        and container[3] >= candidate[3] - epsilon
        and container[4] <= candidate[4] + epsilon
        and container[5] >= candidate[5] - epsilon
    )


def _bounds_size(bounds: tuple[float, float, float, float, float, float]) -> tuple[float, float, float]:
    return (
        float(bounds[1] - bounds[0]),
        float(bounds[3] - bounds[2]),
        float(bounds[5] - bounds[4]),
    )


def _bounds_center(bounds: tuple[float, float, float, float, float, float]) -> tuple[float, float, float]:
    return (
        float((bounds[0] + bounds[1]) * 0.5),
        float((bounds[2] + bounds[3]) * 0.5),
        float((bounds[4] + bounds[5]) * 0.5),
    )


def _surface_boolean_provenance_payload(operands: SurfaceBooleanOperands) -> dict[str, object]:
    return SurfaceCSGProvenanceMetadataRecord(
        operation=operands.operation,
        operand_ids=operands.body_ids,
    ).canonical_payload()


def _surface_boolean_result_metadata(operands: SurfaceBooleanOperands) -> dict[str, object]:
    inherited_kernel = dict(operands.bodies[0].kernel_metadata())
    inherited_consumer = dict(operands.bodies[0].consumer_metadata())
    for body in operands.bodies[1:]:
        inherited_kernel.update(body.kernel_metadata())
        inherited_consumer.update(body.consumer_metadata())
    provenance = _surface_boolean_provenance_payload(operands)
    inherited_kernel["boolean_backend"] = "surface"
    inherited_kernel["boolean_operation"] = operands.operation
    inherited_kernel["boolean_operand_ids"] = operands.body_ids
    inherited_kernel["boolean_provenance"] = provenance
    inherited_consumer["boolean_backend"] = "surface"
    inherited_consumer["boolean_operation"] = operands.operation
    inherited_consumer["boolean_operand_ids"] = operands.body_ids
    inherited_consumer["boolean_provenance"] = provenance
    return {
        "kernel": inherited_kernel,
        "consumer": inherited_consumer,
    }


def _surface_boolean_boundary_key(boundary) -> tuple[int, str]:
    return (boundary.patch_index, boundary.boundary_id)


def _surface_boolean_trim_key(trim_loop: TrimLoop) -> tuple[object, ...]:
    normalized = trim_loop.normalized()
    return (
        normalized.category,
        *(tuple(float(value) for value in point) for point in np.round(normalized.points_uv, decimals=12)),
    )


def _surface_boolean_cleanup_patch(
    patch,
    *,
    epsilon: float = 1e-12,
):
    if not patch.trim_loops:
        return patch

    cleaned_trim_loops: list[TrimLoop] = []
    seen: set[tuple[object, ...]] = set()
    for trim_loop in patch.trim_loops:
        normalized = trim_loop.normalized()
        if abs(normalized.area) <= epsilon:
            continue
        trim_key = _surface_boolean_trim_key(normalized)
        if trim_key in seen:
            continue
        seen.add(trim_key)
        cleaned_trim_loops.append(normalized)
    ordered_trim_loops = tuple(
        sorted(
            cleaned_trim_loops,
            key=lambda loop: (0 if loop.category == "outer" else 1, _surface_boolean_trim_key(loop)),
        )
    )
    return replace(patch, trim_loops=ordered_trim_loops)


def _surface_boolean_cleanup_shell(shell) -> object:
    cleaned_patches = tuple(
        _surface_boolean_cleanup_patch(patch)
        for patch in shell.iter_patches(world=True)
    )
    cleaned_seams = []
    seen_seam_keys: set[tuple[object, ...]] = set()
    for seam in sorted(
        shell.seams,
        key=lambda item: (
            tuple(sorted(_surface_boolean_boundary_key(boundary) for boundary in item.boundaries)),
            item.continuity,
            item.seam_id,
        ),
    ):
        canonical_boundaries = tuple(sorted(seam.boundaries, key=_surface_boolean_boundary_key))
        seam_key = (
            tuple(_surface_boolean_boundary_key(boundary) for boundary in canonical_boundaries),
            seam.continuity,
        )
        if seam_key in seen_seam_keys:
            continue
        seen_seam_keys.add(seam_key)
        cleaned_seams.append(replace(seam, boundaries=canonical_boundaries))

    cleaned_adjacency = tuple(
        sorted(
            shell.adjacency,
            key=lambda record: (
                record.source.patch_index,
                record.source.boundary_id,
                -1 if record.target is None else record.target.patch_index,
                "" if record.target is None else record.target.boundary_id,
                "" if record.seam_id is None else record.seam_id,
                record.continuity,
            ),
        )
    )
    return make_surface_shell(
        cleaned_patches,
        connected=shell.connected,
        seams=tuple(cleaned_seams),
        adjacency=cleaned_adjacency,
        metadata=shell.metadata,
    )


def rebuild_surface_csg_shell_seams(shell: SurfaceShell) -> SurfaceCSGSeamRebuildRecord:
    """Canonicalize seam, boundary-use, and adjacency truth for a provisional CSG shell."""

    rebuilt_shell = _surface_boolean_cleanup_shell(shell)
    boundary_counts: dict[tuple[int, str], list[str]] = {}
    from .tessellation import _patch_boundary_ids

    for patch_index, patch in enumerate(rebuilt_shell.patches):
        for boundary_id in _patch_boundary_ids(patch):
            boundary_counts[(patch_index, boundary_id)] = []
    for seam in rebuilt_shell.seams:
        for boundary in seam.boundaries:
            boundary_counts.setdefault(_surface_boolean_boundary_key(boundary), []).append(seam.seam_id)
    boundary_uses = tuple(
        SurfaceCSGBoundaryUseProvenanceRecord(
            boundary=SurfaceBoundaryRef(patch_index, boundary_id),
            use_count=len(seam_ids),
            seam_ids=tuple(sorted(seam_ids)),
        )
        for (patch_index, boundary_id), seam_ids in sorted(boundary_counts.items())
    )
    invalid_reason = _surface_boolean_shell_invalid_reason(rebuilt_shell)
    diagnostics = () if invalid_reason is None else (invalid_reason,)
    return SurfaceCSGSeamRebuildRecord(
        shell=rebuilt_shell,
        boundary_uses=boundary_uses,
        diagnostics=diagnostics,
    )


def _surface_boolean_cleanup_body(body: SurfaceBody) -> SurfaceBody:
    cleaned_shells = tuple(rebuild_surface_csg_shell_seams(shell).shell for shell in body.iter_shells(world=True))
    return make_surface_body(cleaned_shells, metadata=body.metadata)


def finalize_surface_csg_validity_gate(
    operation: SurfaceBooleanOperation,
    operands: SurfaceBooleanOperands,
    body: SurfaceBody,
) -> SurfaceCSGValidityGateRecord:
    """Apply bounded cleanup, validate, and finalize provenance for a CSG body."""

    provenance = SurfaceCSGProvenanceMetadataRecord(
        operation=operation,
        operand_ids=operands.body_ids,
    )
    cleaned_body = _surface_boolean_cleanup_body(body)
    diagnostics: list[SurfaceCSGValidityDiagnostic] = []

    for shell in cleaned_body.iter_shells(world=True):
        invalid_reason = _surface_boolean_shell_invalid_reason(shell)
        if invalid_reason is not None:
            diagnostics.append(
                SurfaceCSGValidityDiagnostic(
                    code="invalid-shell",
                    message=invalid_reason,
                )
            )

    if not diagnostics:
        classification = _classify_surface_body(cleaned_body)
        if classification != "closed":
            diagnostics.append(
                SurfaceCSGValidityDiagnostic(
                    code="non-closed-result",
                    message="Surface boolean validity gate rejected a non-closed reconstructed result.",
                )
            )

    if diagnostics:
        return SurfaceCSGValidityGateRecord(
            status="invalid",
            diagnostics=tuple(diagnostics),
            provenance=provenance,
        )

    return SurfaceCSGValidityGateRecord(
        status="succeeded",
        body=cleaned_body,
        provenance=provenance,
    )


def validate_surface_csg_result_handoff(
    assembly: SurfaceCSGShellAssemblyRecord,
    operands: SurfaceBooleanOperands,
    *,
    metadata: dict[str, object] | None = None,
) -> SurfaceCSGValidityHandoffRecord:
    """Validate a reconstructed CSG body candidate through seam rebuild and validity gates."""

    diagnostics: list[SurfaceCSGPostReconstructionValidityDiagnostic] = []
    if assembly.operation != operands.operation:
        diagnostics.append(
            SurfaceCSGPostReconstructionValidityDiagnostic(
                code="invalid-assembly",
                message=(
                    f"Surface CSG validity handoff received {assembly.operation!r} assembly "
                    f"for {operands.operation!r} operands."
                ),
            )
        )
    if not assembly.supported:
        diagnostics.extend(
            SurfaceCSGPostReconstructionValidityDiagnostic(
                code="invalid-assembly",
                message=diagnostic.message,
                underlying_code=diagnostic.code,
            )
            for diagnostic in assembly.diagnostics
        )
    if diagnostics:
        return SurfaceCSGValidityHandoffRecord(
            operation=operands.operation,
            classification=assembly.classification,
            status="invalid",
            assembly=assembly,
            diagnostics=tuple(diagnostics),
        )
    if assembly.classification == "empty":
        return SurfaceCSGValidityHandoffRecord(
            operation=operands.operation,
            classification="empty",
            status="succeeded",
            assembly=assembly,
        )

    body_metadata = metadata if metadata is not None else _surface_boolean_result_metadata(operands)
    try:
        candidate = assembly.to_body(metadata=body_metadata)
    except SurfaceBooleanEligibilityError as exc:
        diagnostics.append(
            SurfaceCSGPostReconstructionValidityDiagnostic(
                code="invalid-assembly",
                message=str(exc),
            )
        )
        return SurfaceCSGValidityHandoffRecord(
            operation=operands.operation,
            classification=assembly.classification,
            status="invalid",
            assembly=assembly,
            diagnostics=tuple(diagnostics),
        )
    if candidate is None:
        return SurfaceCSGValidityHandoffRecord(
            operation=operands.operation,
            classification="empty",
            status="succeeded",
            assembly=assembly,
        )

    seam_rebuilds = tuple(rebuild_surface_csg_shell_seams(shell) for shell in candidate.iter_shells(world=True))
    for shell_index, rebuild in enumerate(seam_rebuilds):
        diagnostics.extend(
            SurfaceCSGPostReconstructionValidityDiagnostic(
                code="seam-rebuild-failed",
                message=message,
                result_shell_index=shell_index,
            )
            for message in rebuild.diagnostics
        )

    gate = finalize_surface_csg_validity_gate(operands.operation, operands, candidate)
    if not gate.accepted:
        diagnostics.extend(
            SurfaceCSGPostReconstructionValidityDiagnostic(
                code="validity-gate-rejected",
                message=diagnostic.message,
                underlying_code=diagnostic.code,
            )
            for diagnostic in gate.diagnostics
        )

    if diagnostics:
        return SurfaceCSGValidityHandoffRecord(
            operation=operands.operation,
            classification=assembly.classification,
            status="invalid",
            assembly=assembly,
            seam_rebuilds=seam_rebuilds,
            validity_gate=gate,
            diagnostics=tuple(diagnostics),
        )

    return SurfaceCSGValidityHandoffRecord(
        operation=operands.operation,
        classification=assembly.classification,
        status="succeeded",
        assembly=assembly,
        body=gate.body,
        seam_rebuilds=seam_rebuilds,
        validity_gate=gate,
    )


def normalize_surface_csg_operand_ordering(
    operation: SurfaceBooleanOperation,
    operands: SurfaceBooleanOperands,
) -> SurfaceCSGOperandOrderingNormalizationRecord:
    """Return deterministic operand ordering for provenance comparison."""

    operand_ids = operands.body_ids
    if operation in {"union", "intersection"}:
        normalized_pairs = tuple(sorted(enumerate(operand_ids), key=lambda item: (item[1], item[0])))
    else:
        normalized_pairs = tuple(enumerate(operand_ids))
    return SurfaceCSGOperandOrderingNormalizationRecord(
        operation=operation,
        operand_ids=operand_ids,
        normalized_operand_ids=tuple(body_id for _index, body_id in normalized_pairs),
        normalized_to_original_indices=tuple(index for index, _body_id in normalized_pairs),
    )


def _surface_csg_cap_payload_lookup(
    cap_construction: SurfaceCSGCapConstructionRecord | None,
) -> dict[tuple[SurfaceBooleanPatchRef, tuple[str, ...]], int]:
    if cap_construction is None:
        return {}
    return {
        (payload.source_patch, tuple(sorted(payload.cut_curve_ids))): index
        for index, payload in enumerate(cap_construction.cap_payloads)
    }


def _surface_csg_boundary_attachment_lookup(
    cut_boundary: SurfaceCSGCutBoundaryRecord | None,
) -> dict[int, int]:
    if cut_boundary is None:
        return {}
    return {
        attachment.cap_payload_index: index
        for index, attachment in enumerate(cut_boundary.trim_attachments)
    }


def _surface_csg_result_patch_role(
    assembly: SurfaceCSGShellAssemblyRecord,
    provenance: SurfaceCSGFragmentProvenanceRecord,
    cap_lookup: Mapping[tuple[SurfaceBooleanPatchRef, tuple[str, ...]], int],
) -> Literal["surviving-fragment", "generated-cap"]:
    cap_key = (provenance.source_patch, tuple(sorted(provenance.cut_curve_ids)))
    if cap_key in cap_lookup:
        try:
            patch = assembly.shells[provenance.result_shell_index].patches[provenance.result_patch_index]
            if patch.kernel_metadata().get("generated_role") == "csg_generated_cap":
                return "generated-cap"
        except (AttributeError, IndexError):
            return "generated-cap"
    return "surviving-fragment"


def build_surface_csg_result_provenance_map(
    assembly: SurfaceCSGShellAssemblyRecord,
    operands: SurfaceBooleanOperands,
    *,
    graph: SurfaceCSGFragmentGraphRecord | None = None,
    cap_construction: SurfaceCSGCapConstructionRecord | None = None,
    cut_boundary: SurfaceCSGCutBoundaryRecord | None = None,
) -> SurfaceCSGResultProvenanceMap:
    """Build stable operand, fragment, cap, and boundary provenance for a CSG result."""

    ordering = normalize_surface_csg_operand_ordering(operands.operation, operands)
    diagnostics: list[SurfaceCSGProvenanceDiagnostic] = []
    if assembly.operation != operands.operation:
        diagnostics.append(
            SurfaceCSGProvenanceDiagnostic(
                code="invalid-assembly",
                message=(
                    f"Surface CSG provenance map received {assembly.operation!r} assembly "
                    f"for {operands.operation!r} operands."
                ),
            )
        )
    if not assembly.supported:
        diagnostics.extend(
            SurfaceCSGProvenanceDiagnostic(
                code="invalid-assembly",
                message=diagnostic.message,
                source_patch=diagnostic.source_patch,
            )
            for diagnostic in assembly.diagnostics
        )
    if graph is not None and graph.plan.operands is not None and graph.plan.operands.body_ids != operands.body_ids:
        diagnostics.append(
            SurfaceCSGProvenanceDiagnostic(
                code="invalid-assembly",
                message="Surface CSG provenance map received fragment graph for different operands.",
            )
        )

    cap_lookup = _surface_csg_cap_payload_lookup(cap_construction)
    boundary_lookup = _surface_csg_boundary_attachment_lookup(cut_boundary)
    records: list[SurfaceCSGResultPatchProvenanceRecord] = []
    for provenance in sorted(
        assembly.provenance,
        key=lambda record: (
            record.result_shell_index,
            record.result_patch_index,
            _surface_csg_patch_ref_sort_key(record.source_patch),
            record.cut_curve_ids,
        ),
    ):
        try:
            assembly.shells[provenance.result_shell_index].patches[provenance.result_patch_index]
        except IndexError:
            diagnostics.append(
                SurfaceCSGProvenanceDiagnostic(
                    code="missing-result-patch",
                    message=(
                        "Surface CSG provenance map references a result patch outside "
                        f"shell {provenance.result_shell_index}."
                    ),
                    result_shell_index=provenance.result_shell_index,
                    result_patch_index=provenance.result_patch_index,
                    source_patch=provenance.source_patch,
                )
            )
            continue
        role = _surface_csg_result_patch_role(assembly, provenance, cap_lookup)
        cap_payload_index = None
        boundary_attachment_index = None
        if role == "generated-cap":
            cap_payload_index = cap_lookup.get((provenance.source_patch, tuple(sorted(provenance.cut_curve_ids))))
            if cap_payload_index is not None:
                boundary_attachment_index = boundary_lookup.get(cap_payload_index)
            if boundary_attachment_index is None:
                diagnostics.append(
                    SurfaceCSGProvenanceDiagnostic(
                        code="missing-boundary-attachment",
                        message="Generated CSG cap provenance is missing a cut-boundary attachment.",
                        result_shell_index=provenance.result_shell_index,
                        result_patch_index=provenance.result_patch_index,
                        source_patch=provenance.source_patch,
                    )
                )
        records.append(
            SurfaceCSGResultPatchProvenanceRecord(
                result_shell_index=provenance.result_shell_index,
                result_patch_index=provenance.result_patch_index,
                source_patch=provenance.source_patch,
                source_role=role,
                cut_curve_ids=provenance.cut_curve_ids,
                cap_payload_index=cap_payload_index,
                boundary_attachment_index=boundary_attachment_index,
            )
        )

    return SurfaceCSGResultProvenanceMap(
        operation=operands.operation,
        operand_ordering=ordering,
        result_patches=tuple(records),
        diagnostics=tuple(diagnostics),
    )


def _surface_boolean_shell_invalid_reason(
    shell,
    *,
    epsilon: float = 1e-9,
) -> str | None:
    from .tessellation import _boundary_is_collapsed, _patch_boundary_ids

    if not shell.connected:
        return "Surface boolean validity gate rejected a disconnected shell."

    boundary_use_counts: dict[tuple[int, str], int] = {}
    for patch_index, patch in enumerate(shell.patches):
        for boundary_id in _patch_boundary_ids(patch):
            boundary_use_counts[(patch_index, boundary_id)] = 0

    for seam in shell.seams:
        if seam.is_open:
            return f"Surface boolean validity gate rejected open seam {seam.seam_id!r}."
        for boundary in seam.boundaries:
            boundary_key = _surface_boolean_boundary_key(boundary)
            if boundary_key not in boundary_use_counts:
                return (
                    "Surface boolean validity gate found a seam boundary outside the reconstructed "
                    "patch boundary set."
                )
            boundary_use_counts[boundary_key] += 1

    for (patch_index, boundary_id), use_count in boundary_use_counts.items():
        patch = shell.patches[patch_index]
        if use_count == 0 and _boundary_is_collapsed(patch, boundary_id, tolerance=epsilon):
            continue
        if use_count == 0:
            return (
                "Surface boolean validity gate rejected a shell with missing seam coverage on "
                f"patch {patch_index} boundary {boundary_id!r}."
            )
        if use_count > 1:
            return (
                "Surface boolean validity gate rejected duplicate seam use on "
                f"patch {patch_index} boundary {boundary_id!r}."
            )
    return None


def _surface_boolean_finalize_body_result(
    operation: SurfaceBooleanOperation,
    operands: SurfaceBooleanOperands,
    body: SurfaceBody,
) -> SurfaceBooleanResult:
    gate = finalize_surface_csg_validity_gate(operation, operands, body)
    if not gate.accepted:
        return SurfaceBooleanResult(
            operation=operation,
            operands=operands,
            status="invalid",
            failure_reason="; ".join(diagnostic.message for diagnostic in gate.diagnostics),
        )
    return SurfaceBooleanResult(
        operation=operation,
        operands=operands,
        status="succeeded",
        body=gate.body,
        classification="closed",
    )


def _clone_surface_shell(shell) -> object:
    return make_surface_shell(
        shell.iter_patches(world=True),
        connected=shell.connected,
        seams=shell.seams,
        adjacency=shell.adjacency,
        metadata=shell.metadata,
    )


def _clone_surface_body_with_metadata(
    body: SurfaceBody,
    *,
    metadata: dict[str, object],
) -> SurfaceBody:
    shells = tuple(_clone_surface_shell(shell) for shell in body.iter_shells(world=True))
    return make_surface_body(shells, metadata=metadata)


def _combine_surface_bodies_with_metadata(
    bodies: Sequence[SurfaceBody],
    *,
    metadata: dict[str, object],
) -> SurfaceBody:
    shells = tuple(_clone_surface_shell(shell) for body in bodies for shell in body.iter_shells(world=True))
    return make_surface_body(shells, metadata=metadata)


def _surface_body_primitive_family(body: SurfaceBody) -> str | None:
    if body.shell_count != 1:
        return None
    shell = body.iter_shells(world=True)[0]
    kernel = shell.metadata.get("kernel", {})
    family = kernel.get("primitive_family")
    if isinstance(family, str) and family:
        return family
    if shell.patch_count == 1:
        patch_family = shell.iter_patches(world=True)[0].kernel_metadata().get("primitive_family")
        if isinstance(patch_family, str) and patch_family:
            return patch_family
    return None


def _bounds_corners(bounds: tuple[float, float, float, float, float, float]) -> np.ndarray:
    return np.asarray(
        [
            (bounds[0], bounds[2], bounds[4]),
            (bounds[0], bounds[2], bounds[5]),
            (bounds[0], bounds[3], bounds[4]),
            (bounds[0], bounds[3], bounds[5]),
            (bounds[1], bounds[2], bounds[4]),
            (bounds[1], bounds[2], bounds[5]),
            (bounds[1], bounds[3], bounds[4]),
            (bounds[1], bounds[3], bounds[5]),
        ],
        dtype=float,
    )


def _surface_body_sphere_parameters(
    body: SurfaceBody,
    *,
    epsilon: float = 1e-9,
) -> tuple[np.ndarray, float] | None:
    if _surface_body_primitive_family(body) != "sphere" or body.shell_count != 1:
        return None
    shell = body.iter_shells(world=True)[0]
    if shell.patch_count != 1:
        return None
    patch = shell.iter_patches(world=True)[0]
    if not isinstance(patch, RevolutionSurfacePatch):
        return None
    bounds = body.bounds_estimate()
    spans = _bounds_size(bounds)
    if not (abs(spans[0] - spans[1]) <= epsilon and abs(spans[1] - spans[2]) <= epsilon):
        return None
    radius = spans[0] * 0.5
    if radius <= epsilon:
        return None
    return np.asarray(_bounds_center(bounds), dtype=float), float(radius)


def _surface_body_contains_exact(
    container: SurfaceBody,
    candidate: SurfaceBody,
    *,
    epsilon: float = 1e-9,
) -> bool:
    container_family = _surface_body_primitive_family(container)
    candidate_family = _surface_body_primitive_family(candidate)
    container_bounds = container.bounds_estimate()
    candidate_bounds = candidate.bounds_estimate()

    if container_family == "box" and candidate_family == "box":
        return _contains_bounds(container_bounds, candidate_bounds, epsilon=epsilon)
    if container_family == "box" and candidate_family == "sphere":
        return _contains_bounds(container_bounds, candidate_bounds, epsilon=epsilon)
    if container_family == "sphere" and candidate_family == "sphere":
        container_sphere = _surface_body_sphere_parameters(container, epsilon=epsilon)
        candidate_sphere = _surface_body_sphere_parameters(candidate, epsilon=epsilon)
        if container_sphere is None or candidate_sphere is None:
            return False
        container_center, container_radius = container_sphere
        candidate_center, candidate_radius = candidate_sphere
        return (
            float(np.linalg.norm(candidate_center - container_center)) + candidate_radius
            <= container_radius + epsilon
        )
    if container_family == "sphere" and candidate_family == "box":
        container_sphere = _surface_body_sphere_parameters(container, epsilon=epsilon)
        if container_sphere is None:
            return False
        container_center, container_radius = container_sphere
        corners = _bounds_corners(candidate_bounds)
        distances = np.linalg.norm(corners - container_center, axis=1)
        return bool(np.all(distances <= container_radius + epsilon))
    return False


def _surface_boolean_trivial_result(operands: SurfaceBooleanOperands) -> SurfaceBooleanResult | None:
    metadata = _surface_boolean_result_metadata(operands)

    if operands.operation == "difference":
        base = operands.bodies[0]
        cutters = operands.bodies[1:]
        if all(
            _surface_boolean_body_relation(base.bounds_estimate(), cutter.bounds_estimate()) in {"disjoint", "touching"}
            for cutter in cutters
        ):
            body = _clone_surface_body_with_metadata(base, metadata=metadata)
            return _surface_boolean_finalize_body_result("difference", operands, body)
        if any(
            _surface_boolean_body_relation(base.bounds_estimate(), cutter.bounds_estimate()) == "equal"
            or _surface_body_contains_exact(cutter, base)
            for cutter in cutters
        ):
            return SurfaceBooleanResult(
                operation="difference",
                operands=operands,
                status="succeeded",
                classification="empty",
            )
        return None

    if operands.operand_count != 2:
        return None

    left, right = operands.bodies
    relation = _surface_boolean_body_relation(left.bounds_estimate(), right.bounds_estimate())
    left_contains_right = relation in {"containment", "equal"} and _surface_body_contains_exact(left, right)
    right_contains_left = relation in {"containment", "equal"} and _surface_body_contains_exact(right, left)

    if operands.operation == "union":
        if relation == "disjoint":
            body = _combine_surface_bodies_with_metadata((left, right), metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        if relation == "equal" or (left_contains_right and right_contains_left):
            body = _clone_surface_body_with_metadata(left, metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        if left_contains_right:
            body = _clone_surface_body_with_metadata(left, metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        if right_contains_left:
            body = _clone_surface_body_with_metadata(right, metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        return None

    if relation in {"disjoint", "touching"}:
        return SurfaceBooleanResult(
            operation="intersection",
            operands=operands,
            status="succeeded",
            classification="empty",
        )
    if relation == "equal" or (left_contains_right and right_contains_left):
        body = _clone_surface_body_with_metadata(left, metadata=metadata)
        return _surface_boolean_finalize_body_result("intersection", operands, body)
    if left_contains_right:
        body = _clone_surface_body_with_metadata(right, metadata=metadata)
        return _surface_boolean_finalize_body_result("intersection", operands, body)
    if right_contains_left:
        body = _clone_surface_body_with_metadata(left, metadata=metadata)
        return _surface_boolean_finalize_body_result("intersection", operands, body)
    return None


def _surface_box_body_from_bounds(
    bounds: tuple[float, float, float, float, float, float],
    *,
    metadata: dict[str, object],
) -> SurfaceBody:
    from ._surface_primitives import make_surface_box

    return make_surface_box(
        size=_bounds_size(bounds),
        center=_bounds_center(bounds),
        metadata=metadata,
    )


def _sorted_unique_axis_values(
    values: Iterable[float],
    *,
    epsilon: float = 1e-9,
) -> tuple[float, ...]:
    unique: list[float] = []
    for value in sorted(float(item) for item in values):
        if not unique or abs(value - unique[-1]) > epsilon:
            unique.append(value)
    return tuple(unique)


def _point_inside_bounds(
    point: Sequence[float],
    bounds: tuple[float, float, float, float, float, float],
    *,
    epsilon: float = 1e-9,
) -> bool:
    x, y, z = (float(coord) for coord in point)
    return (
        (bounds[0] - epsilon) <= x <= (bounds[1] + epsilon)
        and (bounds[2] - epsilon) <= y <= (bounds[3] + epsilon)
        and (bounds[4] - epsilon) <= z <= (bounds[5] + epsilon)
    )


@dataclass(frozen=True)
class _RectilinearBoundaryFace:
    axis: int
    side: Literal["min", "max"]
    cell_index: tuple[int, int, int]
    mins: tuple[float, float, float]
    maxs: tuple[float, float, float]


def _rectilinear_boundary_sort_key(face: _RectilinearBoundaryFace) -> tuple[object, ...]:
    return (
        face.axis,
        0 if face.side == "min" else 1,
        *(round(value, 12) for value in face.mins),
        *(round(value, 12) for value in face.maxs),
        *face.cell_index,
    )


def _surface_rectangular_patch_from_face(
    face: _RectilinearBoundaryFace,
    *,
    operation: SurfaceBooleanOperation,
) -> PlanarSurfacePatch:
    x0, y0, z0 = face.mins
    x1, y1, z1 = face.maxs
    axis_name = _axis_name(face.axis)
    metadata = {
        "kernel": {
            "primitive_family": "orthogonal_boolean",
            "boolean_operation": operation,
            "axis": axis_name,
            "side": face.side,
        }
    }
    if face.axis == 0 and face.side == "max":
        return PlanarSurfacePatch(
            family="planar",
            origin=(x1, y0, z0),
            u_axis=(0.0, 0.0, z1 - z0),
            v_axis=(0.0, y1 - y0, 0.0),
            metadata=metadata,
        )
    if face.axis == 0 and face.side == "min":
        return PlanarSurfacePatch(
            family="planar",
            origin=(x0, y0, z1),
            u_axis=(0.0, 0.0, z0 - z1),
            v_axis=(0.0, y1 - y0, 0.0),
            metadata=metadata,
        )
    if face.axis == 1 and face.side == "max":
        return PlanarSurfacePatch(
            family="planar",
            origin=(x0, y1, z0),
            u_axis=(x1 - x0, 0.0, 0.0),
            v_axis=(0.0, 0.0, z1 - z0),
            metadata=metadata,
        )
    if face.axis == 1 and face.side == "min":
        return PlanarSurfacePatch(
            family="planar",
            origin=(x0, y0, z1),
            u_axis=(x1 - x0, 0.0, 0.0),
            v_axis=(0.0, 0.0, z0 - z1),
            metadata=metadata,
        )
    if face.axis == 2 and face.side == "max":
        return PlanarSurfacePatch(
            family="planar",
            origin=(x1, y0, z1),
            u_axis=(x0 - x1, 0.0, 0.0),
            v_axis=(0.0, y1 - y0, 0.0),
            metadata=metadata,
        )
    return PlanarSurfacePatch(
        family="planar",
        origin=(x0, y0, z0),
        u_axis=(x1 - x0, 0.0, 0.0),
        v_axis=(0.0, y1 - y0, 0.0),
        metadata=metadata,
    )


def _patch_boundary_endpoints(
    patch: PlanarSurfacePatch,
    boundary_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    if boundary_id == "left":
        return patch.point_at(u0, v0), patch.point_at(u0, v1)
    if boundary_id == "right":
        return patch.point_at(u1, v0), patch.point_at(u1, v1)
    if boundary_id == "bottom":
        return patch.point_at(u0, v0), patch.point_at(u1, v0)
    if boundary_id == "top":
        return patch.point_at(u0, v1), patch.point_at(u1, v1)
    raise ValueError(f"Unsupported boundary_id {boundary_id!r} for planar boundary endpoints.")


def _segment_key(
    first: np.ndarray,
    second: np.ndarray,
    *,
    decimals: int = 12,
) -> tuple[object, ...]:
    points = tuple(
        sorted(
            (
                tuple(float(value) for value in np.round(np.asarray(first, dtype=float), decimals=decimals)),
                tuple(float(value) for value in np.round(np.asarray(second, dtype=float), decimals=decimals)),
            )
        )
    )
    return ("segment", *points[0], *points[1])


def _result_cell_occupied(
    operation: SurfaceBooleanOperation,
    point: Sequence[float],
    left_bounds: tuple[float, float, float, float, float, float],
    right_bounds: tuple[float, float, float, float, float, float],
) -> bool:
    in_left = _point_inside_bounds(point, left_bounds)
    in_right = _point_inside_bounds(point, right_bounds)
    if operation == "union":
        return in_left or in_right
    if operation == "difference":
        return in_left and not in_right
    raise ValueError(f"Unsupported orthogonal surfaced boolean operation {operation!r}.")


def _occupied_cell_components(
    occupied: set[tuple[int, int, int]],
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    remaining = set(occupied)
    components: list[tuple[tuple[int, int, int], ...]] = []
    while remaining:
        seed = next(iter(remaining))
        stack = [seed]
        component: list[tuple[int, int, int]] = []
        remaining.remove(seed)
        while stack:
            ix, iy, iz = stack.pop()
            component.append((ix, iy, iz))
            for neighbor in (
                (ix - 1, iy, iz),
                (ix + 1, iy, iz),
                (ix, iy - 1, iz),
                (ix, iy + 1, iz),
                (ix, iy, iz - 1),
                (ix, iy, iz + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _surface_orthogonal_box_boolean_body(
    left_bounds: tuple[float, float, float, float, float, float],
    right_bounds: tuple[float, float, float, float, float, float],
    *,
    operation: SurfaceBooleanOperation,
    metadata: dict[str, object],
) -> SurfaceBody | None:
    xs = _sorted_unique_axis_values((left_bounds[0], left_bounds[1], right_bounds[0], right_bounds[1]))
    ys = _sorted_unique_axis_values((left_bounds[2], left_bounds[3], right_bounds[2], right_bounds[3]))
    zs = _sorted_unique_axis_values((left_bounds[4], left_bounds[5], right_bounds[4], right_bounds[5]))
    if len(xs) < 2 or len(ys) < 2 or len(zs) < 2:
        return None

    occupied: set[tuple[int, int, int]] = set()
    for ix in range(len(xs) - 1):
        for iy in range(len(ys) - 1):
            for iz in range(len(zs) - 1):
                midpoint = (
                    (xs[ix] + xs[ix + 1]) * 0.5,
                    (ys[iy] + ys[iy + 1]) * 0.5,
                    (zs[iz] + zs[iz + 1]) * 0.5,
                )
                if _result_cell_occupied(operation, midpoint, left_bounds, right_bounds):
                    occupied.add((ix, iy, iz))
    if not occupied or len(_occupied_cell_components(occupied)) != 1:
        return None

    faces: list[_RectilinearBoundaryFace] = []
    for ix, iy, iz in sorted(occupied):
        mins = (xs[ix], ys[iy], zs[iz])
        maxs = (xs[ix + 1], ys[iy + 1], zs[iz + 1])
        neighbors = {
            (0, "min"): (ix - 1, iy, iz),
            (0, "max"): (ix + 1, iy, iz),
            (1, "min"): (ix, iy - 1, iz),
            (1, "max"): (ix, iy + 1, iz),
            (2, "min"): (ix, iy, iz - 1),
            (2, "max"): (ix, iy, iz + 1),
        }
        for (axis, side), neighbor in neighbors.items():
            if neighbor in occupied:
                continue
            faces.append(
                _RectilinearBoundaryFace(
                    axis=axis,
                    side=side,
                    cell_index=(ix, iy, iz),
                    mins=mins,
                    maxs=maxs,
                )
            )

    ordered_faces = tuple(sorted(faces, key=_rectilinear_boundary_sort_key))
    patches = tuple(
        _surface_rectangular_patch_from_face(face, operation=operation)
        for face in ordered_faces
    )

    segment_map: dict[tuple[object, ...], list[SurfaceBoundaryRef]] = {}
    for patch_index, patch in enumerate(patches):
        for boundary_id in ("left", "right", "bottom", "top"):
            first, second = _patch_boundary_endpoints(patch, boundary_id)
            segment_map.setdefault(_segment_key(first, second), []).append(
                SurfaceBoundaryRef(patch_index, boundary_id)
            )

    seam_objects = []
    for seam_index, segment_key in enumerate(sorted(segment_map)):
        boundaries = tuple(sorted(segment_map[segment_key], key=lambda ref: (ref.patch_index, ref.boundary_id)))
        if len(boundaries) != 2:
            return None
        seam_objects.append(
            SurfaceSeam(
                seam_id=f"orthogonal-boolean-seam-{seam_index:03d}",
                boundaries=boundaries,
            )
        )
    seam_objects = tuple(seam_objects)

    adjacency: dict[int, set[int]] = {patch_index: set() for patch_index in range(len(patches))}
    for seam in seam_objects:
        if len(seam.boundaries) != 2:
            return None
        first, second = seam.boundaries
        adjacency[first.patch_index].add(second.patch_index)
        adjacency[second.patch_index].add(first.patch_index)
    if patches:
        visited = set()
        stack = [0]
        while stack:
            patch_index = stack.pop()
            if patch_index in visited:
                continue
            visited.add(patch_index)
            stack.extend(sorted(adjacency[patch_index] - visited))
        if len(visited) != len(patches):
            return None

    shell = make_surface_shell(
        patches,
        connected=True,
        seams=seam_objects,
        metadata={"kernel": {"primitive_family": "orthogonal_boolean", "boolean_operation": operation}},
    )
    return make_surface_body((shell,), metadata=metadata)


@dataclass(frozen=True)
class _AxisAlignedPlanarPatch:
    operand_index: int
    patch_index: int
    patch: PlanarSurfacePatch
    axis: int
    coordinate: float
    min_corner: np.ndarray
    max_corner: np.ndarray


def _axis_name(axis_index: int) -> str:
    return ("x", "y", "z")[axis_index]


def _planar_patch_point_to_uv(patch: PlanarSurfacePatch, point: Sequence[float]) -> tuple[float, float]:
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    anchor = patch.point_at(u0, v0)
    du, dv = patch.derivatives_at(u0, v0)
    basis = np.column_stack((du, dv))
    delta = np.asarray(point, dtype=float).reshape(3) - anchor
    params, *_rest = np.linalg.lstsq(basis, delta, rcond=None)
    return (float(u0 + params[0]), float(v0 + params[1]))


def _extract_axis_aligned_planar_patches(body: SurfaceBody, *, operand_index: int) -> tuple[_AxisAlignedPlanarPatch, ...] | None:
    if body.shell_count != 1:
        return None
    shell = body.iter_shells(world=True)[0]
    if shell.patch_count != 6:
        return None
    extracted: list[_AxisAlignedPlanarPatch] = []
    for patch_index, patch in enumerate(shell.iter_patches(world=True)):
        if not isinstance(patch, PlanarSurfacePatch) or patch.trim_loops:
            return None
        u0, u1 = patch.domain.u_range
        v0, v1 = patch.domain.v_range
        corners = np.asarray(
            [
                patch.point_at(u0, v0),
                patch.point_at(u0, v1),
                patch.point_at(u1, v0),
                patch.point_at(u1, v1),
            ],
            dtype=float,
        )
        du, dv = patch.derivatives_at((u0 + u1) * 0.5, (v0 + v1) * 0.5)
        normal = np.cross(du, dv)
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            return None
        normal /= norm
        axis = int(np.argmax(np.abs(normal)))
        if not np.allclose(np.abs(normal), np.eye(3, dtype=float)[axis], atol=1e-8):
            return None
        coordinate = float(corners[:, axis].mean())
        if not np.allclose(corners[:, axis], coordinate, atol=1e-8):
            return None
        extracted.append(
            _AxisAlignedPlanarPatch(
                operand_index=operand_index,
                patch_index=patch_index,
                patch=patch,
                axis=axis,
                coordinate=coordinate,
                min_corner=corners.min(axis=0),
                max_corner=corners.max(axis=0),
            )
        )
    return tuple(extracted)


def _classify_patch_against_bounds(
    patch: PlanarSurfacePatch,
    bounds: tuple[float, float, float, float, float, float],
    *,
    epsilon: float = 1e-9,
) -> SurfaceBooleanPatchRelation:
    u = float(sum(patch.domain.u_range) * 0.5)
    v = float(sum(patch.domain.v_range) * 0.5)
    point = patch.point_at(u, v)
    return classify_surface_csg_point_against_bounds(point, bounds, epsilon=epsilon)


def select_surface_csg_fragment_sample(
    patch: PlanarSurfacePatch,
    *,
    trim_loop: TrimLoop | None = None,
) -> tuple[float, float]:
    """Select a deterministic patch-local sample point for fragment classification."""

    if trim_loop is not None:
        points = np.asarray(trim_loop.points_uv, dtype=float)
        centroid = points.mean(axis=0)
        return (float(centroid[0]), float(centroid[1]))
    return (
        float(sum(patch.domain.u_range) * 0.5),
        float(sum(patch.domain.v_range) * 0.5),
    )


def classify_surface_csg_point_against_bounds(
    point: Sequence[float],
    bounds: tuple[float, float, float, float, float, float],
    *,
    epsilon: float = 1e-9,
) -> SurfaceBooleanPatchRelation:
    """Classify one surface-native sample point against an opposing closed bounds proxy."""

    point_array = np.asarray(point, dtype=float).reshape(3)
    inside = (
        (bounds[0] + epsilon) < point_array[0] < (bounds[1] - epsilon)
        and (bounds[2] + epsilon) < point_array[1] < (bounds[3] - epsilon)
        and (bounds[4] + epsilon) < point_array[2] < (bounds[5] - epsilon)
    )
    if inside:
        return "inside"
    on = (
        (bounds[0] - epsilon) <= point_array[0] <= (bounds[1] + epsilon)
        and (bounds[2] - epsilon) <= point_array[1] <= (bounds[3] + epsilon)
        and (bounds[4] - epsilon) <= point_array[2] <= (bounds[5] + epsilon)
        and (
            abs(point_array[0] - bounds[0]) <= epsilon
            or abs(point_array[0] - bounds[1]) <= epsilon
            or abs(point_array[1] - bounds[2]) <= epsilon
            or abs(point_array[1] - bounds[3]) <= epsilon
            or abs(point_array[2] - bounds[4]) <= epsilon
            or abs(point_array[2] - bounds[5]) <= epsilon
        )
    )
    return "on" if on else "outside"


def classify_surface_csg_fragment_against_body(
    patch_ref: SurfaceBooleanPatchRef,
    patch: PlanarSurfacePatch,
    opposing_body: SurfaceBody,
    *,
    trim_loop: TrimLoop | None = None,
    cut_curve_ids: Sequence[str] = (),
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> SurfaceCSGFragmentClassificationRecord:
    """Classify one split fragment against the opposing surface body."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    sample_uv = select_surface_csg_fragment_sample(patch, trim_loop=trim_loop)
    diagnostics: list[SurfaceCSGFragmentClassificationDiagnostic] = []
    if not patch.domain.contains(sample_uv[0], sample_uv[1]):
        diagnostics.append(
            SurfaceCSGFragmentClassificationDiagnostic(
                code="outside-domain",
                message="Fragment classification sample is outside the patch domain.",
                patch=patch_ref,
            )
        )
        sample_point = (float("nan"), float("nan"), float("nan"))
        relation: SurfaceBooleanPatchRelation = "outside"
    else:
        point_array = patch.point_at(*sample_uv)
        sample_point = (float(point_array[0]), float(point_array[1]), float(point_array[2]))
        try:
            if _classify_surface_body(opposing_body) != "closed":
                diagnostics.append(
                    SurfaceCSGFragmentClassificationDiagnostic(
                        code="open-body",
                        message="Opposing body is not closed-valid for CSG containment classification.",
                        patch=patch_ref,
                        sample_point=sample_point,
                    )
                )
        except Exception:
            diagnostics.append(
                SurfaceCSGFragmentClassificationDiagnostic(
                    code="open-body",
                    message="Opposing body classification failed during CSG containment classification.",
                    patch=patch_ref,
                    sample_point=sample_point,
                )
            )
        relation = classify_surface_csg_point_against_bounds(
            sample_point,
            opposing_body.bounds_estimate(),
            epsilon=normalized_policy.equality_tolerance,
        )
        if relation == "on" and not cut_curve_ids:
            diagnostics.append(
                SurfaceCSGFragmentClassificationDiagnostic(
                    code="ambiguous-boundary",
                    message="Fragment sample lies on the opposing boundary without cut-curve provenance.",
                    patch=patch_ref,
                    sample_point=sample_point,
                )
            )
    return SurfaceCSGFragmentClassificationRecord(
        patch=patch_ref,
        relation=relation,
        sample_uv=sample_uv,
        sample_point=sample_point,
        cut_curve_ids=tuple(sorted(cut_curve_ids)),
        diagnostics=tuple(diagnostics),
    )


def _surface_boolean_split_role(
    operation: SurfaceBooleanOperation,
    *,
    operand_index: int,
    relation: SurfaceBooleanPatchRelation,
) -> SurfaceBooleanSplitRole:
    selection = select_surface_csg_operation_fragment(
        operation,
        SurfaceCSGFragmentClassificationRecord(
            patch=SurfaceBooleanPatchRef(operand_index, -1),
            relation=relation,
            sample_uv=(0.0, 0.0),
            sample_point=(0.0, 0.0, 0.0),
        ),
    )
    return selection.role


def select_surface_csg_operation_fragment(
    operation: SurfaceBooleanOperation,
    classification: SurfaceCSGFragmentClassificationRecord,
) -> SurfaceCSGOperationSelectionRecord:
    """Apply operation-specific survive/discard/cut-cap rules to one fragment."""

    relation = classification.relation
    if operation == "union":
        role: SurfaceBooleanSplitRole = "discard" if relation == "inside" else "survive"
        cap_required = False
        reason = "union keeps exterior and boundary fragments"
    elif operation == "intersection":
        role = "discard" if relation == "outside" else "survive"
        cap_required = False
        reason = "intersection keeps interior and boundary fragments"
    elif classification.patch.operand_index == 0:
        role = "discard" if relation == "inside" else "survive"
        cap_required = False
        reason = "difference keeps base exterior and boundary fragments"
    else:
        role = "discard" if relation == "outside" else "cut_cap"
        cap_required = role == "cut_cap"
        reason = "difference converts cutter interior and boundary fragments into cut caps"
    cut_cap = SurfaceCSGCutCapRequirementRecord(
        patch=classification.patch,
        required=cap_required,
        reason=reason,
        cut_curve_ids=classification.cut_curve_ids,
    )
    return SurfaceCSGOperationSelectionRecord(
        operation=operation,
        patch=classification.patch,
        relation=relation,
        role=role,
        cut_cap=cut_cap,
        cut_curve_ids=classification.cut_curve_ids,
    )


def select_surface_csg_operation_fragments(
    operation: SurfaceBooleanOperation,
    classifications: Sequence[SurfaceCSGFragmentClassificationRecord],
) -> tuple[SurfaceCSGOperationSelectionRecord, ...]:
    """Return deterministic operation-selection records for classified fragments."""

    return tuple(
        sorted(
            (select_surface_csg_operation_fragment(operation, classification) for classification in classifications),
            key=lambda record: (record.patch.operand_index, record.patch.patch_index, record.relation, record.role),
        )
    )


def surface_csg_selection_is_empty(selections: Sequence[SurfaceCSGOperationSelectionRecord]) -> bool:
    """Return whether operation selection produces an explicit empty result."""

    return not any(selection.survives for selection in selections)


def _cut_curve_id(
    first: SurfaceBooleanPatchRef,
    second: SurfaceBooleanPatchRef,
    points_3d: tuple[tuple[float, float, float], ...],
) -> str:
    return (
        f"operand{first.operand_index}:patch{first.patch_index}|"
        f"operand{second.operand_index}:patch{second.patch_index}|"
        f"{points_3d[0]}->{points_3d[-1]}"
    )


def _intersect_axis_aligned_patch_pair(
    first: _AxisAlignedPlanarPatch,
    second: _AxisAlignedPlanarPatch,
    *,
    epsilon: float = 1e-9,
) -> SurfaceBooleanCutCurve | None:
    first_ref = SurfaceBooleanPatchRef(first.operand_index, first.patch_index)
    second_ref = SurfaceBooleanPatchRef(second.operand_index, second.patch_index)
    record = intersect_planar_linear_patch_pair(
        first_ref,
        first.patch,
        second_ref,
        second.patch,
        policy={"degeneracy_tolerance": epsilon, "domain_tolerance": epsilon},
    )
    if not record.supported or record.curve is None or len(record.patch_local_curves) != 2:
        return None
    first_trim = SurfaceBooleanTrimFragment(
        patch=first_ref,
        points_uv=record.patch_local_curves[0].points_uv,
    )
    second_trim = SurfaceBooleanTrimFragment(
        patch=second_ref,
        points_uv=record.patch_local_curves[1].points_uv,
    )
    return SurfaceBooleanCutCurve(
        cut_curve_id=_cut_curve_id(first_ref, second_ref, record.curve.points_3d),
        points_3d=record.curve.points_3d,
        patches=(first_ref, second_ref),
        trim_fragments=(first_trim, second_trim),
        curve=record.curve,
        patch_local_curves=record.patch_local_curves,
    )


def _sorted_cut_curve_ids_for_patch(
    cut_curves: Sequence[SurfaceBooleanCutCurve],
    patch: SurfaceBooleanPatchRef,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            curve.cut_curve_id
            for curve in cut_curves
            if any(
                ref.operand_index == patch.operand_index and ref.patch_index == patch.patch_index
                for ref in curve.patches
            )
        )
    )


def _patch_local_curves_for_patch(
    cut_curves: Sequence[SurfaceBooleanCutCurve],
    patch: SurfaceBooleanPatchRef,
) -> tuple[SurfaceCSGPatchLocalCurve, ...]:
    curves: list[SurfaceCSGPatchLocalCurve] = []
    for cut_curve in cut_curves:
        curves.extend(
            local_curve
            for local_curve in cut_curve.patch_local_curves
            if local_curve.patch.operand_index == patch.operand_index
            and local_curve.patch.patch_index == patch.patch_index
        )
    return tuple(
        sorted(
            curves,
            key=lambda curve: (curve.source_curve_digest, curve.points_uv),
        )
    )


def _trim_loop_for_overlap_fragment(
    fragment: _AxisAlignedPlanarPatch,
    overlap_bounds: tuple[float, float, float, float, float, float],
) -> TrimLoop:
    corners: list[tuple[float, float, float]] = []
    for first in (0, 1):
        for second in (0, 1):
            point = [0.0, 0.0, 0.0]
            point[fragment.axis] = fragment.coordinate
            other_axes = [axis for axis in (0, 1, 2) if axis != fragment.axis]
            point[other_axes[0]] = overlap_bounds[(other_axes[0] * 2) + first]
            point[other_axes[1]] = overlap_bounds[(other_axes[1] * 2) + second]
            corners.append((point[0], point[1], point[2]))
    ordered_corners = (corners[0], corners[1], corners[3], corners[2])
    u0, v0 = _planar_patch_point_to_uv(fragment.patch, ordered_corners[0])
    points_uv = [(u0, v0)]
    for point in ordered_corners[1:]:
        points_uv.append(_planar_patch_point_to_uv(fragment.patch, point))
    return TrimLoop(points_uv, category="outer").normalized()


def _boundary_contributor_patch(
    patches: Sequence[_AxisAlignedPlanarPatch],
    *,
    axis: int,
    coordinate: float,
    epsilon: float = 1e-9,
) -> _AxisAlignedPlanarPatch | None:
    matches = [
        patch
        for patch in patches
        if patch.axis == axis and abs(patch.coordinate - coordinate) <= epsilon
    ]
    if not matches:
        return None
    return min(matches, key=lambda patch: (patch.operand_index, patch.patch_index))


def _trim_loop_min_edge_length(loop: TrimLoop) -> float:
    points = np.asarray(loop.points_uv, dtype=float)
    if len(points) < 2:
        return 0.0
    closed = np.vstack((points, points[0]))
    distances = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    return float(np.min(distances))


def _arrangement_diagnostic(
    code: Literal["ambiguous-overlap", "self-intersection", "zero-length-fragment", "outside-domain"],
    message: str,
    *,
    patch_ref: SurfaceBooleanPatchRef,
    cut_curve_ids: tuple[str, ...] = (),
) -> SurfaceCSGArrangementDiagnostic:
    return SurfaceCSGArrangementDiagnostic(
        code=code,
        message=message,
        patch=patch_ref,
        cut_curve_ids=cut_curve_ids,
    )


def build_surface_csg_patch_arrangement(
    patch_ref: SurfaceBooleanPatchRef,
    patch: PlanarSurfacePatch,
    *,
    patch_local_curves: Sequence[SurfaceCSGPatchLocalCurve] = (),
    trim_loops: Sequence[TrimLoop] = (),
    generated_loop: TrimLoop | None = None,
    cut_curve_ids: Sequence[str] = (),
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> SurfaceCSGPatchLocalArrangementGraph:
    """Build a deterministic patch-local arrangement and split trim loop records."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    diagnostics: list[SurfaceCSGArrangementDiagnostic] = []
    local_curves = tuple(
        sorted(
            patch_local_curves,
            key=lambda curve: (
                curve.source_curve_digest,
                curve.patch.operand_index,
                curve.patch.patch_index,
                curve.points_uv,
            ),
        )
    )
    for curve in local_curves:
        for diagnostic in validate_surface_csg_patch_local_curve_domain(curve, policy=normalized_policy):
            diagnostics.append(
                _arrangement_diagnostic(
                    "outside-domain",
                    diagnostic.message,
                    patch_ref=patch_ref,
                    cut_curve_ids=tuple(cut_curve_ids),
                )
            )
    loops = tuple(trim_loops)
    if generated_loop is not None:
        loops = (*loops, generated_loop)
    if not loops:
        loops = patch.trim_loops
    split_loops: list[SurfaceCSGSplitTrimLoopRecord] = []
    for loop in loops:
        normalized_loop = loop.normalized()
        if _trim_loop_min_edge_length(normalized_loop) <= normalized_policy.degeneracy_tolerance:
            diagnostics.append(
                _arrangement_diagnostic(
                    "zero-length-fragment",
                    "Trim loop splitting produced a zero-length fragment.",
                    patch_ref=patch_ref,
                    cut_curve_ids=tuple(cut_curve_ids),
                )
            )
            continue
        split_loops.append(
            SurfaceCSGSplitTrimLoopRecord(
                patch=patch_ref,
                loop=normalized_loop,
                source_category=normalized_loop.category,
                cut_curve_ids=tuple(sorted(cut_curve_ids)),
            )
        )
    return SurfaceCSGPatchLocalArrangementGraph(
        patch=patch_ref,
        patch_local_curves=local_curves,
        split_loops=tuple(split_loops),
        diagnostics=tuple(diagnostics),
    )


def surface_boolean_overlap_fragments(operands: SurfaceBooleanOperands) -> tuple[SurfaceBooleanTrimmedPatchFragment, ...]:
    """Reconstruct trimmed planar overlap fragments for the initial box intersection slice."""

    if operands.operation != "intersection":
        return ()
    stage = surface_boolean_intersection_stage(operands)
    if not stage.supported or stage.body_relation != "overlap" or operands.operand_count != 2:
        return ()

    left, right = operands.bodies
    left_patches = _extract_axis_aligned_planar_patches(left, operand_index=0)
    right_patches = _extract_axis_aligned_planar_patches(right, operand_index=1)
    if left_patches is None or right_patches is None:
        return ()

    overlap_bounds = _aabb_overlap(left.bounds_estimate(), right.bounds_estimate())
    all_patches = (*left_patches, *right_patches)
    fragments: list[SurfaceBooleanTrimmedPatchFragment] = []
    for axis in (0, 1, 2):
        for coordinate in (overlap_bounds[axis * 2], overlap_bounds[(axis * 2) + 1]):
            contributor = _boundary_contributor_patch(all_patches, axis=axis, coordinate=coordinate)
            if contributor is None:
                continue
            source_ref = SurfaceBooleanPatchRef(contributor.operand_index, contributor.patch_index)
            cut_curve_ids = _sorted_cut_curve_ids_for_patch(stage.cut_curves, source_ref)
            arrangement = build_surface_csg_patch_arrangement(
                source_ref,
                contributor.patch,
                patch_local_curves=_patch_local_curves_for_patch(stage.cut_curves, source_ref),
                generated_loop=_trim_loop_for_overlap_fragment(contributor, overlap_bounds),
                cut_curve_ids=cut_curve_ids,
            )
            if not arrangement.supported or not arrangement.split_loops:
                continue
            trimmed_patch = replace(
                contributor.patch,
                trim_loops=tuple(record.loop for record in arrangement.split_loops),
            )
            fragments.append(
                SurfaceBooleanTrimmedPatchFragment(
                    source_patch=source_ref,
                    patch=trimmed_patch,
                    cut_curve_ids=cut_curve_ids,
                )
            )
    return tuple(
        sorted(
            fragments,
            key=lambda fragment: (fragment.source_patch.operand_index, fragment.source_patch.patch_index),
        )
    )


def _surface_csg_patch_ref_sort_key(patch_ref: SurfaceBooleanPatchRef) -> tuple[int, int]:
    return (int(patch_ref.operand_index), int(patch_ref.patch_index))


def _surface_csg_shell_sort_key(
    patches_with_sources: Sequence[tuple[object, SurfaceBooleanPatchRef, tuple[str, ...]]],
) -> tuple[object, ...]:
    if not patches_with_sources:
        return ("empty",)
    return (
        len(patches_with_sources),
        tuple(_surface_csg_patch_ref_sort_key(source) for _patch, source, _cut_ids in patches_with_sources),
        tuple(getattr(patch, "family", patch.__class__.__name__) for patch, _source, _cut_ids in patches_with_sources),
    )


def _surface_csg_kernel_patch_source(source_patch: SurfaceBooleanPatchRef, *, role: str) -> dict[str, object]:
    return {
        "generated_role": role,
        "source_operand_index": source_patch.operand_index,
        "source_patch_index": source_patch.patch_index,
    }


def _surface_csg_patch_with_kernel_source(
    patch: object,
    *,
    source_patch: SurfaceBooleanPatchRef,
    role: str,
    cut_curve_ids: Sequence[str] = (),
):
    if not hasattr(patch, "metadata"):
        return patch
    metadata = dict(getattr(patch, "metadata"))
    kernel = dict(patch.kernel_metadata())  # type: ignore[attr-defined]
    consumer = dict(patch.consumer_metadata())  # type: ignore[attr-defined]
    kernel.update(_surface_csg_kernel_patch_source(source_patch, role=role))
    if cut_curve_ids:
        kernel["cut_curve_ids"] = tuple(sorted(cut_curve_ids))
    metadata["kernel"] = kernel
    if consumer:
        metadata["consumer"] = consumer
    try:
        return replace(patch, metadata=metadata)
    except TypeError:
        return patch


def _surface_csg_shell_ordering_record(
    *,
    result_shell_index: int,
    patches_with_sources: Sequence[tuple[object, SurfaceBooleanPatchRef, tuple[str, ...]]],
) -> SurfaceCSGShellOrderingRecord:
    return SurfaceCSGShellOrderingRecord(
        result_shell_index=result_shell_index,
        patch_count=len(patches_with_sources),
        source_patches=tuple(source for _patch, source, _cut_ids in patches_with_sources),
        sort_key=_surface_csg_shell_sort_key(patches_with_sources),
    )


def assemble_surface_csg_shells_from_fragments(
    operation: SurfaceBooleanOperation,
    fragments: Sequence[SurfaceBooleanTrimmedPatchFragment],
    *,
    multi_shell: bool = False,
) -> SurfaceCSGShellAssemblyRecord:
    """Assemble selected CSG fragments into provisional result shells."""

    if not fragments:
        return SurfaceCSGShellAssemblyRecord(operation=operation, classification="empty")
    sorted_fragments = tuple(
        sorted(
            fragments,
            key=lambda fragment: _surface_csg_patch_ref_sort_key(fragment.source_patch),
        )
    )
    if multi_shell:
        shells = tuple(
            make_surface_shell(
                (fragment.patch,),
                connected=True,
                metadata={"kernel": {"source_patch": fragment.source_patch}},
            )
            for fragment in sorted_fragments
        )
        shell_ordering = tuple(
            _surface_csg_shell_ordering_record(
                result_shell_index=index,
                patches_with_sources=((fragment.patch, fragment.source_patch, fragment.cut_curve_ids),),
            )
            for index, fragment in enumerate(sorted_fragments)
        )
        provenance = tuple(
            SurfaceCSGFragmentProvenanceRecord(
                source_patch=fragment.source_patch,
                result_shell_index=index,
                result_patch_index=0,
                cut_curve_ids=fragment.cut_curve_ids,
            )
            for index, fragment in enumerate(sorted_fragments)
        )
    else:
        patches_with_sources = tuple(
            (fragment.patch, fragment.source_patch, fragment.cut_curve_ids)
            for fragment in sorted_fragments
        )
        shell = make_surface_shell(
            tuple(fragment.patch for fragment in sorted_fragments),
            connected=True,
            metadata={"kernel": {"primitive_family": "csg_fragment_assembly", "boolean_operation": operation}},
        )
        shells = (shell,)
        shell_ordering = (
            _surface_csg_shell_ordering_record(
                result_shell_index=0,
                patches_with_sources=patches_with_sources,
            ),
        )
        provenance = tuple(
            SurfaceCSGFragmentProvenanceRecord(
                source_patch=fragment.source_patch,
                result_shell_index=0,
                result_patch_index=index,
                cut_curve_ids=fragment.cut_curve_ids,
            )
            for index, fragment in enumerate(sorted_fragments)
        )
    return SurfaceCSGShellAssemblyRecord(
        operation=operation,
        classification="closed",
        shells=shells,
        provenance=provenance,
        shell_ordering=shell_ordering,
    )


def assemble_surface_csg_result_shells(
    graph: SurfaceCSGFragmentGraphRecord,
    cap_construction: SurfaceCSGCapConstructionRecord,
    cut_boundary: SurfaceCSGCutBoundaryRecord,
    *,
    multi_shell: bool = False,
) -> SurfaceCSGShellAssemblyRecord:
    """Assemble surviving fragments and generated caps into durable result shells."""

    diagnostics: list[SurfaceCSGReconstructionDiagnostic] = []
    if not graph.supported or graph.plan.operands is None:
        diagnostics.append(
            SurfaceCSGReconstructionDiagnostic(
                code="invalid-fragment-graph",
                message="Surface CSG result shell assembly requires a supported fragment graph.",
            )
        )
        return SurfaceCSGShellAssemblyRecord(
            operation=graph.operation,
            classification="empty",
            diagnostics=tuple(diagnostics),
        )
    if not cap_construction.supported:
        diagnostics.extend(
            SurfaceCSGReconstructionDiagnostic(
                code="missing-cap-payload",
                message=diagnostic.message,
                source_patch=diagnostic.patch,
            )
            for diagnostic in cap_construction.diagnostics
        )
    if not cut_boundary.supported:
        diagnostics.extend(
            SurfaceCSGReconstructionDiagnostic(
                code="invalid-cut-boundary",
                message=diagnostic.message,
                source_patch=diagnostic.source_patch,
                cap_payload_index=diagnostic.cap_payload_index,
            )
            for diagnostic in cut_boundary.diagnostics
        )
    if diagnostics:
        return SurfaceCSGShellAssemblyRecord(
            operation=graph.operation,
            classification="empty",
            diagnostics=tuple(diagnostics),
        )

    patches_with_sources: list[tuple[object, SurfaceBooleanPatchRef, tuple[str, ...]]] = []
    for edge in graph.classification_edges:
        if edge.role != "survive":
            continue
        source_patch = _surface_csg_patch_for_ref(graph.plan.operands, edge.patch)
        if source_patch is None:
            diagnostics.append(
                SurfaceCSGReconstructionDiagnostic(
                    code="missing-source-patch",
                    source_patch=edge.patch,
                    message=(
                        "Surface CSG result shell assembly could not resolve surviving "
                        f"operand {edge.patch.operand_index} patch {edge.patch.patch_index}."
                    ),
                )
            )
            continue
        patches_with_sources.append(
            (
                _surface_csg_patch_with_kernel_source(
                    source_patch,
                    source_patch=edge.patch,
                    role="csg_surviving_fragment",
                    cut_curve_ids=edge.cut_curve_ids,
                ),
                edge.patch,
                tuple(sorted(edge.cut_curve_ids)),
            )
        )

    attachments_by_payload = {attachment.cap_payload_index: attachment for attachment in cut_boundary.trim_attachments}
    for cap_index, payload in enumerate(cap_construction.cap_payloads):
        attachment = attachments_by_payload.get(cap_index)
        if attachment is None:
            diagnostics.append(
                SurfaceCSGReconstructionDiagnostic(
                    code="missing-cap-payload",
                    source_patch=payload.source_patch,
                    cap_payload_index=cap_index,
                    message=f"Surface CSG result shell assembly is missing cut-boundary attachment {cap_index}.",
                )
            )
            continue
        cap_patch = replace(
            payload.patch,
            trim_loops=(attachment.trim_loop,),
        )
        patches_with_sources.append(
            (
                _surface_csg_patch_with_kernel_source(
                    cap_patch,
                    source_patch=payload.source_patch,
                    role="csg_generated_cap",
                    cut_curve_ids=payload.cut_curve_ids,
                ),
                payload.source_patch,
                tuple(sorted(payload.cut_curve_ids)),
            )
        )

    if diagnostics:
        return SurfaceCSGShellAssemblyRecord(
            operation=graph.operation,
            classification="empty",
            diagnostics=tuple(diagnostics),
        )
    if not patches_with_sources:
        return SurfaceCSGShellAssemblyRecord(operation=graph.operation, classification="empty")

    ordered = tuple(sorted(patches_with_sources, key=lambda item: (*_surface_csg_patch_ref_sort_key(item[1]), item[2])))
    if multi_shell:
        shell_groups = tuple((item,) for item in ordered)
    else:
        shell_groups = (ordered,)

    sorted_shell_groups = tuple(sorted(shell_groups, key=_surface_csg_shell_sort_key))
    shells: list[SurfaceShell] = []
    ordering: list[SurfaceCSGShellOrderingRecord] = []
    provenance: list[SurfaceCSGFragmentProvenanceRecord] = []
    for shell_index, shell_group in enumerate(sorted_shell_groups):
        if not shell_group:
            diagnostics.append(
                SurfaceCSGReconstructionDiagnostic(
                    code="empty-shell",
                    message=f"Surface CSG result shell assembly produced empty shell {shell_index}.",
                )
            )
            continue
        shell_metadata = {
            "kernel": {
                "primitive_family": "csg_result_shell_assembly",
                "boolean_operation": graph.operation,
                "result_shell_index": shell_index,
                "source_patches": tuple(
                    {
                        "operand_index": source.operand_index,
                        "patch_index": source.patch_index,
                    }
                    for _patch, source, _cut_ids in shell_group
                ),
            }
        }
        try:
            shell = make_surface_shell(
                tuple(patch for patch, _source, _cut_ids in shell_group),
                connected=True,
                metadata=shell_metadata,
            )
        except (TypeError, ValueError) as exc:
            diagnostics.append(
                SurfaceCSGReconstructionDiagnostic(
                    code="assembly-error",
                    message=f"Surface CSG result shell assembly failed to create shell {shell_index}: {exc}",
                )
            )
            continue
        shells.append(shell)
        ordering.append(
            _surface_csg_shell_ordering_record(
                result_shell_index=shell_index,
                patches_with_sources=shell_group,
            )
        )
        provenance.extend(
            SurfaceCSGFragmentProvenanceRecord(
                source_patch=source,
                result_shell_index=shell_index,
                result_patch_index=patch_index,
                cut_curve_ids=cut_ids,
            )
            for patch_index, (_patch, source, cut_ids) in enumerate(shell_group)
        )

    if diagnostics:
        return SurfaceCSGShellAssemblyRecord(
            operation=graph.operation,
            classification="empty",
            diagnostics=tuple(diagnostics),
        )
    if not shells:
        return SurfaceCSGShellAssemblyRecord(operation=graph.operation, classification="empty")
    return SurfaceCSGShellAssemblyRecord(
        operation=graph.operation,
        classification="closed",
        shells=tuple(shells),
        provenance=tuple(provenance),
        shell_ordering=tuple(ordering),
    )


def surface_boolean_intersection_stage(operands: SurfaceBooleanOperands) -> SurfaceBooleanIntersectionStage:
    """Compute the first bounded surfaced intersection/classification stage."""

    if operands.operand_count != 2:
        return SurfaceBooleanIntersectionStage(
            operation=operands.operation,
            operands=operands,
            supported=False,
            body_relation="disjoint",
            support_reason="The initial surfaced boolean intersection stage supports exactly two operands.",
        )

    left, right = operands.bodies
    left_bounds = left.bounds_estimate()
    right_bounds = right.bounds_estimate()
    relation = _surface_boolean_body_relation(left_bounds, right_bounds)
    left_patches = _extract_axis_aligned_planar_patches(left, operand_index=0)
    right_patches = _extract_axis_aligned_planar_patches(right, operand_index=1)
    if left_patches is None or right_patches is None:
        return SurfaceBooleanIntersectionStage(
            operation=operands.operation,
            operands=operands,
            supported=False,
            body_relation=relation,
            support_reason=(
                "The initial surfaced boolean intersection stage currently supports only simple "
                "single-shell axis-aligned planar box-style operands without trims."
            ),
        )

    cut_curves = tuple(
        curve
        for left_patch in left_patches
        for right_patch in right_patches
        for curve in (_intersect_axis_aligned_patch_pair(left_patch, right_patch),)
        if curve is not None
    )

    cut_curve_ids_by_patch: dict[tuple[int, int], list[str]] = {}
    for curve in cut_curves:
        for patch_ref in curve.patches:
            cut_curve_ids_by_patch.setdefault((patch_ref.operand_index, patch_ref.patch_index), []).append(curve.cut_curve_id)

    patch_classifications = tuple(
        SurfaceBooleanPatchClassification(
            patch=SurfaceBooleanPatchRef(operand_index, patch.patch_index),
            relation=_classify_patch_against_bounds(
                patch.patch,
                right_bounds if operand_index == 0 else left_bounds,
            ),
            cut_curve_ids=tuple(sorted(cut_curve_ids_by_patch.get((operand_index, patch.patch_index), ()))),
        )
        for operand_index, patch_set in ((0, left_patches), (1, right_patches))
        for patch in patch_set
    )
    split_records = tuple(
        SurfaceBooleanSplitRecord(
            patch=classification.patch,
            relation=classification.relation,
            role=_surface_boolean_split_role(
                operands.operation,
                operand_index=classification.patch.operand_index,
                relation=classification.relation,
            ),
            cut_curve_ids=classification.cut_curve_ids,
        )
        for classification in patch_classifications
    )
    return SurfaceBooleanIntersectionStage(
        operation=operands.operation,
        operands=operands,
        supported=True,
        body_relation=relation,
        cut_curves=tuple(sorted(cut_curves, key=lambda curve: curve.cut_curve_id)),
        patch_classifications=patch_classifications,
        split_records=split_records,
    )


def build_surface_csg_fragment_graph(
    plan: SurfaceCSGOperationPlan,
    *,
    intersection_stage: SurfaceBooleanIntersectionStage | None = None,
) -> SurfaceCSGFragmentGraphRecord:
    """Build the transient classified fragment graph for a planned CSG operation."""

    diagnostics: list[SurfaceCSGFragmentGraphDiagnostic] = []
    if not plan.executable or plan.operands is None:
        diagnostics.append(
            SurfaceCSGFragmentGraphDiagnostic(
                code="non-executable-plan",
                message="Surface CSG fragment graph requires an executable operation plan.",
            )
        )
        return SurfaceCSGFragmentGraphRecord(
            operation=plan.operation,
            plan=plan,
            diagnostics=tuple(diagnostics),
        )

    stage = intersection_stage if intersection_stage is not None else surface_boolean_intersection_stage(plan.operands)
    if not stage.supported:
        diagnostics.append(
            SurfaceCSGFragmentGraphDiagnostic(
                code="unsupported-intersection-stage",
                message=stage.support_reason or "Surface CSG intersection stage is unsupported.",
            )
        )
        return SurfaceCSGFragmentGraphRecord(
            operation=plan.operation,
            plan=plan,
            intersection_stage=stage,
            diagnostics=tuple(diagnostics),
        )

    selection_by_patch = {record.patch: record for record in stage.split_records}
    edges: list[SurfaceCSGFragmentClassificationEdgeRecord] = []
    for classification in stage.patch_classifications:
        selection = selection_by_patch.get(classification.patch)
        if selection is None:
            diagnostics.append(
                SurfaceCSGFragmentGraphDiagnostic(
                    code="missing-selection",
                    message=(
                        "Surface CSG fragment graph is missing an operation selection for "
                        f"operand {classification.patch.operand_index} patch {classification.patch.patch_index}."
                    ),
                )
            )
            continue
        edges.append(
            SurfaceCSGFragmentClassificationEdgeRecord(
                patch=classification.patch,
                relation=classification.relation,
                role=selection.role,
                cut_curve_ids=tuple(sorted(classification.cut_curve_ids)),
            )
        )

    return SurfaceCSGFragmentGraphRecord(
        operation=plan.operation,
        plan=plan,
        intersection_stage=stage,
        classification_edges=tuple(edges),
        diagnostics=tuple(diagnostics),
    )


def _surface_csg_patch_for_ref(operands: SurfaceBooleanOperands, patch_ref: SurfaceBooleanPatchRef):
    try:
        shell = operands.bodies[patch_ref.operand_index].iter_shells(world=True)[0]
        return shell.iter_patches(world=True)[patch_ref.patch_index]
    except (IndexError, TypeError):
        return None


def _surface_csg_generated_cap_metadata(
    patch: PlanarSurfacePatch,
    *,
    operation: SurfaceBooleanOperation,
    edge: SurfaceCSGFragmentClassificationEdgeRecord,
) -> dict[str, object]:
    metadata = dict(patch.metadata)
    kernel = dict(patch.kernel_metadata())
    consumer = dict(patch.consumer_metadata())
    kernel.update(
        {
            "generated_role": "csg_cap",
            "boolean_operation": operation,
            "source_operand_index": edge.patch.operand_index,
            "source_patch_index": edge.patch.patch_index,
            "cut_curve_ids": edge.cut_curve_ids,
        }
    )
    metadata["kernel"] = kernel
    if consumer:
        metadata["consumer"] = consumer
    return metadata


def build_surface_csg_cap_patches(graph: SurfaceCSGFragmentGraphRecord) -> SurfaceCSGCapConstructionRecord:
    """Select cap families and construct generated surface-native cap payloads."""

    diagnostics: list[SurfaceCSGUnsupportedCapDiagnostic] = []
    payloads: list[SurfaceCSGGeneratedCapPatchPayloadRecord] = []
    if not graph.supported or graph.plan.operands is None:
        diagnostics.append(
            SurfaceCSGUnsupportedCapDiagnostic(
                code="invalid-fragment-graph",
                message="Surface CSG cap construction requires a supported fragment graph.",
            )
        )
        return SurfaceCSGCapConstructionRecord(operation=graph.operation, diagnostics=tuple(diagnostics))

    for edge in graph.classification_edges:
        if edge.role != "cut_cap":
            continue
        source_patch = _surface_csg_patch_for_ref(graph.plan.operands, edge.patch)
        if source_patch is None:
            diagnostics.append(
                SurfaceCSGUnsupportedCapDiagnostic(
                    code="missing-source-patch",
                    patch=edge.patch,
                    message=(
                        "Surface CSG cap construction could not resolve the source patch "
                        f"for operand {edge.patch.operand_index} patch {edge.patch.patch_index}."
                    ),
                )
            )
            continue
        if not isinstance(source_patch, PlanarSurfacePatch):
            diagnostics.append(
                SurfaceCSGUnsupportedCapDiagnostic(
                    code="unsupported-cap-family",
                    patch=edge.patch,
                    cap_family=source_patch.family,
                    message=(
                        f"Surface CSG cap construction supports planar cap payloads; "
                        f"source family {source_patch.family!r} requires a later cap producer."
                    ),
                )
            )
            continue
        generated_patch = replace(
            source_patch,
            capability_flags=frozenset((*source_patch.capability_flags, "generated-csg-cap")),
            metadata=_surface_csg_generated_cap_metadata(source_patch, operation=graph.operation, edge=edge),
        )
        payloads.append(
            SurfaceCSGGeneratedCapPatchPayloadRecord(
                source_patch=edge.patch,
                cap_family="planar",
                patch=generated_patch,
                cut_curve_ids=edge.cut_curve_ids,
            )
        )

    return SurfaceCSGCapConstructionRecord(
        operation=graph.operation,
        cap_payloads=tuple(payloads),
        diagnostics=tuple(diagnostics),
    )


def _surface_csg_patch_domain_outer_trim(patch: PlanarSurfacePatch) -> TrimLoop:
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    return TrimLoop(
        (
            (float(u0), float(v0)),
            (float(u1), float(v0)),
            (float(u1), float(v1)),
            (float(u0), float(v1)),
        ),
        category="outer",
    ).normalized()


def build_surface_csg_cut_boundary_trims(
    graph: SurfaceCSGFragmentGraphRecord,
    cap_construction: SurfaceCSGCapConstructionRecord,
) -> SurfaceCSGCutBoundaryRecord:
    """Attach cut-boundary trim loops to generated CSG cap payloads."""

    diagnostics: list[SurfaceCSGBoundaryExposureDiagnostic] = []
    attachments: list[SurfaceCSGTrimAttachmentRecord] = []
    if not graph.supported or not cap_construction.supported:
        diagnostics.append(
            SurfaceCSGBoundaryExposureDiagnostic(
                code="invalid-cap-construction",
                message="Surface CSG cut-boundary construction requires supported graph and cap records.",
            )
        )
        return SurfaceCSGCutBoundaryRecord(operation=graph.operation, diagnostics=tuple(diagnostics))

    for index, payload in enumerate(cap_construction.cap_payloads):
        trim_loop = payload.patch.outer_trim or _surface_csg_patch_domain_outer_trim(payload.patch)
        exposure: Literal["shared", "open"] = "shared" if payload.cut_curve_ids else "open"
        attachments.append(
            SurfaceCSGTrimAttachmentRecord(
                source_patch=payload.source_patch,
                cap_payload_index=index,
                trim_loop=trim_loop,
                cut_curve_ids=payload.cut_curve_ids,
                exposure=exposure,
            )
        )
        if exposure == "open":
            diagnostics.append(
                SurfaceCSGBoundaryExposureDiagnostic(
                    code="open-boundary",
                    source_patch=payload.source_patch,
                    cap_payload_index=index,
                    message=(
                        "Generated CSG cap payload has no cut-curve provenance; "
                        "closed-body reconstruction must treat it as an exposed boundary."
                    ),
                )
            )

    return SurfaceCSGCutBoundaryRecord(
        operation=graph.operation,
        trim_attachments=tuple(attachments),
        diagnostics=tuple(diagnostics),
    )


def _surface_boolean_supported_box_result(
    operands: SurfaceBooleanOperands,
    stage: SurfaceBooleanIntersectionStage,
) -> SurfaceBooleanResult | None:
    if not stage.supported or operands.operand_count != 2:
        return None
    left, right = operands.bodies
    left_bounds = left.bounds_estimate()
    right_bounds = right.bounds_estimate()
    relation = stage.body_relation
    metadata = _surface_boolean_result_metadata(operands)

    if operands.operation == "intersection":
        overlap = _aabb_overlap(left_bounds, right_bounds)
        spans = _bounds_size(overlap)
        if relation in {"disjoint", "touching"} or any(span <= 1e-9 for span in spans):
            return SurfaceBooleanResult(
                operation="intersection",
                operands=operands,
                status="succeeded",
                classification="empty",
            )
        body = _surface_box_body_from_bounds(overlap, metadata=metadata)
        return _surface_boolean_finalize_body_result("intersection", operands, body)

    if operands.operation == "union":
        if relation == "disjoint":
            left_shell = left.iter_shells(world=True)[0]
            right_shell = right.iter_shells(world=True)[0]
            body = make_surface_body((left_shell, right_shell), metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        if relation == "equal":
            body = _surface_box_body_from_bounds(left_bounds, metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        if relation == "containment":
            container_bounds = left_bounds if _contains_bounds(left_bounds, right_bounds) else right_bounds
            body = _surface_box_body_from_bounds(container_bounds, metadata=metadata)
            return _surface_boolean_finalize_body_result("union", operands, body)
        if relation == "overlap":
            body = _surface_orthogonal_box_boolean_body(
                left_bounds,
                right_bounds,
                operation="union",
                metadata=metadata,
            )
            if body is not None:
                return _surface_boolean_finalize_body_result("union", operands, body)
        return None

    if operands.operation == "difference":
        if relation in {"disjoint", "touching"}:
            body = _surface_box_body_from_bounds(left_bounds, metadata=metadata)
            return _surface_boolean_finalize_body_result("difference", operands, body)
        if relation == "equal" or _contains_bounds(right_bounds, left_bounds):
            return SurfaceBooleanResult(
                operation="difference",
                operands=operands,
                status="succeeded",
                classification="empty",
            )
        if relation == "overlap":
            body = _surface_orthogonal_box_boolean_body(
                left_bounds,
                right_bounds,
                operation="difference",
                metadata=metadata,
            )
            if body is not None:
                return _surface_boolean_finalize_body_result("difference", operands, body)
        return None

    return None


def _canonicalize_surface_boolean_body(body: SurfaceBody, *, role: str) -> SurfaceBody:
    if not isinstance(body, SurfaceBody):
        raise TypeError(f"{role} must be a SurfaceBody.")
    if body.shell_count != 1:
        raise SurfaceBooleanEligibilityError(f"{role} must contain exactly one shell for surfaced booleans.")

    shell = body.iter_shells(world=True)[0]
    if not shell.connected:
        raise SurfaceBooleanEligibilityError(f"{role} shell must be connected for surfaced booleans.")

    canonical_shell = make_surface_shell(
        shell.iter_patches(world=True),
        connected=shell.connected,
        seams=shell.seams,
        adjacency=shell.adjacency,
        metadata=shell.metadata,
    )
    canonical_body = make_surface_body([canonical_shell], metadata=body.metadata)
    classification = _classify_surface_body(canonical_body)
    if classification != "closed":
        raise SurfaceBooleanEligibilityError(
            f"{role} must be closed-valid under shell seam and boundary truth for surfaced booleans."
        )
    return canonical_body


def prepare_surface_boolean_operands(
    operation: Literal["union", "intersection"],
    bodies: Iterable[SurfaceBody],
) -> SurfaceBooleanOperands:
    if operation not in {"union", "intersection"}:
        raise ValueError("operation must be 'union' or 'intersection'.")
    canonical = tuple(
        _canonicalize_surface_boolean_body(body, role=f"{operation} operand {index}")
        for index, body in enumerate(bodies)
    )
    if len(canonical) < 2:
        raise ValueError(f"surface boolean {operation} requires at least two SurfaceBody operands.")
    return SurfaceBooleanOperands(operation=operation, bodies=canonical)


def prepare_surface_boolean_difference_operands(
    base: SurfaceBody,
    cutters: Iterable[SurfaceBody],
) -> SurfaceBooleanOperands:
    canonical_base = _canonicalize_surface_boolean_body(base, role="difference base")
    canonical_cutters = tuple(
        _canonicalize_surface_boolean_body(body, role=f"difference cutter {index}")
        for index, body in enumerate(cutters)
    )
    if not canonical_cutters:
        raise ValueError("surface boolean difference requires at least one cutter SurfaceBody.")
    return SurfaceBooleanOperands(operation="difference", bodies=(canonical_base, *canonical_cutters))


def surface_boolean_result(operation: SurfaceBooleanOperation, operands: SurfaceBooleanOperands) -> SurfaceBooleanResult:
    """Return the structured surfaced boolean result for the current v1 implementation."""

    if operands.operation != operation:
        raise ValueError("Surface boolean result operation must match prepared operands.")
    plan = plan_prepared_surface_csg_operation(operands)
    if not plan.executable:
        return SurfaceBooleanResult(
            operation=operation,
            operands=operands,
            status="unsupported",
            failure_reason="; ".join(diagnostic.message for diagnostic in plan.diagnostics),
        )
    trivial_result = _surface_boolean_trivial_result(operands)
    if trivial_result is not None:
        return trivial_result
    stage = surface_boolean_intersection_stage(operands)
    supported_result = _surface_boolean_supported_box_result(operands, stage)
    if supported_result is not None:
        return supported_result
    return SurfaceBooleanResult(
        operation=operation,
        operands=operands,
        status="unsupported",
        failure_reason=(
            f"Surface boolean {operation} execution is not implemented yet after canonical input preparation."
        ),
    )


def _raise_surface_boolean_execution_unavailable(operands: SurfaceBooleanOperands) -> None:
    result = surface_boolean_result(operands.operation, operands)
    raise SurfaceBooleanExecutionUnavailableError(result.operation, result.operands.body_ids)


def _surface_boolean_result_after_family_gate(
    operation: SurfaceBooleanOperation,
    bodies: tuple[object, ...],
    *,
    caller_id: str,
) -> SurfaceBooleanResult | None:
    if any(not isinstance(body, SurfaceBody) for body in bodies):
        return None
    raw_operands = SurfaceBooleanOperands(operation=operation, bodies=bodies)
    gate = surface_csg_feature_gate(caller_id, operation, bodies)
    if gate.supported:
        return None
    return SurfaceBooleanResult(
        operation=operation,
        operands=raw_operands,
        status="unsupported",
        failure_reason=gate.reason,
    )


def boolean_union(
    meshes: Iterable[Mesh | MeshGroup | SurfaceBody],
    tolerance: float = 1e-4,
    backend: BooleanBackend = "manifold",
) -> Mesh | SurfaceBooleanResult:
    _ensure_backend(backend)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if backend == "surface":
        bodies = tuple(meshes)
        gated = _surface_boolean_result_after_family_gate("union", bodies, caller_id="csg.boolean_union")  # type: ignore[arg-type]
        if gated is not None:
            return gated
        operands = prepare_surface_boolean_operands("union", bodies)  # type: ignore[arg-type]
        return assert_no_hidden_surface_csg_mesh_fallback(
            "csg.boolean_union",
            surface_boolean_result("union", operands),
        )
    warn_mesh_primary_api(
        "boolean_union",
        replacement="SurfaceBody boolean operations once the surface-first CSG path lands",
    )
    return _apply_boolean(_flatten_meshes(meshes), "union")


def boolean_difference(
    base: Mesh | MeshGroup | SurfaceBody,
    cutters: Iterable[Mesh | MeshGroup | SurfaceBody],
    tolerance: float = 1e-4,
    backend: BooleanBackend = "manifold",
) -> Mesh | SurfaceBooleanResult:
    _ensure_backend(backend)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if backend == "surface":
        cutter_tuple = tuple(cutters)
        gated = _surface_boolean_result_after_family_gate(
            "difference",
            (base, *cutter_tuple),
            caller_id="csg.boolean_difference",
        )  # type: ignore[arg-type]
        if gated is not None:
            return gated
        operands = prepare_surface_boolean_difference_operands(base, cutter_tuple)  # type: ignore[arg-type]
        return assert_no_hidden_surface_csg_mesh_fallback(
            "csg.boolean_difference",
            surface_boolean_result("difference", operands),
        )
    warn_mesh_primary_api(
        "boolean_difference",
        replacement="SurfaceBody boolean operations once the surface-first CSG path lands",
    )
    if isinstance(base, MeshGroup):
        base_mesh = base.to_mesh()
    else:
        base_mesh = base
    meshes = _flatten_meshes([base_mesh]) + _flatten_meshes(cutters)
    return _apply_boolean(meshes, "difference")


def boolean_intersection(
    meshes: Iterable[Mesh | MeshGroup | SurfaceBody],
    tolerance: float = 1e-4,
    backend: BooleanBackend = "manifold",
) -> Mesh | SurfaceBooleanResult:
    _ensure_backend(backend)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if backend == "surface":
        bodies = tuple(meshes)
        gated = _surface_boolean_result_after_family_gate(
            "intersection",
            bodies,
            caller_id="csg.boolean_intersection",
        )  # type: ignore[arg-type]
        if gated is not None:
            return gated
        operands = prepare_surface_boolean_operands("intersection", bodies)  # type: ignore[arg-type]
        return assert_no_hidden_surface_csg_mesh_fallback(
            "csg.boolean_intersection",
            surface_boolean_result("intersection", operands),
        )
    warn_mesh_primary_api(
        "boolean_intersection",
        replacement="SurfaceBody boolean operations once the surface-first CSG path lands",
    )
    return _apply_boolean(_flatten_meshes(meshes), "intersection")


def union_meshes(
    meshes: Union[Iterable[Mesh | MeshGroup], Mapping[object, Mesh | MeshGroup]],
    tolerance: float = 1e-4,
    backend: BooleanBackend = "manifold",
) -> Mesh:
    """Retained standalone mesh union tool.

    This helper remains useful for mesh analysis, repair, and debugging
    workflows, but it is not canonical surfaced modeling truth.
    """

    warn_mesh_primary_api(
        "union_meshes",
        replacement="SurfaceBody-native composition plus tessellation at the boundary",
    )
    if isinstance(meshes, Mapping):
        meshes = meshes.values()
    return boolean_union(meshes, tolerance=tolerance, backend=backend)
