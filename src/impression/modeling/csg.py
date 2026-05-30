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
    BSplineSurfacePatch,
    ImplicitFieldExpressionGraph,
    ImplicitFieldNode,
    ImplicitFieldSafetyValidationReport,
    ImplicitOperandFieldAdapterRecord,
    ImplicitOperandFieldAdapterRefusalDiagnostic,
    ImplicitSurfacePatch,
    NURBSSurfacePatch,
    NURBSWeightValidationDiagnostic,
    PATCH_FAMILY_CAPABILITY_MATRIX,
    SurfaceBoundaryRef,
    DisplacementIdentityDiagnostic,
    DisplacementSourceProvenanceRecord,
    DisplacementSourceResolutionResult,
    DisplacementSurfacePatch,
    HeightmapSurfacePatch,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SubdivisionSurfacePatch,
    SurfacePatch,
    SweepSurfacePatch,
    SurfaceBody,
    SurfaceSeam,
    TrimLoop,
    make_surface_body,
    make_surface_shell,
    validate_nurbs_weights,
    adapt_surface_patch_to_implicit_field,
    build_implicit_field_safety_validation_report,
    implicit_difference_field,
    implicit_intersection_field,
    implicit_union_field,
    resolve_displacement_source_identity,
)
from .heightmap import (
    HeightmapGridAlignmentRecord,
    plan_heightmap_grid_alignment,
)
from .surface_intersections import (
    SurfaceAnalyticSplineResidualReport,
    SurfaceIntersectionOverlapRegionRecord,
    SurfaceIntersectionResultRecord,
    SurfaceIntersectionSupportDiagnostic,
    SurfaceSplineSplineResidualReport,
    SurfaceSubdivisionIntersectionAdapterReport,
    SurfaceSweepPairResidualReport,
    make_surface_intersection_request,
    normalize_surface_intersection_result,
    solve_analytic_spline_surface_intersection,
    solve_spline_spline_surface_intersection,
    solve_subdivision_surface_intersection_adapter,
    solve_sweep_surface_intersection_adapter,
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
SurfaceSampledImplicitCSGRouteStatus = Literal[
    "in-progress",
    "native-route",
    "promotion-route",
    "representation-refusal",
    "non-csg-replacement",
]
SurfaceSampledImplicitPromotionTargetFamily = Literal[
    "implicit",
    "subdivision",
    "nurbs",
    "bspline",
    "representation-refusal",
    "non-csg-replacement",
]
SurfaceCSGRoutePairClass = Literal[
    "low-order-analytic",
    "analytic-to-bspline",
    "analytic-to-nurbs",
    "analytic-to-sweep",
    "analytic-to-subdivision",
    "spline-nurbs-pair",
    "sweep-pair",
    "subdivision-pair",
    "sampled-boundary",
    "unsupported-family",
]
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
class ImplicitCompositionOperandSignPolicy:
    """Operand sign treatment for hard implicit Boolean composition."""

    operation: SurfaceBooleanOperation
    operand_index: int
    role: Literal["base", "cutter", "member"]
    sign: Literal["preserve", "negate"]

    def __post_init__(self) -> None:
        operation = str(self.operation)
        role = str(self.role)
        sign = str(self.sign)
        if operation not in SURFACE_BOOLEAN_OPERATIONS:
            raise ValueError("Implicit composition sign policy operation is unsupported.")
        if role not in {"base", "cutter", "member"}:
            raise ValueError("Implicit composition sign policy role is unsupported.")
        if sign not in {"preserve", "negate"}:
            raise ValueError("Implicit composition sign policy sign is unsupported.")
        operand_index = int(self.operand_index)
        if operand_index < 0:
            raise ValueError("Implicit composition operand_index must be non-negative.")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "operand_index", operand_index)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "sign", sign)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "operand_index": self.operand_index,
            "role": self.role,
            "sign": self.sign,
        }


@dataclass(frozen=True)
class ImplicitCompositionDiagnostic:
    """Deterministic refusal or warning for implicit CSG composition."""

    code: Literal["invalid-operation", "insufficient-operands", "unsupported-adapter", "unsafe-result"]
    message: str
    operand_index: int | None = None
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "operand_index": self.operand_index,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class ImplicitCompositionOperationRecord:
    """Hard Boolean implicit composition operation record."""

    operation: SurfaceBooleanOperation
    operand_ids: tuple[str, ...]
    sign_policies: tuple[ImplicitCompositionOperandSignPolicy, ...]
    result_graph: ImplicitFieldExpressionGraph | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "operand_ids": self.operand_ids,
            "sign_policies": [policy.canonical_payload() for policy in self.sign_policies],
            "result_graph": None if self.result_graph is None else self.result_graph.canonical_payload(),
        }


@dataclass(frozen=True)
class ImplicitCompositionResult:
    """Surface-native implicit CSG composition result."""

    operation: SurfaceBooleanOperation
    supported: bool
    operation_record: ImplicitCompositionOperationRecord
    body: SurfaceBody | None = None
    patch: ImplicitSurfacePatch | None = None
    safety: ImplicitFieldSafetyValidationReport | None = None
    diagnostics: tuple[ImplicitCompositionDiagnostic, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "supported": self.supported,
            "operation_record": self.operation_record.canonical_payload(),
            "body_id": None if self.body is None else self.body.stable_identity,
            "patch_id": None if self.patch is None else self.patch.stable_identity,
            "safety": None if self.safety is None else self.safety.canonical_payload(),
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class HeightmapCompositionDiagnostic:
    """Deterministic diagnostic for heightmap-preserving CSG composition."""

    code: Literal[
        "invalid-operation",
        "unsupported-operand",
        "alignment-refusal",
        "representability-refusal",
        "unrepresentable-result",
    ]
    message: str
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class HeightmapCompositionRecord:
    """Surface-native heightmap CSG composition operation record."""

    operation: SurfaceBooleanOperation
    operand_ids: tuple[str, str]
    alignment: HeightmapGridAlignmentRecord
    sample_shape: tuple[int, int] | None = None
    resample_kernel: Literal["none", "bilinear"] = "none"
    result_family: Literal["heightmap"] = "heightmap"
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "operand_ids": self.operand_ids,
            "alignment": self.alignment.canonical_payload(),
            "sample_shape": self.sample_shape,
            "resample_kernel": self.resample_kernel,
            "result_family": self.result_family,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class HeightmapOverhangDiagnostic:
    """Heightmap representability diagnostic for non-2.5D projected states."""

    code: Literal[
        "invalid-projection",
        "overhang",
        "multi-valued-projection",
        "unsafe-grid",
    ]
    message: str
    patch_id: str | None = None
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "patch_id": self.patch_id,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class HeightmapRepresentabilityReport:
    """Pre-execution 2.5D representability report for heightmap CSG."""

    operation: SurfaceBooleanOperation
    operand_ids: tuple[str, str]
    alignment: HeightmapGridAlignmentRecord
    diagnostics: tuple[HeightmapOverhangDiagnostic, ...] = ()

    @property
    def representable(self) -> bool:
        return not self.diagnostics and self.alignment.supported

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "operand_ids": self.operand_ids,
            "alignment": self.alignment.canonical_payload(),
            "representable": self.representable,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


HeightmapPromotionTargetFamily = Literal["implicit", "subdivision"]


@dataclass(frozen=True)
class HeightmapPromotionDiagnostic:
    """Diagnostic emitted while selecting a heightmap CSG promotion target."""

    code: Literal["non-applicable", "missing-route", "unsafe-source"]
    message: str
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class HeightmapPromotionTriggerRecord:
    """Promotion trigger facts derived from a heightmap representability report."""

    operation: SurfaceBooleanOperation
    operand_ids: tuple[str, str]
    trigger_codes: tuple[str, ...]
    reason: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "operand_ids": self.operand_ids,
            "trigger_codes": self.trigger_codes,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HeightmapPromotionDecision:
    """Declared promotion route for a heightmap CSG result that cannot remain a heightmap."""

    operation: SurfaceBooleanOperation
    source_families: tuple[str, str]
    target_family: HeightmapPromotionTargetFamily | None
    supported: bool
    trigger: HeightmapPromotionTriggerRecord
    report: HeightmapRepresentabilityReport
    lossiness: Literal["lossless", "sampled-reconstruction", "volumetric-field"]
    diagnostics: tuple[HeightmapPromotionDiagnostic, ...] = ()
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "source_families": self.source_families,
            "target_family": self.target_family,
            "supported": self.supported,
            "trigger": self.trigger.canonical_payload(),
            "report": self.report.canonical_payload(),
            "lossiness": self.lossiness,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class DisplacementSourceIdentityRecord:
    """CSG-facing source identity for one displacement operand."""

    operand_index: int
    patch_id: str
    source_family: str | None
    source_patch_id: str | None
    source_transform_digest: str | None
    provenance: DisplacementSourceProvenanceRecord | None
    diagnostic: DisplacementIdentityDiagnostic

    @property
    def resolved(self) -> bool:
        return self.source_patch_id is not None and self.diagnostic.code.endswith("resolved")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operand_index": self.operand_index,
            "patch_id": self.patch_id,
            "source_family": self.source_family,
            "source_patch_id": self.source_patch_id,
            "source_transform_digest": self.source_transform_digest,
            "provenance": None if self.provenance is None else self.provenance.canonical_payload(),
            "diagnostic": self.diagnostic.canonical_payload(),
            "resolved": self.resolved,
        }


@dataclass(frozen=True)
class DisplacementSourceMismatchDiagnostic:
    """Blocking CSG diagnostic for incompatible displacement source identities."""

    code: Literal["missing-source", "source-mismatch", "transformed-source-mismatch", "invalid-operand"]
    message: str
    operand_index: int | None = None
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "operand_index": self.operand_index,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class DisplacementSourceCompatibilityReport:
    """Compatibility report for displacement-preserving CSG source identity."""

    operation: SurfaceBooleanOperation
    source_records: tuple[DisplacementSourceIdentityRecord, ...]
    diagnostics: tuple[DisplacementSourceMismatchDiagnostic, ...] = ()

    @property
    def compatible(self) -> bool:
        return bool(self.source_records) and not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "compatible": self.compatible,
            "source_records": [record.canonical_payload() for record in self.source_records],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class DisplacementDomainOverlapRecord:
    """Projection-domain overlap facts for displacement-preserving CSG."""

    overlap_bounds: tuple[float, float, float, float]
    left_index_window: tuple[int, int, int, int]
    right_index_window: tuple[int, int, int, int]

    @property
    def has_overlap(self) -> bool:
        umin, umax, vmin, vmax = self.overlap_bounds
        return umax > umin and vmax > vmin

    def canonical_payload(self) -> dict[str, object]:
        return {
            "overlap_bounds": self.overlap_bounds,
            "left_index_window": self.left_index_window,
            "right_index_window": self.right_index_window,
            "has_overlap": self.has_overlap,
        }


@dataclass(frozen=True)
class DisplacementTangentFrameDiagnostic:
    """Diagnostic for incompatible displacement sampling frames."""

    code: Literal["source-mismatch", "frame-mismatch", "disjoint-domain", "resampling-budget-exceeded"]
    message: str
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class DisplacementResamplingRecord:
    """Aligned displacement sample plan over a shared source-domain overlap."""

    supported: bool
    alignment: Literal["aligned", "resample-required", "refused"]
    overlap: DisplacementDomainOverlapRecord | None = None
    result_shape: tuple[int, int] | None = None
    resample_kernel: Literal["none", "bilinear"] = "none"
    lossiness: Literal["lossless", "sampled-reconstruction"] = "lossless"
    diagnostics: tuple[DisplacementTangentFrameDiagnostic, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "alignment": self.alignment,
            "overlap": None if self.overlap is None else self.overlap.canonical_payload(),
            "result_shape": self.result_shape,
            "resample_kernel": self.resample_kernel,
            "lossiness": self.lossiness,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class DisplacementCompositionDiagnostic:
    """Deterministic diagnostic for displacement-preserving CSG composition."""

    code: Literal["invalid-operation", "domain-refusal", "empty-result"]
    message: str
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class DisplacementCompositionRecord:
    """Surface-native displacement offset CSG composition operation record."""

    operation: SurfaceBooleanOperation
    operand_ids: tuple[str, str]
    source_patch_id: str
    resampling: DisplacementResamplingRecord
    sample_shape: tuple[int, int] | None = None
    result_family: Literal["displacement"] = "displacement"
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "operand_ids": self.operand_ids,
            "source_patch_id": self.source_patch_id,
            "resampling": self.resampling.canonical_payload(),
            "sample_shape": self.sample_shape,
            "result_family": self.result_family,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class DisplacementCompositionResult:
    """Displacement-preserving CSG result or explicit refusal."""

    operation: SurfaceBooleanOperation
    supported: bool
    operation_record: DisplacementCompositionRecord
    body: SurfaceBody | None = None
    patch: DisplacementSurfacePatch | None = None
    diagnostics: tuple[DisplacementCompositionDiagnostic, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "supported": self.supported,
            "operation_record": self.operation_record.canonical_payload(),
            "body_id": None if self.body is None else self.body.stable_identity,
            "patch_id": None if self.patch is None else self.patch.stable_identity,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class DisplacementSourceMismatchRefusalRecord:
    """Supported refusal route for displacement CSG source/domain incompatibility."""

    operation: SurfaceBooleanOperation
    refused: bool
    reason_code: Literal[
        "none",
        "source-mismatch",
        "transformed-source-mismatch",
        "missing-source",
        "incompatible-frame",
        "incompatible-domain",
        "resampling-budget-exceeded",
    ]
    message: str
    replacement_hint: Literal["execute-displacement", "promote-to-implicit", "promote-to-subdivision"]
    source_report: DisplacementSourceCompatibilityReport
    resampling: DisplacementResamplingRecord | None = None
    no_mesh_fallback: bool = True

    @property
    def supported_refusal(self) -> bool:
        return self.refused and self.reason_code != "none" and self.no_mesh_fallback

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "refused": self.refused,
            "supported_refusal": self.supported_refusal,
            "reason_code": self.reason_code,
            "message": self.message,
            "replacement_hint": self.replacement_hint,
            "source_report": self.source_report.canonical_payload(),
            "resampling": None if self.resampling is None else self.resampling.canonical_payload(),
            "no_mesh_fallback": self.no_mesh_fallback,
        }


DisplacementPromotionTargetFamily = Literal["implicit", "subdivision"]


@dataclass(frozen=True)
class DisplacementPromotionDiagnostic:
    """Diagnostic emitted while selecting a displacement CSG promotion target."""

    code: Literal["non-applicable", "missing-route", "unsafe-source-detach"]
    message: str
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class DisplacementSourceDetachTriggerRecord:
    """Promotion trigger facts from a displacement source/domain refusal."""

    operation: SurfaceBooleanOperation
    reason_code: str
    replacement_hint: str
    reason: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "reason_code": self.reason_code,
            "replacement_hint": self.replacement_hint,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DisplacementPromotionDecision:
    """Declared promotion route for source-detaching displacement CSG results."""

    operation: SurfaceBooleanOperation
    target_family: DisplacementPromotionTargetFamily | None
    supported: bool
    trigger: DisplacementSourceDetachTriggerRecord
    refusal: DisplacementSourceMismatchRefusalRecord
    lossiness: Literal["lossless", "sampled-reconstruction", "volumetric-field"]
    diagnostics: tuple[DisplacementPromotionDiagnostic, ...] = ()
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "target_family": self.target_family,
            "supported": self.supported,
            "trigger": self.trigger.canonical_payload(),
            "refusal": self.refusal.canonical_payload(),
            "lossiness": self.lossiness,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class HeightmapCompositionResult:
    """Heightmap-preserving CSG result or explicit refusal."""

    operation: SurfaceBooleanOperation
    supported: bool
    operation_record: HeightmapCompositionRecord
    body: SurfaceBody | None = None
    patch: HeightmapSurfacePatch | None = None
    diagnostics: tuple[HeightmapCompositionDiagnostic, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "supported": self.supported,
            "operation_record": self.operation_record.canonical_payload(),
            "body_id": None if self.body is None else self.body.stable_identity,
            "patch_id": None if self.patch is None else self.patch.stable_identity,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


def implicit_composition_operand_sign_policies(
    operation: SurfaceBooleanOperation,
    operand_count: int,
) -> tuple[ImplicitCompositionOperandSignPolicy, ...]:
    """Return deterministic sign policy for hard implicit Boolean composition."""

    if operation not in SURFACE_BOOLEAN_OPERATIONS:
        raise ValueError("Implicit composition operation must be union, difference, or intersection.")
    count = int(operand_count)
    if count < 2:
        raise ValueError("Implicit composition requires at least two operands.")
    policies: list[ImplicitCompositionOperandSignPolicy] = []
    for index in range(count):
        if operation == "difference":
            role: Literal["base", "cutter", "member"] = "base" if index == 0 else "cutter"
            sign: Literal["preserve", "negate"] = "preserve" if index == 0 else "negate"
        else:
            role = "member"
            sign = "preserve"
        policies.append(
            ImplicitCompositionOperandSignPolicy(
                operation=operation,
                operand_index=index,
                role=role,
                sign=sign,
            )
        )
    return tuple(policies)


def _heightmap_patch_world_height_and_mask(
    patch: HeightmapSurfacePatch,
    x: float,
    y: float,
    *,
    tolerance: float,
) -> tuple[float, bool]:
    rows, cols = patch.height_samples.shape
    sx, sy = patch.xy_scale
    u = (float(x) - float(patch.center[0])) / sx + (cols - 1) / 2.0
    v = (rows - 1) - ((float(y) - float(patch.center[1])) / sy + (rows - 1) / 2.0)
    if u < -tolerance or v < -tolerance or u > (cols - 1) + tolerance or v > (rows - 1) + tolerance:
        return 0.0, False
    height = float(patch.center[2] + patch._height_at(u, v))
    if patch.alpha_mode == "ignore":
        return height, True
    sample_row = int(np.clip(round(v), 0, rows - 1))
    sample_col = int(np.clip(round(u), 0, cols - 1))
    return height, bool(patch.alpha_mask[sample_row, sample_col])


def _heightmap_transform_representability_diagnostics(
    patch: HeightmapSurfacePatch,
    *,
    tolerance: float,
) -> tuple[HeightmapOverhangDiagnostic, ...]:
    matrix = np.asarray(patch.transform_matrix, dtype=float)
    diagnostics: list[HeightmapOverhangDiagnostic] = []
    if abs(float(matrix[3, 3]) - 1.0) > tolerance or np.any(np.abs(matrix[3, :3]) > tolerance):
        diagnostics.append(
            HeightmapOverhangDiagnostic(
                code="invalid-projection",
                patch_id=patch.stable_identity,
                message="Heightmap transform uses projective coordinates and cannot preserve a finite 2.5D projection.",
            )
        )
    if abs(float(matrix[0, 2])) > tolerance or abs(float(matrix[1, 2])) > tolerance:
        diagnostics.append(
            HeightmapOverhangDiagnostic(
                code="overhang",
                patch_id=patch.stable_identity,
                message="Heightmap transform maps height into projected x/y coordinates, creating possible overhangs.",
            )
        )
    projected_jacobian = matrix[:2, :2]
    if abs(float(np.linalg.det(projected_jacobian))) <= tolerance:
        diagnostics.append(
            HeightmapOverhangDiagnostic(
                code="multi-valued-projection",
                patch_id=patch.stable_identity,
                message="Heightmap transform collapses the projected x/y axes, creating a multi-valued height projection.",
            )
        )
    if patch.alpha_mode == "mask" and not bool(np.any(patch.alpha_mask)):
        diagnostics.append(
            HeightmapOverhangDiagnostic(
                code="unsafe-grid",
                patch_id=patch.stable_identity,
                message="Heightmap operand mask contains no visible samples for heightmap CSG execution.",
            )
        )
    return tuple(diagnostics)


def heightmap_representability_report(
    operation: SurfaceBooleanOperation,
    left: HeightmapSurfacePatch,
    right: HeightmapSurfacePatch,
    *,
    alignment: HeightmapGridAlignmentRecord | None = None,
    tolerance: float = 1e-9,
) -> HeightmapRepresentabilityReport:
    """Return the pre-execution refusal report for heightmap-preserving CSG."""

    if operation not in SURFACE_BOOLEAN_OPERATIONS:
        raise ValueError("Heightmap representability operation must be union, difference, or intersection.")
    if not isinstance(left, HeightmapSurfacePatch) or not isinstance(right, HeightmapSurfacePatch):
        raise TypeError("Heightmap representability requires HeightmapSurfacePatch operands.")
    tol = float(tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("Heightmap representability tolerance must be finite and non-negative.")
    checked_alignment = alignment if alignment is not None else plan_heightmap_grid_alignment(left, right, tolerance=tol)
    diagnostics: list[HeightmapOverhangDiagnostic] = []
    if not checked_alignment.supported:
        message = "Heightmap CSG projection/grid alignment is not representable as a heightmap."
        if checked_alignment.diagnostics:
            message = checked_alignment.diagnostics[0].message
        diagnostics.append(
            HeightmapOverhangDiagnostic(
                code="invalid-projection",
                message=f"{message} No mesh fallback was attempted.",
            )
        )
    diagnostics.extend(_heightmap_transform_representability_diagnostics(left, tolerance=tol))
    diagnostics.extend(_heightmap_transform_representability_diagnostics(right, tolerance=tol))
    return HeightmapRepresentabilityReport(
        operation=operation,
        operand_ids=(left.stable_identity, right.stable_identity),
        alignment=checked_alignment,
        diagnostics=tuple(diagnostics),
    )


def select_heightmap_promotion_target(
    report: HeightmapRepresentabilityReport,
    *,
    allowed_targets: Sequence[HeightmapPromotionTargetFamily] = ("implicit", "subdivision"),
) -> HeightmapPromotionDecision:
    """Select the declared non-heightmap route for an unrepresentable heightmap CSG result."""

    allowed = tuple(str(target) for target in allowed_targets)
    invalid_targets = tuple(target for target in allowed if target not in {"implicit", "subdivision"})
    if invalid_targets:
        raise ValueError("Heightmap promotion targets must be implicit or subdivision.")
    trigger_codes = tuple(dict.fromkeys(diagnostic.code for diagnostic in report.diagnostics))
    trigger = HeightmapPromotionTriggerRecord(
        operation=report.operation,
        operand_ids=report.operand_ids,
        trigger_codes=trigger_codes,
        reason="; ".join(diagnostic.message for diagnostic in report.diagnostics) or "heightmap result is representable",
    )
    if report.representable:
        return HeightmapPromotionDecision(
            operation=report.operation,
            source_families=("heightmap", "heightmap"),
            target_family=None,
            supported=False,
            trigger=trigger,
            report=report,
            lossiness="lossless",
            diagnostics=(
                HeightmapPromotionDiagnostic(
                    code="non-applicable",
                    message="Heightmap CSG result remains heightmap-representable; promotion is not applicable.",
                ),
            ),
        )
    if "unsafe-grid" in trigger_codes or "invalid-projection" in trigger_codes:
        return HeightmapPromotionDecision(
            operation=report.operation,
            source_families=("heightmap", "heightmap"),
            target_family=None,
            supported=False,
            trigger=trigger,
            report=report,
            lossiness="sampled-reconstruction",
            diagnostics=(
                HeightmapPromotionDiagnostic(
                    code="unsafe-source",
                    message=(
                        "Heightmap CSG promotion refused because the source route is unsafe or has no valid "
                        "projected domain; no mesh fallback was attempted."
                    ),
                ),
            ),
        )
    target: HeightmapPromotionTargetFamily
    lossiness: Literal["sampled-reconstruction", "volumetric-field"]
    if "overhang" in trigger_codes:
        target = "implicit"
        lossiness = "volumetric-field"
    else:
        target = "subdivision"
        lossiness = "sampled-reconstruction"
    if target not in allowed:
        return HeightmapPromotionDecision(
            operation=report.operation,
            source_families=("heightmap", "heightmap"),
            target_family=target,
            supported=False,
            trigger=trigger,
            report=report,
            lossiness=lossiness,
            diagnostics=(
                HeightmapPromotionDiagnostic(
                    code="missing-route",
                    message=f"Heightmap CSG promotion target {target} is not declared for this route; no mesh fallback was attempted.",
                ),
            ),
        )
    return HeightmapPromotionDecision(
        operation=report.operation,
        source_families=("heightmap", "heightmap"),
        target_family=target,
        supported=True,
        trigger=trigger,
        report=report,
        lossiness=lossiness,
    )


def plan_heightmap_promotion_route(
    operation: SurfaceBooleanOperation,
    left: HeightmapSurfacePatch,
    right: HeightmapSurfacePatch,
    *,
    allowed_targets: Sequence[HeightmapPromotionTargetFamily] = ("implicit", "subdivision"),
    tolerance: float = 1e-9,
) -> HeightmapPromotionDecision:
    """Build the heightmap refusal-to-promotion route decision for CSG."""

    report = heightmap_representability_report(operation, left, right, tolerance=tolerance)
    return select_heightmap_promotion_target(report, allowed_targets=allowed_targets)


def _displacement_csg_source_transform_digest(patch: DisplacementSurfacePatch) -> str:
    combined = np.asarray(patch.source_patch.transform_matrix, dtype=float) @ np.asarray(patch.transform_matrix, dtype=float)
    payload = combined.round(12).tolist()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def resolve_displacement_csg_source_identity(
    patch: DisplacementSurfacePatch,
    *,
    operand_index: int = 0,
    candidate_patches: Sequence[SurfacePatch] = (),
    allow_cross_body_reference: bool = False,
) -> DisplacementSourceIdentityRecord:
    """Resolve one displacement operand source identity for CSG planning."""

    if not isinstance(patch, DisplacementSurfacePatch):
        raise TypeError("Displacement CSG source identity requires a DisplacementSurfacePatch.")
    resolution: DisplacementSourceResolutionResult = resolve_displacement_source_identity(
        source_patch=patch.source_patch,
        candidate_patches=candidate_patches,
        allow_cross_body_reference=allow_cross_body_reference,
    )
    return DisplacementSourceIdentityRecord(
        operand_index=int(operand_index),
        patch_id=patch.stable_identity,
        source_family=None if resolution.provenance is None else resolution.provenance.source_family,
        source_patch_id=None if resolution.provenance is None else resolution.provenance.source_patch_id,
        source_transform_digest=_displacement_csg_source_transform_digest(patch),
        provenance=resolution.provenance,
        diagnostic=resolution.diagnostic,
    )


def displacement_source_compatibility_report(
    operation: SurfaceBooleanOperation,
    patches: Sequence[DisplacementSurfacePatch],
    *,
    candidate_patches: Sequence[SurfacePatch] = (),
    allow_cross_body_reference: bool = False,
) -> DisplacementSourceCompatibilityReport:
    """Validate source-surface identity compatibility for displacement-preserving CSG."""

    if operation not in SURFACE_BOOLEAN_OPERATIONS:
        raise ValueError("Displacement CSG source compatibility operation must be union, difference, or intersection.")
    records: list[DisplacementSourceIdentityRecord] = []
    diagnostics: list[DisplacementSourceMismatchDiagnostic] = []
    for index, patch in enumerate(patches):
        if not isinstance(patch, DisplacementSurfacePatch):
            diagnostics.append(
                DisplacementSourceMismatchDiagnostic(
                    code="invalid-operand",
                    operand_index=index,
                    message="Displacement-preserving CSG requires DisplacementSurfacePatch operands; no mesh fallback was attempted.",
                )
            )
            continue
        record = resolve_displacement_csg_source_identity(
            patch,
            operand_index=index,
            candidate_patches=candidate_patches,
            allow_cross_body_reference=allow_cross_body_reference,
        )
        records.append(record)
        if not record.resolved:
            diagnostics.append(
                DisplacementSourceMismatchDiagnostic(
                    code="missing-source",
                    operand_index=index,
                    message=f"Displacement operand {index} source identity is unresolved: {record.diagnostic.message}; no mesh fallback was attempted.",
                )
            )
    if records and not diagnostics:
        first = records[0]
        for record in records[1:]:
            if record.source_patch_id != first.source_patch_id:
                diagnostics.append(
                    DisplacementSourceMismatchDiagnostic(
                        code="source-mismatch",
                        operand_index=record.operand_index,
                        message=(
                            "Displacement-preserving CSG requires all operands to resolve to the same source patch identity; "
                            f"{first.source_patch_id} != {record.source_patch_id}; no mesh fallback was attempted."
                        ),
                    )
                )
            elif record.source_transform_digest != first.source_transform_digest:
                diagnostics.append(
                    DisplacementSourceMismatchDiagnostic(
                        code="transformed-source-mismatch",
                        operand_index=record.operand_index,
                        message=(
                            "Displacement-preserving CSG requires matching source transform digests for transformed sources; "
                            "no mesh fallback was attempted."
                        ),
                    )
                )
    return DisplacementSourceCompatibilityReport(
        operation=operation,
        source_records=tuple(records),
        diagnostics=tuple(diagnostics),
    )


def _displacement_sample_spacing(patch: DisplacementSurfacePatch) -> tuple[float, float]:
    umin, umax, vmin, vmax = patch.projection_bounds
    rows, cols = patch.displacement_samples.shape
    return (
        float((umax - umin) / max(cols - 1, 1)),
        float((vmax - vmin) / max(rows - 1, 1)),
    )


def _displacement_index_window(
    patch: DisplacementSurfacePatch,
    overlap: tuple[float, float, float, float],
    *,
    tolerance: float,
) -> tuple[int, int, int, int]:
    u0, u1, v0, v1 = overlap
    umin, _umax, vmin, _vmax = patch.projection_bounds
    sx, sy = _displacement_sample_spacing(patch)
    rows, cols = patch.displacement_samples.shape
    c0 = max(0, int(np.floor(((u0 - umin) / sx) + tolerance)))
    c1 = min(cols - 1, int(np.ceil(((u1 - umin) / sx) - tolerance)))
    r0 = max(0, int(np.floor(((v0 - vmin) / sy) + tolerance)))
    r1 = min(rows - 1, int(np.ceil(((v1 - vmin) / sy) - tolerance)))
    return (r0, r1, c0, c1)


def _displacement_grid_lines_align(
    left: DisplacementSurfacePatch,
    right: DisplacementSurfacePatch,
    *,
    tolerance: float,
) -> bool:
    if not np.allclose(_displacement_sample_spacing(left), _displacement_sample_spacing(right), atol=tolerance):
        return False
    sx, sy = _displacement_sample_spacing(left)
    dx = abs((left.projection_bounds[0] - right.projection_bounds[0]) / sx)
    dy = abs((left.projection_bounds[2] - right.projection_bounds[2]) / sy)
    return abs(dx - round(dx)) <= tolerance and abs(dy - round(dy)) <= tolerance


def _displacement_frame_matches(left: DisplacementSurfacePatch, right: DisplacementSurfacePatch) -> bool:
    return (
        left.projection == right.projection
        and left.plane == right.plane
        and left.direction == right.direction
    )


def _projected_overlap_bounds(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (max(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), min(left[3], right[3]))


def plan_displacement_domain_resampling(
    operation: SurfaceBooleanOperation,
    left: DisplacementSurfacePatch,
    right: DisplacementSurfacePatch,
    *,
    max_sample_count: int = 1_000_000,
    tolerance: float = 1e-9,
) -> DisplacementResamplingRecord:
    """Plan source-domain overlap, clipping, and resampling for displacement CSG."""

    if operation not in SURFACE_BOOLEAN_OPERATIONS:
        raise ValueError("Displacement domain resampling operation must be union, difference, or intersection.")
    if not isinstance(left, DisplacementSurfacePatch) or not isinstance(right, DisplacementSurfacePatch):
        raise TypeError("Displacement domain resampling requires DisplacementSurfacePatch operands.")
    tol = float(tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("Displacement domain resampling tolerance must be finite and non-negative.")
    budget = int(max_sample_count)
    if budget <= 0:
        raise ValueError("Displacement resampling max_sample_count must be positive.")
    compatibility = displacement_source_compatibility_report(operation, (left, right))
    if not compatibility.compatible:
        return DisplacementResamplingRecord(
            supported=False,
            alignment="refused",
            diagnostics=(
                DisplacementTangentFrameDiagnostic(
                    code="source-mismatch",
                    message=f"{compatibility.diagnostics[0].message} No mesh fallback was attempted.",
                ),
            ),
        )
    if not _displacement_frame_matches(left, right):
        return DisplacementResamplingRecord(
            supported=False,
            alignment="refused",
            diagnostics=(
                DisplacementTangentFrameDiagnostic(
                    code="frame-mismatch",
                    message="Displacement operands use incompatible projection planes or displacement directions; no mesh fallback was attempted.",
                ),
            ),
        )
    overlap_bounds = _projected_overlap_bounds(left.projection_bounds, right.projection_bounds)
    overlap = DisplacementDomainOverlapRecord(
        overlap_bounds=overlap_bounds,
        left_index_window=_displacement_index_window(left, overlap_bounds, tolerance=tol),
        right_index_window=_displacement_index_window(right, overlap_bounds, tolerance=tol),
    )
    if not overlap.has_overlap:
        return DisplacementResamplingRecord(
            supported=False,
            alignment="refused",
            overlap=overlap,
            diagnostics=(
                DisplacementTangentFrameDiagnostic(
                    code="disjoint-domain",
                    message="Displacement projection domains are disjoint; no mesh fallback was attempted.",
                ),
            ),
        )
    aligned = _displacement_grid_lines_align(left, right, tolerance=tol)
    if aligned:
        result_shape = (
            max(1, overlap.left_index_window[1] - overlap.left_index_window[0] + 1),
            max(1, overlap.left_index_window[3] - overlap.left_index_window[2] + 1),
        )
        if result_shape[0] * result_shape[1] > budget:
            return DisplacementResamplingRecord(
                supported=False,
                alignment="refused",
                overlap=overlap,
                result_shape=result_shape,
                diagnostics=(
                    DisplacementTangentFrameDiagnostic(
                        code="resampling-budget-exceeded",
                        message="Displacement aligned output grid exceeds max_sample_count; no mesh fallback was attempted.",
                    ),
                ),
            )
        return DisplacementResamplingRecord(
            supported=True,
            alignment="aligned",
            overlap=overlap,
            result_shape=result_shape,
            resample_kernel="none",
            lossiness="lossless",
        )
    min_sx = min(_displacement_sample_spacing(left)[0], _displacement_sample_spacing(right)[0])
    min_sy = min(_displacement_sample_spacing(left)[1], _displacement_sample_spacing(right)[1])
    result_shape = (
        max(2, int(np.floor((overlap_bounds[3] - overlap_bounds[2]) / min_sy + tol)) + 1),
        max(2, int(np.floor((overlap_bounds[1] - overlap_bounds[0]) / min_sx + tol)) + 1),
    )
    if result_shape[0] * result_shape[1] > budget:
        return DisplacementResamplingRecord(
            supported=False,
            alignment="refused",
            overlap=overlap,
            result_shape=result_shape,
            diagnostics=(
                DisplacementTangentFrameDiagnostic(
                    code="resampling-budget-exceeded",
                    message="Displacement resampling output grid exceeds max_sample_count; no mesh fallback was attempted.",
                ),
            ),
        )
    return DisplacementResamplingRecord(
        supported=True,
        alignment="resample-required",
        overlap=overlap,
        result_shape=result_shape,
        resample_kernel="bilinear",
        lossiness="sampled-reconstruction",
    )


def _sample_displacement_offset(
    patch: DisplacementSurfacePatch,
    u: float,
    v: float,
    *,
    tolerance: float,
) -> tuple[float, bool]:
    umin, umax, vmin, vmax = patch.projection_bounds
    if u < umin - tolerance or u > umax + tolerance or v < vmin - tolerance or v > vmax + tolerance:
        return 0.0, False
    rows, cols = patch.displacement_samples.shape
    x = (float(u) - umin) / (umax - umin) * (cols - 1)
    y = (float(v) - vmin) / (vmax - vmin) * (rows - 1)
    x = float(np.clip(x, 0.0, cols - 1))
    y = float(np.clip(y, 0.0, rows - 1))
    c0 = int(np.floor(x))
    r0 = int(np.floor(y))
    c1 = min(c0 + 1, cols - 1)
    r1 = min(r0 + 1, rows - 1)
    dx = x - c0
    dy = y - r0
    h00 = patch.displacement_samples[r0, c0]
    h10 = patch.displacement_samples[r0, c1]
    h01 = patch.displacement_samples[r1, c0]
    h11 = patch.displacement_samples[r1, c1]
    height = (
        (1.0 - dx) * (1.0 - dy) * h00
        + dx * (1.0 - dy) * h10
        + (1.0 - dx) * dy * h01
        + dx * dy * h11
    )
    mx = int(np.clip(round(x), 0, cols - 1))
    my = int(np.clip(round(y), 0, rows - 1))
    masked = not bool(patch.alpha_mask[my, mx])
    if patch.alpha_mode == "mask" and masked:
        return 0.0, False
    if patch.alpha_mode == "ignore" and masked:
        height = 0.0
    return float(height * patch.height_scale), True


def _compose_displacement_samples(
    operation: SurfaceBooleanOperation,
    left: DisplacementSurfacePatch,
    right: DisplacementSurfacePatch,
    plan: DisplacementResamplingRecord,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    if plan.overlap is None or plan.result_shape is None:
        raise ValueError("Displacement composition requires an overlap and result shape.")
    rows, cols = plan.result_shape
    umin, umax, vmin, vmax = plan.overlap.overlap_bounds
    us = np.linspace(umin, umax, cols, dtype=float)
    vs = np.linspace(vmin, vmax, rows, dtype=float)
    result = np.zeros((rows, cols), dtype=float)
    mask = np.zeros((rows, cols), dtype=bool)
    for row, v in enumerate(vs):
        for col, u in enumerate(us):
            left_offset, left_valid = _sample_displacement_offset(left, float(u), float(v), tolerance=tolerance)
            right_offset, right_valid = _sample_displacement_offset(right, float(u), float(v), tolerance=tolerance)
            if operation == "union":
                if left_valid and right_valid:
                    result[row, col] = max(left_offset, right_offset)
                    mask[row, col] = True
                elif left_valid:
                    result[row, col] = left_offset
                    mask[row, col] = True
                elif right_valid:
                    result[row, col] = right_offset
                    mask[row, col] = True
            elif operation == "intersection":
                if left_valid and right_valid:
                    result[row, col] = min(left_offset, right_offset)
                    mask[row, col] = True
            elif operation == "difference":
                if left_valid:
                    result[row, col] = max(left_offset - right_offset, 0.0) if right_valid else left_offset
                    mask[row, col] = True
            else:
                raise ValueError("Displacement composition operation must be union, difference, or intersection.")
    return result, mask, plan.overlap.overlap_bounds


def compose_displacement_csg_result(
    operation: SurfaceBooleanOperation,
    left: DisplacementSurfacePatch,
    right: DisplacementSurfacePatch,
    *,
    max_sample_count: int = 1_000_000,
    tolerance: float = 1e-9,
) -> DisplacementCompositionResult:
    """Compose two displacement offsets over a compatible source domain without mesh CSG."""

    if operation not in SURFACE_BOOLEAN_OPERATIONS:
        raise ValueError("Displacement composition operation must be union, difference, or intersection.")
    if not isinstance(left, DisplacementSurfacePatch) or not isinstance(right, DisplacementSurfacePatch):
        raise TypeError("Displacement composition requires DisplacementSurfacePatch operands.")
    tol = float(tolerance)
    plan = plan_displacement_domain_resampling(
        operation,
        left,
        right,
        max_sample_count=max_sample_count,
        tolerance=tol,
    )
    source_id = left.source_patch.stable_identity
    record = DisplacementCompositionRecord(
        operation=operation,
        operand_ids=(left.stable_identity, right.stable_identity),
        source_patch_id=source_id,
        resampling=plan,
        sample_shape=plan.result_shape,
    )
    if not plan.supported:
        return DisplacementCompositionResult(
            operation=operation,
            supported=False,
            operation_record=record,
            diagnostics=(
                DisplacementCompositionDiagnostic(
                    code="domain-refusal",
                    message=(
                        "Displacement CSG composition refused because source-domain planning failed: "
                        f"{'; '.join(diagnostic.message for diagnostic in plan.diagnostics)}"
                    ),
                ),
            ),
        )
    samples, mask, bounds = _compose_displacement_samples(operation, left, right, plan, tolerance=tol)
    if not np.any(mask):
        return DisplacementCompositionResult(
            operation=operation,
            supported=False,
            operation_record=record,
            diagnostics=(
                DisplacementCompositionDiagnostic(
                    code="empty-result",
                    message="Displacement CSG composition produced no visible samples; no mesh fallback was attempted.",
                ),
            ),
        )
    metadata = {
        "kernel": {
            "displacement_csg_composition": {
                **record.canonical_payload(),
                "sample_shape": tuple(int(value) for value in samples.shape),
                "projection_bounds": bounds,
                "lossiness": plan.lossiness,
            }
        }
    }
    patch = DisplacementSurfacePatch(
        family="displacement",
        source_patch=left.source_patch,
        displacement_samples=samples,
        alpha_mask=mask,
        alpha_mode="mask",
        height_scale=1.0,
        direction=left.direction,
        projection=left.projection,
        plane=left.plane,
        projection_bounds=bounds,
        metadata=metadata,
    )
    body = make_surface_body((make_surface_shell((patch,), connected=False),), metadata=metadata)
    final_record = DisplacementCompositionRecord(
        operation=operation,
        operand_ids=record.operand_ids,
        source_patch_id=source_id,
        resampling=plan,
        sample_shape=tuple(int(value) for value in samples.shape),
    )
    return DisplacementCompositionResult(
        operation=operation,
        supported=True,
        operation_record=final_record,
        body=body,
        patch=patch,
    )


def classify_displacement_source_mismatch_refusal(
    source_report: DisplacementSourceCompatibilityReport,
    resampling: DisplacementResamplingRecord | None = None,
) -> tuple[
    Literal[
        "none",
        "source-mismatch",
        "transformed-source-mismatch",
        "missing-source",
        "incompatible-frame",
        "incompatible-domain",
        "resampling-budget-exceeded",
    ],
    Literal["execute-displacement", "promote-to-implicit", "promote-to-subdivision"],
    str,
]:
    """Classify displacement CSG incompatibility as a supported refusal reason."""

    if source_report.diagnostics:
        code = source_report.diagnostics[0].code
        if code == "missing-source":
            return "missing-source", "promote-to-implicit", source_report.diagnostics[0].message
        if code == "transformed-source-mismatch":
            return "transformed-source-mismatch", "promote-to-subdivision", source_report.diagnostics[0].message
        return "source-mismatch", "promote-to-implicit", source_report.diagnostics[0].message
    if resampling is not None and resampling.diagnostics:
        code = resampling.diagnostics[0].code
        if code == "frame-mismatch":
            return "incompatible-frame", "promote-to-implicit", resampling.diagnostics[0].message
        if code == "resampling-budget-exceeded":
            return "resampling-budget-exceeded", "promote-to-subdivision", resampling.diagnostics[0].message
        return "incompatible-domain", "promote-to-subdivision", resampling.diagnostics[0].message
    return "none", "execute-displacement", "Displacement CSG source/domain checks are compatible."


def displacement_source_mismatch_refusal_record(
    operation: SurfaceBooleanOperation,
    left: DisplacementSurfacePatch,
    right: DisplacementSurfacePatch,
    *,
    max_sample_count: int = 1_000_000,
    tolerance: float = 1e-9,
) -> DisplacementSourceMismatchRefusalRecord:
    """Return a supported refusal record for displacement source/domain incompatibility."""

    source_report = displacement_source_compatibility_report(operation, (left, right))
    resampling = None
    if source_report.compatible:
        resampling = plan_displacement_domain_resampling(
            operation,
            left,
            right,
            max_sample_count=max_sample_count,
            tolerance=tolerance,
        )
    reason_code, replacement_hint, message = classify_displacement_source_mismatch_refusal(source_report, resampling)
    refused = reason_code != "none"
    return DisplacementSourceMismatchRefusalRecord(
        operation=operation,
        refused=refused,
        reason_code=reason_code,
        message=message if refused else "Displacement CSG can execute as a displacement-preserving operation.",
        replacement_hint=replacement_hint,
        source_report=source_report,
        resampling=resampling,
    )


def select_displacement_promotion_target(
    refusal: DisplacementSourceMismatchRefusalRecord,
    *,
    allowed_targets: Sequence[DisplacementPromotionTargetFamily] = ("implicit", "subdivision"),
) -> DisplacementPromotionDecision:
    """Select the declared non-displacement route for a source-detaching displacement CSG refusal."""

    allowed = tuple(str(target) for target in allowed_targets)
    invalid_targets = tuple(target for target in allowed if target not in {"implicit", "subdivision"})
    if invalid_targets:
        raise ValueError("Displacement promotion targets must be implicit or subdivision.")
    trigger = DisplacementSourceDetachTriggerRecord(
        operation=refusal.operation,
        reason_code=refusal.reason_code,
        replacement_hint=refusal.replacement_hint,
        reason=refusal.message,
    )
    if not refusal.refused:
        return DisplacementPromotionDecision(
            operation=refusal.operation,
            target_family=None,
            supported=False,
            trigger=trigger,
            refusal=refusal,
            lossiness="lossless",
            diagnostics=(
                DisplacementPromotionDiagnostic(
                    code="non-applicable",
                    message="Displacement CSG can execute without detaching from its source; promotion is not applicable.",
                ),
            ),
        )
    if refusal.reason_code in {"missing-source"}:
        return DisplacementPromotionDecision(
            operation=refusal.operation,
            target_family=None,
            supported=False,
            trigger=trigger,
            refusal=refusal,
            lossiness="volumetric-field",
            diagnostics=(
                DisplacementPromotionDiagnostic(
                    code="unsafe-source-detach",
                    message="Displacement promotion refused because the source identity is missing; no mesh fallback was attempted.",
                ),
            ),
        )
    target: DisplacementPromotionTargetFamily = (
        "implicit" if refusal.replacement_hint == "promote-to-implicit" else "subdivision"
    )
    lossiness: Literal["sampled-reconstruction", "volumetric-field"] = (
        "volumetric-field" if target == "implicit" else "sampled-reconstruction"
    )
    if target not in allowed:
        return DisplacementPromotionDecision(
            operation=refusal.operation,
            target_family=target,
            supported=False,
            trigger=trigger,
            refusal=refusal,
            lossiness=lossiness,
            diagnostics=(
                DisplacementPromotionDiagnostic(
                    code="missing-route",
                    message=f"Displacement CSG promotion target {target} is not declared for this route; no mesh fallback was attempted.",
                ),
            ),
        )
    return DisplacementPromotionDecision(
        operation=refusal.operation,
        target_family=target,
        supported=True,
        trigger=trigger,
        refusal=refusal,
        lossiness=lossiness,
    )


def plan_displacement_promotion_route(
    operation: SurfaceBooleanOperation,
    left: DisplacementSurfacePatch,
    right: DisplacementSurfacePatch,
    *,
    allowed_targets: Sequence[DisplacementPromotionTargetFamily] = ("implicit", "subdivision"),
    max_sample_count: int = 1_000_000,
    tolerance: float = 1e-9,
) -> DisplacementPromotionDecision:
    """Build the displacement refusal-to-promotion route decision for CSG."""

    refusal = displacement_source_mismatch_refusal_record(
        operation,
        left,
        right,
        max_sample_count=max_sample_count,
        tolerance=tolerance,
    )
    return select_displacement_promotion_target(refusal, allowed_targets=allowed_targets)


def _compose_heightmap_samples(
    operation: SurfaceBooleanOperation,
    left: HeightmapSurfacePatch,
    right: HeightmapSurfacePatch,
    alignment: HeightmapGridAlignmentRecord,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float, float]]:
    if alignment.clipping is None or alignment.result_shape is None:
        raise ValueError("Heightmap composition requires an overlap clipping plan and result shape.")
    rows, cols = alignment.result_shape
    xmin, xmax, ymin, ymax = alignment.clipping.overlap_bounds
    if rows < 2 or cols < 2:
        raise ValueError("Heightmap composition result must contain at least a 2x2 grid.")
    xs = np.linspace(xmin, xmax, cols, dtype=float)
    ys = np.linspace(ymax, ymin, rows, dtype=float)
    result = np.zeros((rows, cols), dtype=float)
    mask = np.zeros((rows, cols), dtype=bool)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            left_height, left_valid = _heightmap_patch_world_height_and_mask(left, float(x), float(y), tolerance=tolerance)
            right_height, right_valid = _heightmap_patch_world_height_and_mask(right, float(x), float(y), tolerance=tolerance)
            if operation == "union":
                if left_valid and right_valid:
                    result[row, col] = max(left_height, right_height)
                    mask[row, col] = True
                elif left_valid:
                    result[row, col] = left_height
                    mask[row, col] = True
                elif right_valid:
                    result[row, col] = right_height
                    mask[row, col] = True
            elif operation == "intersection":
                if left_valid and right_valid:
                    result[row, col] = min(left_height, right_height)
                    mask[row, col] = True
            elif operation == "difference":
                if left_valid:
                    result[row, col] = max(left_height - right_height, 0.0) if right_valid else left_height
                    mask[row, col] = True
            else:
                raise ValueError("Heightmap composition operation must be union, difference, or intersection.")
    xy_scale = (
        float((xmax - xmin) / max(cols - 1, 1)),
        float((ymax - ymin) / max(rows - 1, 1)),
    )
    center = (float((xmin + xmax) * 0.5), float((ymin + ymax) * 0.5), 0.0)
    return result, mask, xy_scale, center


def compose_heightmap_csg_result(
    operation: SurfaceBooleanOperation,
    left: HeightmapSurfacePatch,
    right: HeightmapSurfacePatch,
    *,
    tolerance: float = 1e-9,
) -> HeightmapCompositionResult:
    """Compose two native heightmaps over an aligned output grid without mesh CSG."""

    if operation not in SURFACE_BOOLEAN_OPERATIONS:
        placeholder_alignment = plan_heightmap_grid_alignment(left, right, tolerance=tolerance)
        record = HeightmapCompositionRecord(
            operation="union",
            operand_ids=(left.stable_identity, right.stable_identity),
            alignment=placeholder_alignment,
            sample_shape=placeholder_alignment.result_shape,
            resample_kernel=placeholder_alignment.resample_kernel,
        )
        return HeightmapCompositionResult(
            operation="union",
            supported=False,
            operation_record=record,
            diagnostics=(
                HeightmapCompositionDiagnostic(
                    code="invalid-operation",
                    message="Heightmap CSG composition operation must be union, difference, or intersection; no mesh fallback was attempted.",
                ),
            ),
        )
    if not isinstance(left, HeightmapSurfacePatch) or not isinstance(right, HeightmapSurfacePatch):
        raise TypeError("Heightmap CSG composition requires HeightmapSurfacePatch operands.")
    tol = float(tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("Heightmap CSG composition tolerance must be finite and non-negative.")
    alignment = plan_heightmap_grid_alignment(left, right, tolerance=tol)
    record = HeightmapCompositionRecord(
        operation=operation,
        operand_ids=(left.stable_identity, right.stable_identity),
        alignment=alignment,
        sample_shape=alignment.result_shape,
        resample_kernel=alignment.resample_kernel,
    )
    if not alignment.supported:
        return HeightmapCompositionResult(
            operation=operation,
            supported=False,
            operation_record=record,
            diagnostics=(
                HeightmapCompositionDiagnostic(
                    code="alignment-refusal",
                    message=(
                        "Heightmap CSG composition refused because grid alignment failed; "
                        "no mesh fallback was attempted."
                    ),
                ),
            ),
        )
    representability = heightmap_representability_report(
        operation,
        left,
        right,
        alignment=alignment,
        tolerance=tol,
    )
    if not representability.representable:
        metadata_record = record.canonical_payload()
        metadata_record["representability"] = representability.canonical_payload()
        refusal_record = HeightmapCompositionRecord(
            operation=record.operation,
            operand_ids=record.operand_ids,
            alignment=record.alignment,
            sample_shape=record.sample_shape,
            resample_kernel=record.resample_kernel,
        )
        return HeightmapCompositionResult(
            operation=operation,
            supported=False,
            operation_record=refusal_record,
            diagnostics=(
                HeightmapCompositionDiagnostic(
                    code="representability-refusal",
                    message=(
                        "Heightmap CSG composition refused before execution because the result is not "
                        f"representable as a single-valued 2.5D heightmap: "
                        f"{'; '.join(diagnostic.message for diagnostic in representability.diagnostics)} "
                        "No mesh fallback was attempted."
                    ),
                ),
            ),
        )
    try:
        samples, mask, xy_scale, center = _compose_heightmap_samples(
            operation,
            left,
            right,
            alignment,
            tolerance=tol,
        )
    except ValueError as exc:
        return HeightmapCompositionResult(
            operation=operation,
            supported=False,
            operation_record=record,
            diagnostics=(
                HeightmapCompositionDiagnostic(
                    code="unrepresentable-result",
                    message=f"Heightmap CSG composition result is unrepresentable: {exc}; no mesh fallback was attempted.",
                ),
            ),
        )
    if not np.any(mask):
        return HeightmapCompositionResult(
            operation=operation,
            supported=False,
            operation_record=record,
            diagnostics=(
                HeightmapCompositionDiagnostic(
                    code="unrepresentable-result",
                    message="Heightmap CSG composition produced no visible samples; no mesh fallback was attempted.",
                ),
            ),
        )
    metadata = {
        "kernel": {
            "heightmap_csg_composition": {
                **record.canonical_payload(),
                "sample_shape": tuple(int(value) for value in samples.shape),
                "projection_frame": {
                    "projection": alignment.left.projection,
                    "plane": alignment.left.plane,
                    "bounds": alignment.clipping.overlap_bounds if alignment.clipping is not None else None,
                },
                "lossiness": "lossless" if alignment.resample_kernel == "none" else "sampled-reconstruction",
            }
        }
    }
    patch = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=samples,
        alpha_mask=mask,
        alpha_mode="mask",
        xy_scale=xy_scale,
        center=np.asarray(center, dtype=float),
        height_scale=1.0,
        metadata=metadata,
    )
    body = make_surface_body((make_surface_shell((patch,), connected=False),), metadata=metadata)
    final_record = HeightmapCompositionRecord(
        operation=operation,
        operand_ids=record.operand_ids,
        alignment=alignment,
        sample_shape=tuple(int(value) for value in samples.shape),
        resample_kernel=alignment.resample_kernel,
    )
    return HeightmapCompositionResult(
        operation=operation,
        supported=True,
        operation_record=final_record,
        body=body,
        patch=patch,
    )


def _implicit_composition_bounds(
    graphs: Sequence[ImplicitFieldExpressionGraph],
) -> tuple[float, float, float, float, float, float]:
    mins = np.asarray([[graph.bounds[0], graph.bounds[2], graph.bounds[4]] for graph in graphs], dtype=float)
    maxs = np.asarray([[graph.bounds[1], graph.bounds[3], graph.bounds[5]] for graph in graphs], dtype=float)
    lower = mins.min(axis=0)
    upper = maxs.max(axis=0)
    return (float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1]), float(lower[2]), float(upper[2]))


def _compose_implicit_root(
    operation: SurfaceBooleanOperation,
    roots: Sequence[ImplicitFieldNode],
) -> ImplicitFieldNode:
    if operation == "union":
        return implicit_union_field(tuple(roots))
    if operation == "intersection":
        return implicit_intersection_field(tuple(roots))
    if operation == "difference":
        return implicit_difference_field(roots[0], tuple(roots[1:]))
    raise ValueError("Implicit composition operation must be union, difference, or intersection.")


def compose_implicit_field_csg_result(
    operation: SurfaceBooleanOperation,
    adapters: Sequence[ImplicitOperandFieldAdapterRecord],
    *,
    samples: tuple[int, int, int] = (8, 8, 8),
    max_sample_count: int = 262144,
) -> ImplicitCompositionResult:
    """Compose implicit operand adapters into a surface-native implicit CSG result."""

    adapter_tuple = tuple(adapters)
    operand_ids = tuple(adapter.patch_id for adapter in adapter_tuple)
    try:
        sign_policies = implicit_composition_operand_sign_policies(operation, len(adapter_tuple))
    except ValueError as exc:
        diagnostic = ImplicitCompositionDiagnostic(
            code="insufficient-operands" if len(adapter_tuple) < 2 else "invalid-operation",
            message=f"{exc}; no mesh fallback was attempted.",
        )
        record = ImplicitCompositionOperationRecord(operation=operation, operand_ids=operand_ids, sign_policies=())
        return ImplicitCompositionResult(operation=operation, supported=False, operation_record=record, diagnostics=(diagnostic,))

    unsupported: list[ImplicitCompositionDiagnostic] = []
    graphs: list[ImplicitFieldExpressionGraph] = []
    roots: list[ImplicitFieldNode] = []
    for index, adapter in enumerate(adapter_tuple):
        if not adapter.supported or adapter.graph is None:
            message = (
                f"Implicit CSG operand {index} cannot be composed because its field adapter is unsupported; "
                "no mesh fallback was attempted."
            )
            if adapter.diagnostics:
                message = adapter.diagnostics[0].message
            unsupported.append(
                ImplicitCompositionDiagnostic(
                    code="unsupported-adapter",
                    message=message,
                    operand_index=index,
                )
            )
            continue
        graphs.append(adapter.graph)
        roots.append(adapter.graph.root)

    if unsupported:
        record = ImplicitCompositionOperationRecord(operation=operation, operand_ids=operand_ids, sign_policies=sign_policies)
        return ImplicitCompositionResult(
            operation=operation,
            supported=False,
            operation_record=record,
            diagnostics=tuple(unsupported),
        )

    bounds = _implicit_composition_bounds(graphs)
    root = _compose_implicit_root(operation, roots)
    result_graph = ImplicitFieldExpressionGraph(
        root=root,
        bounds=bounds,
        provenance={
            "operation": f"implicit-csg-{operation}",
            "source_family": "implicit-composition",
            "source_ids": operand_ids,
            "route_id": "surface-csg.implicit-composition",
        },
    )
    safety = build_implicit_field_safety_validation_report(
        result_graph,
        samples=samples,
        max_sample_count=max_sample_count,
    )
    record = ImplicitCompositionOperationRecord(
        operation=operation,
        operand_ids=operand_ids,
        sign_policies=sign_policies,
        result_graph=result_graph,
    )
    if not safety.accepted:
        diagnostic = ImplicitCompositionDiagnostic(
            code="unsafe-result",
            message="Implicit CSG composition result failed safety validation; no mesh fallback was attempted.",
        )
        return ImplicitCompositionResult(
            operation=operation,
            supported=False,
            operation_record=record,
            safety=safety,
            diagnostics=(diagnostic,),
        )
    patch = ImplicitSurfacePatch(
        family="implicit",
        field=result_graph.root,
        bounds=result_graph.bounds,
        metadata={
            "kernel": {
                "operation": f"implicit-csg-{operation}",
                "surface_family": "implicit",
                "authoring_boundary": "surface-native",
                "source_operand_ids": operand_ids,
                "no_mesh_fallback": True,
            }
        },
    )
    shell = make_surface_shell((patch,), connected=True, metadata={"kernel": {"operation": f"implicit-csg-{operation}"}})
    body = make_surface_body((shell,), metadata={"kernel": {"operation": f"implicit-csg-{operation}", "surface_family": "implicit"}})
    return ImplicitCompositionResult(
        operation=operation,
        supported=True,
        operation_record=record,
        body=body,
        patch=patch,
        safety=safety,
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

    def canonical_payload(self) -> dict[str, object]:
        return {
            "source_curve": self.source_curve.canonical_payload(),
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "supported": self.supported,
            "curve": None if self.curve is None else self.curve.canonical_payload(),
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceCSGPatchLocalRegionLoop:
    """Coincident or overlap region represented in one patch parameter domain."""

    source_region_id: str
    patch: SurfaceBooleanPatchRef
    loop: TrimLoop
    source_curve_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        source_region_id = str(self.source_region_id).strip()
        if not source_region_id:
            raise ValueError("SurfaceCSGPatchLocalRegionLoop.source_region_id must be non-empty.")
        object.__setattr__(self, "source_region_id", source_region_id)
        object.__setattr__(self, "loop", self.loop.normalized())
        object.__setattr__(self, "source_curve_digests", tuple(str(item) for item in self.source_curve_digests))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "source_region_id": self.source_region_id,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "loop": self.loop.canonical_payload(),
            "source_curve_digests": self.source_curve_digests,
        }


@dataclass(frozen=True)
class SurfaceCSGPatchLocalRegionMappingResult:
    """Result of mapping a coincident or overlap region into patch-local trim space."""

    source_region_id: str
    patch: SurfaceBooleanPatchRef
    region_loop: SurfaceCSGPatchLocalRegionLoop | None = None
    diagnostics: tuple[SurfaceCSGCurveMappingDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return self.region_loop is not None and not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "source_region_id": self.source_region_id,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "supported": self.supported,
            "region_loop": None if self.region_loop is None else self.region_loop.canonical_payload(),
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceCSGIntersectionMappingResult:
    """Aggregate mapping result for all patches affected by one intersection output."""

    curve_mappings: tuple[SurfaceCSGPatchLocalCurveMappingResult, ...] = ()
    region_mappings: tuple[SurfaceCSGPatchLocalRegionMappingResult, ...] = ()

    @property
    def diagnostics(self) -> tuple[SurfaceCSGCurveMappingDiagnostic, ...]:
        return tuple(
            diagnostic
            for mapping in (*self.curve_mappings, *self.region_mappings)
            for diagnostic in mapping.diagnostics
        )

    @property
    def supported(self) -> bool:
        return bool(self.curve_mappings or self.region_mappings) and not self.diagnostics and all(
            mapping.supported for mapping in (*self.curve_mappings, *self.region_mappings)
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "curve_mappings": [mapping.canonical_payload() for mapping in self.curve_mappings],
            "region_mappings": [mapping.canonical_payload() for mapping in self.region_mappings],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


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
class SurfaceCSGAnalyticBSplineIntersectionRecord:
    """Declared-tolerance CSG intersection result for analytic/B-spline patch pairs."""

    first_patch: SurfaceBooleanPatchRef
    second_patch: SurfaceBooleanPatchRef
    intersection: SurfaceIntersectionResultRecord
    residual_report: SurfaceAnalyticSplineResidualReport
    curves: tuple[SurfaceCSGCurvePrimitive, ...] = ()
    patch_local_curves: tuple[SurfaceCSGPatchLocalCurve, ...] = ()
    diagnostics: tuple[SurfaceIntersectionSupportDiagnostic | SurfaceCSGCurveMappingDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return (
            self.intersection.supported
            and self.residual_report.converged
            and bool(self.curves)
            and len(self.patch_local_curves) >= len(self.curves) * 2
            and not self.diagnostics
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "first_patch": {
                "operand_index": self.first_patch.operand_index,
                "patch_index": self.first_patch.patch_index,
            },
            "second_patch": {
                "operand_index": self.second_patch.operand_index,
                "patch_index": self.second_patch.patch_index,
            },
            "supported": self.supported,
            "classification": self.intersection.classification,
            "quality": self.intersection.quality,
            "max_residual": self.intersection.max_residual,
            "curves": [curve.canonical_payload() for curve in self.curves],
            "patch_local_curves": [curve.canonical_payload() for curve in self.patch_local_curves],
            "residual_report": self.residual_report.canonical_payload(),
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceCSGAnalyticNURBSIntersectionRecord:
    """Declared-tolerance CSG intersection result for analytic/NURBS patch pairs."""

    first_patch: SurfaceBooleanPatchRef
    second_patch: SurfaceBooleanPatchRef
    intersection: SurfaceIntersectionResultRecord
    residual_report: SurfaceAnalyticSplineResidualReport
    weight_diagnostics: tuple[NURBSWeightValidationDiagnostic, ...] = ()
    curves: tuple[SurfaceCSGCurvePrimitive, ...] = ()
    patch_local_curves: tuple[SurfaceCSGPatchLocalCurve, ...] = ()
    diagnostics: tuple[
        SurfaceIntersectionSupportDiagnostic | SurfaceCSGCurveMappingDiagnostic | NURBSWeightValidationDiagnostic,
        ...,
    ] = ()

    @property
    def supported(self) -> bool:
        return (
            self.intersection.supported
            and self.residual_report.converged
            and not self.weight_diagnostics
            and bool(self.curves)
            and len(self.patch_local_curves) >= len(self.curves) * 2
            and not self.diagnostics
        )

    @property
    def exact_conic_compatible(self) -> bool:
        return self.intersection.quality == "exact"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "first_patch": {
                "operand_index": self.first_patch.operand_index,
                "patch_index": self.first_patch.patch_index,
            },
            "second_patch": {
                "operand_index": self.second_patch.operand_index,
                "patch_index": self.second_patch.patch_index,
            },
            "supported": self.supported,
            "exact_conic_compatible": self.exact_conic_compatible,
            "classification": self.intersection.classification,
            "quality": self.intersection.quality,
            "max_residual": self.intersection.max_residual,
            "weight_diagnostics": [diagnostic.canonical_payload() for diagnostic in self.weight_diagnostics],
            "curves": [curve.canonical_payload() for curve in self.curves],
            "patch_local_curves": [curve.canonical_payload() for curve in self.patch_local_curves],
            "residual_report": self.residual_report.canonical_payload(),
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceCSGSplinePairIntersectionRecord:
    """Declared-tolerance CSG curve result for B-spline/NURBS patch pairs."""

    first_patch: SurfaceBooleanPatchRef
    second_patch: SurfaceBooleanPatchRef
    intersection: SurfaceIntersectionResultRecord
    residual_report: SurfaceSplineSplineResidualReport
    curves: tuple[SurfaceCSGCurvePrimitive, ...] = ()
    patch_local_curves: tuple[SurfaceCSGPatchLocalCurve, ...] = ()
    tangent_events: tuple[SurfaceCSGDegeneracyRecord, ...] = ()
    diagnostics: tuple[SurfaceIntersectionSupportDiagnostic | SurfaceCSGCurveMappingDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return (
            self.intersection.supported
            and self.residual_report.converged
            and bool(self.curves)
            and len(self.patch_local_curves) >= len(self.curves) * 2
            and not self.diagnostics
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "first_patch": {
                "operand_index": self.first_patch.operand_index,
                "patch_index": self.first_patch.patch_index,
            },
            "second_patch": {
                "operand_index": self.second_patch.operand_index,
                "patch_index": self.second_patch.patch_index,
            },
            "supported": self.supported,
            "classification": self.intersection.classification,
            "quality": self.intersection.quality,
            "max_residual": self.intersection.max_residual,
            "curves": [curve.canonical_payload() for curve in self.curves],
            "patch_local_curves": [curve.canonical_payload() for curve in self.patch_local_curves],
            "tangent_events": [event.canonical_payload() for event in self.tangent_events],
            "residual_report": self.residual_report.canonical_payload(),
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceCSGSplineCoincidentRegionRecord:
    """CSG overlap-region output for coincident B-spline/NURBS patch pairs."""

    first_patch: SurfaceBooleanPatchRef
    second_patch: SurfaceBooleanPatchRef
    intersection: SurfaceIntersectionResultRecord
    region_mappings: tuple[SurfaceCSGPatchLocalRegionMappingResult, ...] = ()
    ownership_diagnostics: tuple[SurfaceCSGCoincidentOwnershipDiagnostic, ...] = ()
    diagnostics: tuple[
        SurfaceIntersectionSupportDiagnostic | SurfaceCSGCurveMappingDiagnostic | SurfaceCSGArrangementDiagnostic,
        ...,
    ] = ()

    @property
    def supported(self) -> bool:
        return (
            self.intersection.supported
            and self.intersection.classification == "overlap"
            and bool(self.region_mappings)
            and all(mapping.supported for mapping in self.region_mappings)
            and not self.ownership_diagnostics
            and not self.diagnostics
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "first_patch": {
                "operand_index": self.first_patch.operand_index,
                "patch_index": self.first_patch.patch_index,
            },
            "second_patch": {
                "operand_index": self.second_patch.operand_index,
                "patch_index": self.second_patch.patch_index,
            },
            "supported": self.supported,
            "classification": self.intersection.classification,
            "quality": self.intersection.quality,
            "max_residual": self.intersection.max_residual,
            "overlap_regions": [region.canonical_payload() for region in self.intersection.overlap_regions],
            "region_mappings": [mapping.canonical_payload() for mapping in self.region_mappings],
            "ownership_diagnostics": [diagnostic.canonical_payload() for diagnostic in self.ownership_diagnostics],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceCSGSweepPairIntersectionRecord:
    """Declared-tolerance CSG curve result for sweep-participating patch pairs."""

    first_patch: SurfaceBooleanPatchRef
    second_patch: SurfaceBooleanPatchRef
    intersection: SurfaceIntersectionResultRecord
    residual_report: SurfaceSweepPairResidualReport
    curves: tuple[SurfaceCSGCurvePrimitive, ...] = ()
    patch_local_curves: tuple[SurfaceCSGPatchLocalCurve, ...] = ()
    ambiguity_diagnostics: tuple[SurfaceCSGAmbiguityDiagnostic, ...] = ()
    diagnostics: tuple[SurfaceIntersectionSupportDiagnostic | SurfaceCSGCurveMappingDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return (
            self.intersection.supported
            and self.residual_report.converged
            and bool(self.curves)
            and len(self.patch_local_curves) >= len(self.curves) * 2
            and not self.ambiguity_diagnostics
            and not self.diagnostics
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "first_patch": {
                "operand_index": self.first_patch.operand_index,
                "patch_index": self.first_patch.patch_index,
            },
            "second_patch": {
                "operand_index": self.second_patch.operand_index,
                "patch_index": self.second_patch.patch_index,
            },
            "supported": self.supported,
            "classification": self.intersection.classification,
            "quality": self.intersection.quality,
            "max_residual": self.intersection.max_residual,
            "curves": [curve.canonical_payload() for curve in self.curves],
            "patch_local_curves": [curve.canonical_payload() for curve in self.patch_local_curves],
            "ambiguity_diagnostics": [diagnostic.canonical_payload() for diagnostic in self.ambiguity_diagnostics],
            "residual_report": self.residual_report.canonical_payload(),
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceCSGSubdivisionPairIntersectionRecord:
    """Declared-tolerance CSG curve result for subdivision-participating patch pairs."""

    first_patch: SurfaceBooleanPatchRef
    second_patch: SurfaceBooleanPatchRef
    intersection: SurfaceIntersectionResultRecord
    adapter_report: SurfaceSubdivisionIntersectionAdapterReport
    curves: tuple[SurfaceCSGCurvePrimitive, ...] = ()
    patch_local_curves: tuple[SurfaceCSGPatchLocalCurve, ...] = ()
    diagnostics: tuple[SurfaceIntersectionSupportDiagnostic | SurfaceCSGCurveMappingDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return (
            self.intersection.supported
            and self.adapter_report.converged
            and bool(self.curves)
            and len(self.patch_local_curves) >= len(self.curves) * 2
            and not self.diagnostics
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "first_patch": {
                "operand_index": self.first_patch.operand_index,
                "patch_index": self.first_patch.patch_index,
            },
            "second_patch": {
                "operand_index": self.second_patch.operand_index,
                "patch_index": self.second_patch.patch_index,
            },
            "supported": self.supported,
            "classification": self.intersection.classification,
            "quality": self.intersection.quality,
            "max_residual": self.intersection.max_residual,
            "curves": [curve.canonical_payload() for curve in self.curves],
            "patch_local_curves": [curve.canonical_payload() for curve in self.patch_local_curves],
            "adapter_report": self.adapter_report.canonical_payload(),
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


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
class SurfaceCSGArrangementVertex:
    """One deterministic vertex in a patch-local CSG arrangement graph."""

    vertex_id: str
    patch: SurfaceBooleanPatchRef
    point_uv: tuple[float, float]
    source: Literal["trim-loop", "cut-curve", "coincident-region"] = "trim-loop"

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_uv", _normalize_curve_point_uv(self.point_uv))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "vertex_id": self.vertex_id,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "point_uv": self.point_uv,
            "source": self.source,
        }


@dataclass(frozen=True)
class SurfaceCSGArrangementEdge:
    """One oriented edge in a patch-local CSG arrangement graph."""

    edge_id: str
    patch: SurfaceBooleanPatchRef
    start_vertex_id: str
    end_vertex_id: str
    source: Literal["trim-loop", "cut-curve", "coincident-region"] = "trim-loop"
    cut_curve_ids: tuple[str, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "start_vertex_id": self.start_vertex_id,
            "end_vertex_id": self.end_vertex_id,
            "source": self.source,
            "cut_curve_ids": self.cut_curve_ids,
        }


@dataclass(frozen=True)
class SurfaceCSGArrangementFaceCandidate:
    """One candidate face induced by trim loops and CSG cut curves."""

    face_id: str
    patch: SurfaceBooleanPatchRef
    loop: TrimLoop
    source_category: str
    cut_curve_ids: tuple[str, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "face_id": self.face_id,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "loop": self.loop.canonical_payload(),
            "source_category": self.source_category,
            "cut_curve_ids": self.cut_curve_ids,
        }


@dataclass(frozen=True)
class SurfaceCSGPatchLocalArrangementGraph:
    """Patch-local arrangement of intersection curves and split trim loops."""

    patch: SurfaceBooleanPatchRef
    patch_local_curves: tuple[SurfaceCSGPatchLocalCurve, ...]
    split_loops: tuple[SurfaceCSGSplitTrimLoopRecord, ...]
    vertices: tuple[SurfaceCSGArrangementVertex, ...] = ()
    edges: tuple[SurfaceCSGArrangementEdge, ...] = ()
    face_candidates: tuple[SurfaceCSGArrangementFaceCandidate, ...] = ()
    diagnostics: tuple[SurfaceCSGArrangementDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "supported": self.supported,
            "patch_local_curves": [curve.canonical_payload() for curve in self.patch_local_curves],
            "split_loops": [loop.canonical_payload() for loop in self.split_loops],
            "vertices": [vertex.canonical_payload() for vertex in self.vertices],
            "edges": [edge.canonical_payload() for edge in self.edges],
            "face_candidates": [face.canonical_payload() for face in self.face_candidates],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


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
class SurfaceCSGSurfaceFragment:
    """One surface-native fragment built from a patch-local arrangement face."""

    fragment_id: str
    source_patch: SurfaceBooleanPatchRef
    patch: PlanarSurfacePatch
    loop: TrimLoop
    source_face_id: str
    cut_curve_ids: tuple[str, ...] = ()

    @property
    def sample_uv(self) -> tuple[float, float]:
        return select_surface_csg_fragment_sample(self.patch, trim_loop=self.loop)

    @property
    def sample_point(self) -> tuple[float, float, float]:
        point = self.patch.point_at(*self.sample_uv)
        return (float(point[0]), float(point[1]), float(point[2]))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "cut_curve_ids": self.cut_curve_ids,
            "fragment_id": self.fragment_id,
            "loop": self.loop.canonical_payload(),
            "sample_point": self.sample_point,
            "sample_uv": self.sample_uv,
            "source_face_id": self.source_face_id,
            "source_patch": {
                "operand_index": self.source_patch.operand_index,
                "patch_index": self.source_patch.patch_index,
            },
        }


@dataclass(frozen=True)
class SurfaceCSGFragmentBuildDiagnostic:
    """Diagnostic emitted while converting arrangement faces into fragments."""

    code: Literal["invalid-arrangement", "missing-face-candidate", "invalid-fragment-loop"]
    message: str
    patch: SurfaceBooleanPatchRef
    face_id: str | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "face_id": self.face_id,
            "message": self.message,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
        }


@dataclass(frozen=True)
class SurfaceCSGFragmentBuildResult:
    """Surface-native fragment builder result for one arrangement graph."""

    arrangement: SurfaceCSGPatchLocalArrangementGraph
    fragments: tuple[SurfaceCSGSurfaceFragment, ...] = ()
    diagnostics: tuple[SurfaceCSGFragmentBuildDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return bool(self.fragments) and not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "arrangement": self.arrangement.canonical_payload(),
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "fragments": [fragment.canonical_payload() for fragment in self.fragments],
            "supported": self.supported,
        }


@dataclass(frozen=True)
class SurfaceCSGCoincidentOwnershipDiagnostic:
    """Diagnostic emitted when a boundary/coincident fragment ownership decision is ambiguous."""

    code: Literal["missing-cut-provenance", "ambiguous-coincident-owner"]
    message: str
    fragment_id: str
    patch: SurfaceBooleanPatchRef

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "fragment_id": self.fragment_id,
            "message": self.message,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
        }


@dataclass(frozen=True)
class SurfaceCSGCoincidentOwnershipRecord:
    """Resolved ownership witness for a fragment classified on the opposing boundary."""

    fragment_id: str
    patch: SurfaceBooleanPatchRef
    relation: SurfaceBooleanPatchRelation
    owner_patch: SurfaceBooleanPatchRef | None = None
    policy: str = "not-coincident"
    diagnostics: tuple[SurfaceCSGCoincidentOwnershipDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        owner_payload = None
        if self.owner_patch is not None:
            owner_payload = {
                "operand_index": self.owner_patch.operand_index,
                "patch_index": self.owner_patch.patch_index,
            }
        return {
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "fragment_id": self.fragment_id,
            "owner_patch": owner_payload,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
            "policy": self.policy,
            "relation": self.relation,
            "supported": self.supported,
        }


@dataclass(frozen=True)
class SurfaceCSGClassifiedFragmentSet:
    """Classified surface-native fragments and boundary ownership diagnostics."""

    fragments: tuple[SurfaceCSGSurfaceFragment, ...]
    classifications: tuple[SurfaceCSGFragmentClassificationRecord, ...]
    coincident_ownership: tuple[SurfaceCSGCoincidentOwnershipRecord, ...] = ()

    @property
    def supported(self) -> bool:
        return all(classification.supported for classification in self.classifications) and all(
            ownership.supported for ownership in self.coincident_ownership
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "classifications": [classification.canonical_payload() for classification in self.classifications],
            "coincident_ownership": [ownership.canonical_payload() for ownership in self.coincident_ownership],
            "fragments": [fragment.canonical_payload() for fragment in self.fragments],
            "supported": self.supported,
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
class SurfaceCSGOperationSelectionDiagnostic:
    """Diagnostic emitted while converting classified fragments into operation selections."""

    code: Literal["unsupported-classification", "ambiguous-coincident-ownership"]
    message: str
    patch: SurfaceBooleanPatchRef

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "patch": {
                "operand_index": self.patch.operand_index,
                "patch_index": self.patch.patch_index,
            },
        }


@dataclass(frozen=True)
class SurfaceCSGCapEligibilityRecord:
    """Surface-native cap eligibility decision for one operation-selected fragment."""

    selection: SurfaceCSGOperationSelectionRecord
    required: bool
    eligible: bool
    cap_family: str | None = None
    reason: str = ""
    diagnostics: tuple[SurfaceCSGUnsupportedCapDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "cap_family": self.cap_family,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "eligible": self.eligible,
            "reason": self.reason,
            "required": self.required,
            "selection": self.selection.canonical_payload(),
            "supported": self.supported,
        }


@dataclass(frozen=True)
class SurfaceCSGOperationFragmentSelectionSet:
    """Operation selection result for a classified fragment set."""

    operation: SurfaceBooleanOperation
    selections: tuple[SurfaceCSGOperationSelectionRecord, ...]
    cap_eligibility: tuple[SurfaceCSGCapEligibilityRecord, ...] = ()
    diagnostics: tuple[SurfaceCSGOperationSelectionDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.diagnostics and all(record.supported for record in self.cap_eligibility)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "cap_eligibility": [record.canonical_payload() for record in self.cap_eligibility],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "operation": self.operation,
            "selections": [selection.canonical_payload() for selection in self.selections],
            "supported": self.supported,
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
class SurfaceCSGOrientedFragmentRecord:
    """Selected fragment with operation-specific orientation applied."""

    source_patch: SurfaceBooleanPatchRef
    patch: PlanarSurfacePatch
    role: SurfaceBooleanSplitRole
    orientation: Literal["preserve", "reverse"]
    operation: SurfaceBooleanOperation
    cut_curve_ids: tuple[str, ...] = ()

    @property
    def included(self) -> bool:
        return self.role in {"survive", "cut_cap"}

    def to_trimmed_fragment(self) -> SurfaceBooleanTrimmedPatchFragment | None:
        if not self.included:
            return None
        return SurfaceBooleanTrimmedPatchFragment(
            source_patch=self.source_patch,
            patch=self.patch,
            cut_curve_ids=self.cut_curve_ids,
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "cut_curve_ids": self.cut_curve_ids,
            "included": self.included,
            "operation": self.operation,
            "orientation": self.orientation,
            "role": self.role,
            "source_patch": {
                "operand_index": self.source_patch.operand_index,
                "patch_index": self.source_patch.patch_index,
            },
        }


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
class SurfaceCSGContinuityHandoffDiagnostic:
    """Diagnostic for requested continuity that seam adjacency cannot enforce."""

    code: Literal["unsupported-continuity-enforcement", "invalid-seam-rebuild"]
    message: str
    continuity: str | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "continuity": self.continuity,
            "message": self.message,
        }


@dataclass(frozen=True)
class SurfaceCSGContinuityHandoffRecord:
    """Continuity handoff from CSG seam adjacency to continuity enforcement."""

    seam_rebuild: SurfaceCSGSeamRebuildRecord
    requested_continuity: tuple[str, ...]
    enforceable_continuity: tuple[str, ...]
    diagnostics: tuple[SurfaceCSGContinuityHandoffDiagnostic, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "enforceable_continuity": self.enforceable_continuity,
            "requested_continuity": self.requested_continuity,
            "seam_rebuild": self.seam_rebuild.canonical_payload(),
            "supported": self.supported,
        }


@dataclass(frozen=True)
class SurfaceCSGValidityDiagnostic:
    """Diagnostic emitted by the final CSG validity/provenance gate."""

    code: Literal[
        "invalid-shell",
        "non-closed-result",
        "healing-failed",
        "dangling-trim",
        "mesh-backed-fragment",
        "non-surface-result",
        "unresolved-diagnostic",
    ]
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

    def canonical_payload(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "body_id": None if self.body is None else self.body.stable_identity,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "provenance": None if self.provenance is None else self.provenance.canonical_payload(),
            "status": self.status,
        }


@dataclass(frozen=True)
class SurfaceCSGRuntimeValidityReport:
    """Runtime validity report for the final object returned by surface CSG."""

    operation: SurfaceBooleanOperation
    status: SurfaceBooleanStatus
    result: SurfaceBody | None = None
    diagnostics: tuple[SurfaceCSGValidityDiagnostic, ...] = ()
    validity_gate: SurfaceCSGValidityGateRecord | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "succeeded" and self.result is not None and not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "operation": self.operation,
            "result_id": None if self.result is None else self.result.stable_identity,
            "status": self.status,
            "validity_gate": None if self.validity_gate is None else self.validity_gate.canonical_payload(),
        }


@dataclass(frozen=True)
class SurfaceCSGPersistenceEvidenceRecord:
    """Evidence that a CSG surface body survives `.impress` persistence."""

    fixture_id: str
    passed: bool
    body_id: str
    loaded_body_id: str | None = None
    message: str = ""

    def canonical_payload(self) -> dict[str, object]:
        return {
            "body_id": self.body_id,
            "fixture_id": self.fixture_id,
            "loaded_body_id": self.loaded_body_id,
            "message": self.message,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class SurfaceCSGTessellationBoundaryEvidenceRecord:
    """Evidence that CSG tessellation occurs only after a surface result exists."""

    fixture_id: str
    passed: bool
    body_id: str
    face_count: int = 0
    message: str = ""

    def canonical_payload(self) -> dict[str, object]:
        return {
            "body_id": self.body_id,
            "face_count": self.face_count,
            "fixture_id": self.fixture_id,
            "message": self.message,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class SurfaceCSGReferencePromotionReport:
    """Reference evidence report for a CSG fixture."""

    fixture_id: str
    persistence: SurfaceCSGPersistenceEvidenceRecord
    tessellation: SurfaceCSGTessellationBoundaryEvidenceRecord
    reference_state: Literal["clean", "dirty", "missing"] = "missing"
    diagnostics: tuple[str, ...] = ()

    @property
    def promoted(self) -> bool:
        return (
            self.persistence.passed
            and self.tessellation.passed
            and self.reference_state == "clean"
            and not self.diagnostics
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "diagnostics": self.diagnostics,
            "fixture_id": self.fixture_id,
            "persistence": self.persistence.canonical_payload(),
            "promoted": self.promoted,
            "reference_state": self.reference_state,
            "tessellation": self.tessellation.canonical_payload(),
        }


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
class SurfaceCSGRouteSupportDiagnostic:
    """Route-registry diagnostic for a CSG operation/family pair."""

    code: Literal["missing-route", "non-executable-route", "unknown-family", "route-key-mismatch"]
    message: str
    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    pair_class: SurfaceCSGRoutePairClass | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "pair_class": self.pair_class,
        }


@dataclass(frozen=True)
class SurfaceCSGRouteRegistryRow:
    """Executable-route taxonomy row for one CSG operation/family pair."""

    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    pair_class: SurfaceCSGRoutePairClass
    route_id: str
    supported: bool
    support_state: SurfaceBooleanSupportState
    phase: str
    required_future_capability: str | None = None
    diagnostic: SurfaceCSGRouteSupportDiagnostic | None = None

    @property
    def executable(self) -> bool:
        return self.supported and self.diagnostic is None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "pair_class": self.pair_class,
            "route_id": self.route_id,
            "supported": self.supported,
            "executable": self.executable,
            "support_state": self.support_state,
            "phase": self.phase,
            "required_future_capability": self.required_future_capability,
            "diagnostic": None if self.diagnostic is None else self.diagnostic.canonical_payload(),
        }


@dataclass(frozen=True)
class SurfaceCSGExecutableRowReport:
    """Report of route-registry executability for CSG operation/family rows."""

    rows: tuple[SurfaceCSGRouteRegistryRow, ...]

    @property
    def diagnostics(self) -> tuple[SurfaceCSGRouteSupportDiagnostic, ...]:
        return tuple(row.diagnostic for row in self.rows if row.diagnostic is not None)

    @property
    def passed(self) -> bool:
        return not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "rows": [row.canonical_payload() for row in self.rows],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceCSGResidualRecord:
    """Declared-tolerance residual witness for a higher-order CSG route."""

    route_id: str
    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    max_residual: float
    tolerance: float
    iteration_count: int
    converged: bool
    patch_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        route_id = str(self.route_id).strip()
        if not route_id:
            raise ValueError("SurfaceCSGResidualRecord.route_id must be non-empty.")
        max_residual = float(self.max_residual)
        tolerance = float(self.tolerance)
        iteration_count = int(self.iteration_count)
        if not np.isfinite(max_residual) or max_residual < 0.0:
            raise ValueError("SurfaceCSGResidualRecord.max_residual must be finite and non-negative.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("SurfaceCSGResidualRecord.tolerance must be positive and finite.")
        if iteration_count < 0:
            raise ValueError("SurfaceCSGResidualRecord.iteration_count must be non-negative.")
        object.__setattr__(self, "route_id", route_id)
        object.__setattr__(self, "max_residual", max_residual)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "iteration_count", iteration_count)
        object.__setattr__(self, "patch_ids", tuple(str(patch_id) for patch_id in self.patch_ids))

    @property
    def within_tolerance(self) -> bool:
        return self.converged and self.max_residual <= self.tolerance

    def canonical_payload(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "max_residual": self.max_residual,
            "tolerance": self.tolerance,
            "iteration_count": self.iteration_count,
            "converged": self.converged,
            "within_tolerance": self.within_tolerance,
            "patch_ids": self.patch_ids,
        }


@dataclass(frozen=True)
class SurfaceCSGDegeneracyRecord:
    """Classified higher-order CSG degeneracy or route-quality issue."""

    code: Literal[
        "non-convergence",
        "ambiguous-route",
        "overlap",
        "singularity",
        "budget-refusal",
        "high-residual",
    ]
    message: str
    route_id: str
    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    residual: float | None = None
    blocking: bool = True
    location: str | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "route_id": self.route_id,
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "residual": self.residual,
            "blocking": self.blocking,
            "location": self.location,
        }


@dataclass(frozen=True)
class SurfaceCSGAmbiguityDiagnostic:
    """Blocking authored-ambiguity diagnostic for a higher-order CSG route."""

    route_id: str
    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    message: str
    location: str | None = None
    blocking: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "message": self.message,
            "location": self.location,
            "blocking": self.blocking,
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


@dataclass(frozen=True)
class SurfaceCSGFamilyClassificationRow:
    """One explicit CSG classification row for an operation/family pair."""

    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    supported: bool
    classification: str
    support_state: SurfaceBooleanSupportState
    diagnostic: str = ""

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "supported": self.supported,
            "classification": self.classification,
            "support_state": self.support_state,
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True)
class SurfaceCSGFamilyClassificationReport:
    """CSG classification matrix completeness report."""

    passed: bool
    rows: tuple[SurfaceCSGFamilyClassificationRow, ...]
    diagnostics: tuple[SurfaceCSGSolverRegistryDiagnostic, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "rows": [row.canonical_payload() for row in self.rows],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceCSGNoMeshFallbackEvidenceRecord:
    """Regression evidence that CSG rows refuse instead of falling back to mesh."""

    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    result_kind: Literal["supported-surface", "diagnostic-refusal"]
    message: str
    mesh_fallback_attempted: bool = False

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "result_kind": self.result_kind,
            "message": self.message,
            "mesh_fallback_attempted": self.mesh_fallback_attempted,
        }


@dataclass(frozen=True)
class SurfaceCSGNoMeshFallbackReport:
    """CSG no-hidden-mesh-fallback regression report."""

    passed: bool
    evidence: tuple[SurfaceCSGNoMeshFallbackEvidenceRecord, ...]
    diagnostics: tuple[SurfaceCSGPlanDiagnostic, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "evidence": [record.canonical_payload() for record in self.evidence],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceCSGPairFixtureRow:
    """One fixture-evidence row for a higher-order CSG operation/family pair."""

    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    pair_class: SurfaceCSGRoutePairClass
    route_id: str
    executable: bool
    expected_category: Literal["crossing", "tangent", "coincident", "boundary", "singular", "refusal"]
    mesh_fallback_attempted: bool = False
    diagnostic: str = ""

    def canonical_payload(self) -> dict[str, object]:
        return {
            "diagnostic": self.diagnostic,
            "executable": self.executable,
            "expected_category": self.expected_category,
            "left_family": self.left_family,
            "mesh_fallback_attempted": self.mesh_fallback_attempted,
            "operation": self.operation,
            "pair_class": self.pair_class,
            "right_family": self.right_family,
            "route_id": self.route_id,
        }


@dataclass(frozen=True)
class SurfaceCSGPairFixtureEvidenceReport:
    """Fixture matrix evidence that promoted higher-order CSG rows are executable."""

    rows: tuple[SurfaceCSGPairFixtureRow, ...]
    required_pair_classes: tuple[SurfaceCSGRoutePairClass, ...]
    diagnostics: tuple[SurfaceCSGRouteSupportDiagnostic, ...] = ()

    @property
    def passed(self) -> bool:
        return bool(self.rows) and not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "required_pair_classes": self.required_pair_classes,
            "rows": [row.canonical_payload() for row in self.rows],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceImplicitCSGFixtureRow:
    """One implicit-preserving/promoted CSG route evidence row."""

    fixture_id: str
    route_kind: Literal["success", "unsafe-refusal", "adapter-refusal", "persistence", "no-mesh-fallback"]
    passed: bool
    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    message: str
    mesh_fallback_attempted: bool = False
    reference_state: Literal["clean", "dirty", "missing"] = "clean"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "route_kind": self.route_kind,
            "passed": self.passed,
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "message": self.message,
            "mesh_fallback_attempted": self.mesh_fallback_attempted,
            "reference_state": self.reference_state,
        }


@dataclass(frozen=True)
class SurfaceImplicitCSGEvidenceReport:
    """Evidence matrix for implicit CSG route completion."""

    rows: tuple[SurfaceImplicitCSGFixtureRow, ...]
    required_route_kinds: tuple[str, ...] = ("success", "unsafe-refusal", "adapter-refusal", "persistence", "no-mesh-fallback")
    diagnostics: tuple[SurfaceCSGRouteSupportDiagnostic, ...] = ()

    @property
    def passed(self) -> bool:
        return bool(self.rows) and not self.diagnostics and all(row.passed for row in self.rows)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "required_route_kinds": self.required_route_kinds,
            "rows": [row.canonical_payload() for row in self.rows],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceHeightmapCSGFixtureRow:
    """One heightmap CSG route evidence row."""

    fixture_id: str
    route_kind: Literal["success", "representability-refusal", "promotion", "persistence", "no-mesh-fallback"]
    passed: bool
    operation: SurfaceBooleanOperation
    message: str
    target_family: str | None = None
    mesh_fallback_attempted: bool = False
    reference_state: Literal["clean", "dirty", "missing"] = "clean"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "route_kind": self.route_kind,
            "passed": self.passed,
            "operation": self.operation,
            "message": self.message,
            "target_family": self.target_family,
            "mesh_fallback_attempted": self.mesh_fallback_attempted,
            "reference_state": self.reference_state,
        }


@dataclass(frozen=True)
class SurfaceHeightmapCSGEvidenceReport:
    """Evidence matrix for heightmap-preserving, promoted, and refused CSG routes."""

    rows: tuple[SurfaceHeightmapCSGFixtureRow, ...]
    required_route_kinds: tuple[str, ...] = (
        "success",
        "representability-refusal",
        "promotion",
        "persistence",
        "no-mesh-fallback",
    )
    diagnostics: tuple[SurfaceCSGRouteSupportDiagnostic, ...] = ()

    @property
    def passed(self) -> bool:
        return bool(self.rows) and not self.diagnostics and all(row.passed for row in self.rows)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "required_route_kinds": self.required_route_kinds,
            "rows": [row.canonical_payload() for row in self.rows],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceDisplacementCSGFixtureRow:
    """One displacement CSG route evidence row."""

    fixture_id: str
    route_kind: Literal["success", "source-refusal", "promotion", "persistence", "no-mesh-fallback"]
    passed: bool
    operation: SurfaceBooleanOperation
    message: str
    target_family: str | None = None
    mesh_fallback_attempted: bool = False
    reference_state: Literal["clean", "dirty", "missing"] = "clean"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "route_kind": self.route_kind,
            "passed": self.passed,
            "operation": self.operation,
            "message": self.message,
            "target_family": self.target_family,
            "mesh_fallback_attempted": self.mesh_fallback_attempted,
            "reference_state": self.reference_state,
        }


@dataclass(frozen=True)
class SurfaceDisplacementCSGEvidenceReport:
    """Evidence matrix for displacement-preserving, promoted, and refused CSG routes."""

    rows: tuple[SurfaceDisplacementCSGFixtureRow, ...]
    required_route_kinds: tuple[str, ...] = (
        "success",
        "source-refusal",
        "promotion",
        "persistence",
        "no-mesh-fallback",
    )
    diagnostics: tuple[SurfaceCSGRouteSupportDiagnostic, ...] = ()

    @property
    def passed(self) -> bool:
        return bool(self.rows) and not self.diagnostics and all(row.passed for row in self.rows)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "required_route_kinds": self.required_route_kinds,
            "rows": [row.canonical_payload() for row in self.rows],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitCSGUnsupportedRow:
    """Tracked sampled/implicit CSG row that still needs route promotion."""

    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    support_state: SurfaceBooleanSupportState
    route_status: SurfaceSampledImplicitCSGRouteStatus
    route_id: str
    required_future_capability: str
    mesh_fallback_attempted: bool = False

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "support_state": self.support_state,
            "route_status": self.route_status,
            "route_id": self.route_id,
            "required_future_capability": self.required_future_capability,
            "mesh_fallback_attempted": self.mesh_fallback_attempted,
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitCSGUnsupportedRowReport:
    """Completion tracker for sampled/implicit CSG rows still leaving unsupported."""

    rows: tuple[SurfaceSampledImplicitCSGUnsupportedRow, ...]
    expected_row_count: int
    expected_rows_per_operation: int
    diagnostics: tuple[SurfaceCSGRouteSupportDiagnostic, ...] = ()

    @property
    def passed(self) -> bool:
        return len(self.rows) == self.expected_row_count and not self.diagnostics

    def rows_for_operation(
        self,
        operation: SurfaceBooleanOperation,
    ) -> tuple[SurfaceSampledImplicitCSGUnsupportedRow, ...]:
        return tuple(row for row in self.rows if row.operation == operation)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "expected_row_count": self.expected_row_count,
            "expected_rows_per_operation": self.expected_rows_per_operation,
            "rows": [row.canonical_payload() for row in self.rows],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitPromotionDiagnostic:
    """Diagnostic emitted while assigning sampled/implicit CSG rows to final route classes."""

    code: Literal["missing-target", "incomplete-route", "mesh-fallback"]
    message: str
    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitPromotionPolicyRow:
    """Final sampled/implicit CSG promotion policy for one operation/family pair row."""

    operation: SurfaceBooleanOperation
    left_family: str
    right_family: str
    source_support_state: SurfaceBooleanSupportState
    route_status: SurfaceSampledImplicitCSGRouteStatus
    target_family: SurfaceSampledImplicitPromotionTargetFamily | None
    lossiness: Literal["lossless", "sampled-reconstruction", "volumetric-field", "exact-reconstruction"]
    route_id: str
    reason: str
    required_future_capability: str
    mesh_fallback_attempted: bool = False

    @property
    def complete(self) -> bool:
        return self.route_status != "in-progress" and self.target_family is not None and not self.mesh_fallback_attempted

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "source_support_state": self.source_support_state,
            "route_status": self.route_status,
            "target_family": self.target_family,
            "lossiness": self.lossiness,
            "route_id": self.route_id,
            "reason": self.reason,
            "required_future_capability": self.required_future_capability,
            "mesh_fallback_attempted": self.mesh_fallback_attempted,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitPromotionDecision:
    """Target-selection result for one sampled/implicit CSG unsupported row."""

    supported: bool
    row: SurfaceSampledImplicitPromotionPolicyRow
    diagnostics: tuple[SurfaceSampledImplicitPromotionDiagnostic, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "row": self.row.canonical_payload(),
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitPromotionMatrixReport:
    """Promotion matrix report for all sampled/implicit CSG route rows."""

    rows: tuple[SurfaceSampledImplicitPromotionPolicyRow, ...]
    expected_row_count: int
    expected_rows_per_operation: int
    diagnostics: tuple[SurfaceSampledImplicitPromotionDiagnostic, ...] = ()

    @property
    def passed(self) -> bool:
        return len(self.rows) == self.expected_row_count and not self.diagnostics and all(row.complete for row in self.rows)

    def rows_for_operation(
        self,
        operation: SurfaceBooleanOperation,
    ) -> tuple[SurfaceSampledImplicitPromotionPolicyRow, ...]:
        return tuple(row for row in self.rows if row.operation == operation)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "expected_row_count": self.expected_row_count,
            "expected_rows_per_operation": self.expected_rows_per_operation,
            "rows": [row.canonical_payload() for row in self.rows],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitPromotionLossinessRecord:
    """Normalized lossiness and tolerance facts for a sampled/implicit promotion route."""

    target_family: SurfaceSampledImplicitPromotionTargetFamily
    lossiness: Literal["lossless", "sampled-reconstruction", "volumetric-field", "exact-reconstruction"]
    tolerance: float
    sampling_policy: Literal["none", "bounded-samples", "field-sampling", "chart-reconstruction"]
    reconstruction_kind: Literal["none", "implicit-field", "subdivision-chart", "nurbs-fit", "bspline-fit"]
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "target_family": self.target_family,
            "lossiness": self.lossiness,
            "tolerance": self.tolerance,
            "sampling_policy": self.sampling_policy,
            "reconstruction_kind": self.reconstruction_kind,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitPromotionProvenanceDiagnostic:
    """Diagnostic emitted while building promotion provenance metadata."""

    code: Literal["invalid-tolerance", "incomplete-route", "invalid-operands"]
    message: str
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitPromotionProvenanceRecord:
    """Durable metadata for a sampled/implicit CSG promotion decision."""

    operation: SurfaceBooleanOperation
    source_families: tuple[str, str]
    source_operand_ids: tuple[str, str]
    route_status: SurfaceSampledImplicitCSGRouteStatus
    route_id: str
    target_family: SurfaceSampledImplicitPromotionTargetFamily | None
    lossiness: SurfaceSampledImplicitPromotionLossinessRecord | None
    source_support_state: SurfaceBooleanSupportState
    required_future_capability: str
    diagnostics: tuple[SurfaceSampledImplicitPromotionProvenanceDiagnostic, ...] = ()
    no_mesh_fallback: bool = True

    @property
    def supported(self) -> bool:
        return (
            not self.diagnostics
            and self.route_status != "in-progress"
            and self.target_family is not None
            and self.lossiness is not None
            and self.no_mesh_fallback
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "source_families": self.source_families,
            "source_operand_ids": self.source_operand_ids,
            "route_status": self.route_status,
            "route_id": self.route_id,
            "target_family": self.target_family,
            "lossiness": None if self.lossiness is None else self.lossiness.canonical_payload(),
            "source_support_state": self.source_support_state,
            "required_future_capability": self.required_future_capability,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "no_mesh_fallback": self.no_mesh_fallback,
            "supported": self.supported,
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitReconstructionCriteriaRecord:
    """Eligibility criteria for one sampled/implicit CSG promotion target family."""

    target_family: SurfaceSampledImplicitPromotionTargetFamily
    max_sample_count: int
    max_residual: float | None
    requires_complete_provenance: bool = True
    requires_exact_reconstruction: bool = False

    def canonical_payload(self) -> dict[str, object]:
        return {
            "target_family": self.target_family,
            "max_sample_count": self.max_sample_count,
            "max_residual": self.max_residual,
            "requires_complete_provenance": self.requires_complete_provenance,
            "requires_exact_reconstruction": self.requires_exact_reconstruction,
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitReconstructionDiagnostic:
    """Diagnostic emitted by sampled/implicit promotion reconstruction criteria."""

    code: Literal["incomplete-provenance", "unsupported-target", "sample-budget-exceeded", "residual-exceeded"]
    message: str
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitReconstructionFeasibilityReport:
    """Eligibility verdict for reconstructing a promoted sampled/implicit CSG target."""

    target_family: SurfaceSampledImplicitPromotionTargetFamily | None
    supported: bool
    criteria: SurfaceSampledImplicitReconstructionCriteriaRecord | None
    provenance: SurfaceSampledImplicitPromotionProvenanceRecord
    estimated_sample_count: int
    residual: float | None
    diagnostics: tuple[SurfaceSampledImplicitReconstructionDiagnostic, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "target_family": self.target_family,
            "supported": self.supported,
            "criteria": None if self.criteria is None else self.criteria.canonical_payload(),
            "provenance": self.provenance.canonical_payload(),
            "estimated_sample_count": self.estimated_sample_count,
            "residual": self.residual,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitPromotionFixtureRow:
    """One evidence row for sampled/implicit promotion routes."""

    fixture_id: str
    route_kind: Literal["promotion-target", "criteria", "persistence", "refusal", "no-mesh-fallback"]
    passed: bool
    target_family: SurfaceSampledImplicitPromotionTargetFamily | None
    message: str
    mesh_fallback_attempted: bool = False
    reference_state: Literal["clean", "dirty", "missing"] = "clean"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "route_kind": self.route_kind,
            "passed": self.passed,
            "target_family": self.target_family,
            "message": self.message,
            "mesh_fallback_attempted": self.mesh_fallback_attempted,
            "reference_state": self.reference_state,
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitPromotionEvidenceReport:
    """Clean evidence report for sampled/implicit promotion and no-mesh fixtures."""

    rows: tuple[SurfaceSampledImplicitPromotionFixtureRow, ...]
    required_route_kinds: tuple[str, ...] = (
        "promotion-target",
        "criteria",
        "persistence",
        "refusal",
        "no-mesh-fallback",
    )
    diagnostics: tuple[SurfaceSampledImplicitPromotionDiagnostic, ...] = ()

    @property
    def passed(self) -> bool:
        return bool(self.rows) and not self.diagnostics and all(row.passed for row in self.rows)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "required_route_kinds": self.required_route_kinds,
            "rows": [row.canonical_payload() for row in self.rows],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class SurfaceNonCSGReplacementRecord:
    """Deliberate replacement workflow for a request that should not be treated as CSG."""

    workflow: Literal["author-new-surface", "use-loft-or-sweep", "edit-source-samples", "promote-and-retry"]
    reason: str
    target_family: SurfaceSampledImplicitPromotionTargetFamily | None = None
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "workflow": self.workflow,
            "reason": self.reason,
            "target_family": self.target_family,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class SurfaceRepresentationRefusalRecord:
    """Supported refusal contract for impossible sampled/implicit CSG representations."""

    operation: SurfaceBooleanOperation
    source_families: tuple[str, str]
    reason_code: Literal[
        "heightmap-overhang",
        "displacement-source-mismatch",
        "unsafe-implicit-field",
        "non-csg-replacement",
        "missing-solver-code",
    ]
    message: str
    supported_refusal: bool
    replacement: SurfaceNonCSGReplacementRecord | None = None
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "source_families": self.source_families,
            "reason_code": self.reason_code,
            "message": self.message,
            "supported_refusal": self.supported_refusal,
            "replacement": None if self.replacement is None else self.replacement.canonical_payload(),
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitReferenceFixtureRow:
    """Reference fixture promotion row for sampled/implicit CSG evidence."""

    fixture_id: str
    route_kind: Literal["native", "promoted", "refusal", "unsafe", "malformed"]
    payload_kind: str
    passed: bool
    reference_state: Literal["clean", "dirty", "missing"] = "clean"
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "route_kind": self.route_kind,
            "payload_kind": self.payload_kind,
            "passed": self.passed,
            "reference_state": self.reference_state,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class SurfaceSampledImplicitReferenceFixturePromotionReport:
    """Promoted reference fixture set for sampled/implicit native, promoted, and refusal routes."""

    rows: tuple[SurfaceSampledImplicitReferenceFixtureRow, ...]
    required_route_kinds: tuple[str, ...] = ("native", "promoted", "refusal", "unsafe", "malformed")
    diagnostics: tuple[SurfaceSampledImplicitPromotionDiagnostic, ...] = ()

    @property
    def passed(self) -> bool:
        return bool(self.rows) and not self.diagnostics and all(row.passed for row in self.rows)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "required_route_kinds": self.required_route_kinds,
            "rows": [row.canonical_payload() for row in self.rows],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


_SURFACE_BOOLEAN_EXECUTABLE_FAMILY_PAIRS: frozenset[tuple[str, str]] = frozenset(
    (
        {
        tuple(sorted((left_family, right_family)))
        for left_family, left_capability in PATCH_FAMILY_CAPABILITY_MATRIX.items()
        for right_family, right_capability in PATCH_FAMILY_CAPABILITY_MATRIX.items()
        if left_capability.support_phase == "available" and right_capability.support_phase == "available"
        and left_family in {"planar", "ruled", "revolution"}
        and right_family in {"planar", "ruled", "revolution", "bspline", "nurbs"}
        }
        | {
            tuple(sorted((left_family, right_family)))
            for left_family, left_capability in PATCH_FAMILY_CAPABILITY_MATRIX.items()
            for right_family, right_capability in PATCH_FAMILY_CAPABILITY_MATRIX.items()
            if left_capability.support_phase == "available"
            and right_capability.support_phase == "available"
            and left_family in {"bspline", "nurbs"}
            and right_family in {"bspline", "nurbs"}
        }
        | {
            tuple(sorted((left_family, right_family)))
            for left_family, left_capability in PATCH_FAMILY_CAPABILITY_MATRIX.items()
            for right_family, right_capability in PATCH_FAMILY_CAPABILITY_MATRIX.items()
            if left_capability.support_phase == "available"
            and right_capability.support_phase == "available"
            and "sweep" in {left_family, right_family}
            and {left_family, right_family}
            <= {"planar", "ruled", "revolution", "bspline", "nurbs", "sweep", "subdivision"}
        }
        | {
            tuple(sorted((left_family, right_family)))
            for left_family, left_capability in PATCH_FAMILY_CAPABILITY_MATRIX.items()
            for right_family, right_capability in PATCH_FAMILY_CAPABILITY_MATRIX.items()
            if left_capability.support_phase == "available"
            and right_capability.support_phase == "available"
            and "subdivision" in {left_family, right_family}
            and {left_family, right_family}
            <= {"planar", "ruled", "revolution", "bspline", "nurbs", "sweep", "subdivision"}
        }
    )
)
ANALYTIC_SURFACE_CSG_FAMILIES: frozenset[str] = frozenset({"planar", "ruled", "revolution"})
SAMPLED_SURFACE_CSG_FAMILIES: frozenset[str] = frozenset({"implicit", "heightmap", "displacement"})
PARAMETRIC_HIGHER_ORDER_SURFACE_CSG_FAMILIES: frozenset[str] = frozenset(
    {"bspline", "nurbs", "sweep", "subdivision"}
)
HIGHER_ORDER_SURFACE_CSG_FAMILIES: frozenset[str] = frozenset(
    {"bspline", "nurbs", "sweep", "subdivision", "implicit", "heightmap", "displacement", "torus"}
)

SURFACE_CSG_ROUTE_ID_BY_PAIR_CLASS: dict[SurfaceCSGRoutePairClass, str] = {
    "low-order-analytic": "surface-csg.low-order-analytic.exact",
    "analytic-to-bspline": "surface-csg.analytic-to-bspline.declared-tolerance",
    "analytic-to-nurbs": "surface-csg.analytic-to-nurbs.declared-tolerance",
    "analytic-to-sweep": "surface-csg.analytic-to-sweep.declared-tolerance",
    "analytic-to-subdivision": "surface-csg.analytic-to-subdivision.declared-tolerance",
    "spline-nurbs-pair": "surface-csg.spline-nurbs-pair.declared-tolerance",
    "sweep-pair": "surface-csg.sweep-pair.declared-tolerance",
    "subdivision-pair": "surface-csg.subdivision-pair.declared-tolerance",
    "sampled-boundary": "surface-csg.sampled-boundary.refusal",
    "unsupported-family": "surface-csg.unsupported-family.refusal",
}


def classify_surface_csg_route_pair_class(left_family: str, right_family: str) -> SurfaceCSGRoutePairClass:
    """Classify one CSG family pair into the route taxonomy used by the registry."""

    known_families = set(PATCH_FAMILY_CAPABILITY_MATRIX)
    if left_family not in known_families or right_family not in known_families:
        return "unsupported-family"
    if left_family in SAMPLED_SURFACE_CSG_FAMILIES or right_family in SAMPLED_SURFACE_CSG_FAMILIES:
        return "sampled-boundary"
    if left_family in ANALYTIC_SURFACE_CSG_FAMILIES and right_family in ANALYTIC_SURFACE_CSG_FAMILIES:
        return "low-order-analytic"
    families = {left_family, right_family}
    analytic_pair = left_family in ANALYTIC_SURFACE_CSG_FAMILIES or right_family in ANALYTIC_SURFACE_CSG_FAMILIES
    if analytic_pair and "bspline" in families:
        return "analytic-to-bspline"
    if analytic_pair and "nurbs" in families:
        return "analytic-to-nurbs"
    if analytic_pair and "sweep" in families:
        return "analytic-to-sweep"
    if analytic_pair and "subdivision" in families:
        return "analytic-to-subdivision"
    if families.issubset({"bspline", "nurbs"}):
        return "spline-nurbs-pair"
    if "sweep" in families:
        return "sweep-pair"
    if "subdivision" in families:
        return "subdivision-pair"
    return "unsupported-family"


def surface_csg_route_lookup(
    operation: SurfaceBooleanOperation,
    left_family: str,
    right_family: str,
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
) -> SurfaceCSGRouteRegistryRow:
    """Return the route-taxonomy row for one CSG operation/family pair."""

    source = registry if registry is not None else SURFACE_CSG_SOLVER_REGISTRY
    support = source.support_for(operation, left_family, right_family)
    pair_class = classify_surface_csg_route_pair_class(left_family, right_family)
    route_id = SURFACE_CSG_ROUTE_ID_BY_PAIR_CLASS[pair_class]
    diagnostic: SurfaceCSGRouteSupportDiagnostic | None = None
    if pair_class == "unsupported-family":
        diagnostic = SurfaceCSGRouteSupportDiagnostic(
            code="unknown-family",
            operation=operation,
            left_family=left_family,
            right_family=right_family,
            pair_class=pair_class,
            message=f"Surface CSG route registry does not know family pair {left_family}/{right_family}.",
        )
    elif not support.supported:
        diagnostic = SurfaceCSGRouteSupportDiagnostic(
            code="non-executable-route",
            operation=operation,
            left_family=left_family,
            right_family=right_family,
            pair_class=pair_class,
            message=(
                f"Surface CSG route {route_id} for {operation} {left_family}/{right_family} "
                f"is registered but not executable: {support.required_future_capability}."
            ),
        )
    return SurfaceCSGRouteRegistryRow(
        operation=operation,
        left_family=left_family,
        right_family=right_family,
        pair_class=pair_class,
        route_id=route_id,
        supported=support.supported,
        support_state=support.support_state,
        phase=support.phase,
        required_future_capability=support.required_future_capability,
        diagnostic=diagnostic,
    )


def surface_csg_executable_row_report(
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
    families: Iterable[str] | None = None,
    operations: Iterable[SurfaceBooleanOperation] = SURFACE_BOOLEAN_OPERATIONS,
) -> SurfaceCSGExecutableRowReport:
    """Return route-taxonomy rows and diagnostics for requested CSG matrix cells."""

    family_keys = tuple(sorted(PATCH_FAMILY_CAPABILITY_MATRIX if families is None else tuple(families)))
    rows = tuple(
        surface_csg_route_lookup(operation, left_family, right_family, registry=registry)
        for operation in operations
        for left_family in family_keys
        for right_family in family_keys
    )
    return SurfaceCSGExecutableRowReport(rows=rows)


HIGHER_ORDER_CSG_FIXTURE_PAIR_CLASSES: tuple[SurfaceCSGRoutePairClass, ...] = (
    "analytic-to-bspline",
    "analytic-to-nurbs",
    "analytic-to-sweep",
    "analytic-to-subdivision",
    "spline-nurbs-pair",
    "sweep-pair",
    "subdivision-pair",
)


def enumerate_higher_order_csg_pair_fixture_rows(
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
    operations: Iterable[SurfaceBooleanOperation] = SURFACE_BOOLEAN_OPERATIONS,
    required_pair_classes: Iterable[SurfaceCSGRoutePairClass] = HIGHER_ORDER_CSG_FIXTURE_PAIR_CLASSES,
) -> tuple[SurfaceCSGPairFixtureRow, ...]:
    """Enumerate bounded higher-order CSG fixture rows from the route registry."""

    pair_classes = tuple(required_pair_classes)
    rows: list[SurfaceCSGPairFixtureRow] = []
    family_keys = tuple(sorted(PATCH_FAMILY_CAPABILITY_MATRIX))
    for operation in operations:
        for left_family in family_keys:
            for right_family in family_keys:
                route = surface_csg_route_lookup(operation, left_family, right_family, registry=registry)
                if route.pair_class not in pair_classes:
                    continue
                if not route.executable:
                    rows.append(
                        SurfaceCSGPairFixtureRow(
                            operation=operation,
                            left_family=left_family,
                            right_family=right_family,
                            pair_class=route.pair_class,
                            route_id=route.route_id,
                            executable=False,
                            expected_category="refusal",
                            diagnostic="" if route.diagnostic is None else route.diagnostic.message,
                        )
                    )
                    continue
                rows.append(
                    SurfaceCSGPairFixtureRow(
                        operation=operation,
                        left_family=left_family,
                        right_family=right_family,
                        pair_class=route.pair_class,
                        route_id=route.route_id,
                        executable=True,
                        expected_category=_surface_csg_fixture_category(route.pair_class),
                    )
                )
    return tuple(rows)


def verify_higher_order_csg_pair_fixture_matrix(
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
    required_pair_classes: Iterable[SurfaceCSGRoutePairClass] = HIGHER_ORDER_CSG_FIXTURE_PAIR_CLASSES,
) -> SurfaceCSGPairFixtureEvidenceReport:
    """Verify fixture evidence for promoted higher-order CSG pair classes."""

    required = tuple(required_pair_classes)
    rows = enumerate_higher_order_csg_pair_fixture_rows(registry=registry, required_pair_classes=required)
    diagnostics: list[SurfaceCSGRouteSupportDiagnostic] = []
    by_class = {pair_class: [row for row in rows if row.pair_class == pair_class] for pair_class in required}
    for pair_class, class_rows in by_class.items():
        if not class_rows:
            diagnostics.append(
                SurfaceCSGRouteSupportDiagnostic(
                    code="missing-route",
                    operation="union",
                    left_family="fixture-matrix",
                    right_family="fixture-matrix",
                    pair_class=pair_class,
                    message=f"Higher-order CSG fixture matrix has no rows for pair class {pair_class}.",
                )
            )
            continue
        if not any(row.executable for row in class_rows):
            diagnostics.append(
                SurfaceCSGRouteSupportDiagnostic(
                    code="non-executable-route",
                    operation=class_rows[0].operation,
                    left_family=class_rows[0].left_family,
                    right_family=class_rows[0].right_family,
                    pair_class=pair_class,
                    message=f"Higher-order CSG fixture matrix has no executable rows for pair class {pair_class}.",
                )
            )
    for row in rows:
        if row.mesh_fallback_attempted:
            diagnostics.append(
                SurfaceCSGRouteSupportDiagnostic(
                    code="non-executable-route",
                    operation=row.operation,
                    left_family=row.left_family,
                    right_family=row.right_family,
                    pair_class=row.pair_class,
                    message=f"Higher-order CSG fixture row {row.route_id} attempted mesh fallback.",
                )
            )
    return SurfaceCSGPairFixtureEvidenceReport(
        rows=rows,
        required_pair_classes=required,
        diagnostics=tuple(diagnostics),
    )


def enumerate_implicit_csg_fixture_rows() -> tuple[SurfaceImplicitCSGFixtureRow, ...]:
    """Build deterministic implicit CSG success/refusal/persistence evidence rows."""

    rows: list[SurfaceImplicitCSGFixtureRow] = []
    left = adapt_surface_patch_to_implicit_field(
        PlanarSurfacePatch(family="planar"),
        bounds=(-1.0, 1.0, -1.0, 1.0, -0.1, 0.1),
    )
    right = adapt_surface_patch_to_implicit_field(
        PlanarSurfacePatch(family="planar", origin=np.array([0.0, 0.0, 0.5], dtype=float)),
        bounds=(-1.0, 1.0, -1.0, 1.0, 0.4, 0.6),
    )
    success = compose_implicit_field_csg_result("union", (left, right), samples=(3, 3, 3), max_sample_count=27)
    rows.append(
        SurfaceImplicitCSGFixtureRow(
            fixture_id="implicit-csg/success-union",
            route_kind="success",
            passed=success.supported and success.patch is not None and success.patch.field.kind == "union",
            operation="union",
            left_family=left.family,
            right_family=right.family,
            message="Implicit CSG union produced a surface-native implicit result.",
        )
    )
    unsafe = compose_implicit_field_csg_result("union", (left, right), samples=(4, 4, 4), max_sample_count=4)
    rows.append(
        SurfaceImplicitCSGFixtureRow(
            fixture_id="implicit-csg/unsafe-budget-refusal",
            route_kind="unsafe-refusal",
            passed=not unsafe.supported and bool(unsafe.diagnostics) and unsafe.diagnostics[0].code == "unsafe-result",
            operation="union",
            left_family=left.family,
            right_family=right.family,
            message="" if not unsafe.diagnostics else unsafe.diagnostics[0].message,
        )
    )
    refused_adapter = ImplicitOperandFieldAdapterRecord(
        family="unsupported-field",
        patch_id="unsupported-fixture",
        adapter_kind="refused",
        supported=False,
        diagnostics=(
            ImplicitOperandFieldAdapterRefusalDiagnostic(
                code="unsupported-family",
                message="No field adapter exists; no mesh fallback was attempted.",
                family="unsupported-field",
                patch_id="unsupported-fixture",
            ),
        ),
    )
    adapter_refusal = compose_implicit_field_csg_result("intersection", (left, refused_adapter))
    rows.append(
        SurfaceImplicitCSGFixtureRow(
            fixture_id="implicit-csg/adapter-refusal",
            route_kind="adapter-refusal",
            passed=not adapter_refusal.supported and bool(adapter_refusal.diagnostics),
            operation="intersection",
            left_family=left.family,
            right_family=refused_adapter.family,
            message="" if not adapter_refusal.diagnostics else adapter_refusal.diagnostics[0].message,
        )
    )
    if success.body is None:
        persistence_passed = False
        persistence_message = "Implicit CSG success fixture produced no body."
    else:
        from impression.io import verify_implicit_csg_impress_round_trip

        persistence = verify_implicit_csg_impress_round_trip(success.body)
        persistence_passed = persistence.supported
        persistence_message = persistence.message
    rows.append(
        SurfaceImplicitCSGFixtureRow(
            fixture_id="implicit-csg/impress-persistence",
            route_kind="persistence",
            passed=persistence_passed,
            operation="union",
            left_family=left.family,
            right_family=right.family,
            message=persistence_message,
        )
    )
    no_mesh_passed = all(not row.mesh_fallback_attempted for row in rows) and any(
        "mesh fallback" in row.message.lower() for row in rows
    )
    rows.append(
        SurfaceImplicitCSGFixtureRow(
            fixture_id="implicit-csg/no-mesh-fallback",
            route_kind="no-mesh-fallback",
            passed=no_mesh_passed,
            operation="union",
            left_family="implicit",
            right_family="implicit",
            message="Implicit CSG fixture matrix contains no mesh fallback attempts.",
            mesh_fallback_attempted=False,
        )
    )
    return tuple(rows)


def verify_implicit_csg_fixture_evidence_matrix() -> SurfaceImplicitCSGEvidenceReport:
    """Return a clean evidence report for implicit-preserving CSG routes."""

    rows = enumerate_implicit_csg_fixture_rows()
    diagnostics: list[SurfaceCSGRouteSupportDiagnostic] = []
    by_kind = {kind: [row for row in rows if row.route_kind == kind] for kind in SurfaceImplicitCSGEvidenceReport(()).required_route_kinds}
    for route_kind, kind_rows in by_kind.items():
        if not kind_rows:
            diagnostics.append(
                SurfaceCSGRouteSupportDiagnostic(
                    code="missing-route",
                    operation="union",
                    left_family="implicit",
                    right_family="implicit",
                    pair_class="sampled-boundary",
                    message=f"Implicit CSG evidence matrix is missing route kind {route_kind}.",
                )
            )
        for row in kind_rows:
            if not row.passed or row.mesh_fallback_attempted or row.reference_state != "clean":
                diagnostics.append(
                    SurfaceCSGRouteSupportDiagnostic(
                        code="non-executable-route",
                        operation=row.operation,
                        left_family=row.left_family,
                        right_family=row.right_family,
                        pair_class="sampled-boundary",
                        message=f"Implicit CSG fixture {row.fixture_id} is not clean evidence: {row.message}",
                    )
                )
    return SurfaceImplicitCSGEvidenceReport(rows=rows, diagnostics=tuple(diagnostics))


def enumerate_heightmap_csg_fixture_rows() -> tuple[SurfaceHeightmapCSGFixtureRow, ...]:
    """Build deterministic heightmap CSG success/refusal/promotion/persistence evidence rows."""

    rows: list[SurfaceHeightmapCSGFixtureRow] = []
    left = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float),
        alpha_mask=np.ones((2, 2), dtype=bool),
    )
    right = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.asarray([[1.0, 0.5], [1.5, 4.0]], dtype=float),
        alpha_mask=np.ones((2, 2), dtype=bool),
    )
    success = compose_heightmap_csg_result("union", left, right)
    rows.append(
        SurfaceHeightmapCSGFixtureRow(
            fixture_id="heightmap-csg/success-union",
            route_kind="success",
            passed=success.supported and success.patch is not None and success.operation_record.resample_kernel == "none",
            operation="union",
            message="Heightmap CSG union produced a surface-native heightmap result.",
        )
    )

    overhang_transform = np.eye(4, dtype=float)
    overhang_transform[0, 2] = 0.25
    overhang = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((2, 2), dtype=float),
        transform_matrix=overhang_transform,
    )
    refused = compose_heightmap_csg_result("intersection", overhang, right)
    rows.append(
        SurfaceHeightmapCSGFixtureRow(
            fixture_id="heightmap-csg/representability-refusal",
            route_kind="representability-refusal",
            passed=not refused.supported and bool(refused.diagnostics) and refused.diagnostics[0].code == "representability-refusal",
            operation="intersection",
            message="" if not refused.diagnostics else refused.diagnostics[0].message,
        )
    )

    promotion = plan_heightmap_promotion_route("union", overhang, right)
    rows.append(
        SurfaceHeightmapCSGFixtureRow(
            fixture_id="heightmap-csg/promotion-implicit",
            route_kind="promotion",
            passed=promotion.supported and promotion.target_family == "implicit",
            operation="union",
            target_family=promotion.target_family,
            message="Heightmap CSG overhang promotion selected an implicit target.",
        )
    )

    if success.body is None:
        persistence_passed = False
        persistence_message = "Heightmap CSG success fixture produced no body."
    else:
        from impression.io import verify_heightmap_csg_impress_round_trip

        persistence = verify_heightmap_csg_impress_round_trip(success.body)
        persistence_passed = persistence.supported
        persistence_message = persistence.message
    rows.append(
        SurfaceHeightmapCSGFixtureRow(
            fixture_id="heightmap-csg/impress-persistence",
            route_kind="persistence",
            passed=persistence_passed,
            operation="union",
            message=persistence_message,
        )
    )

    no_mesh_passed = all(not row.mesh_fallback_attempted for row in rows) and any(
        "mesh fallback" in row.message.lower() for row in rows
    )
    rows.append(
        SurfaceHeightmapCSGFixtureRow(
            fixture_id="heightmap-csg/no-mesh-fallback",
            route_kind="no-mesh-fallback",
            passed=no_mesh_passed,
            operation="union",
            message="Heightmap CSG fixture matrix contains no mesh fallback attempts.",
            mesh_fallback_attempted=False,
        )
    )
    return tuple(rows)


def verify_heightmap_csg_fixture_evidence_matrix() -> SurfaceHeightmapCSGEvidenceReport:
    """Return a clean evidence report for heightmap CSG routes."""

    rows = enumerate_heightmap_csg_fixture_rows()
    diagnostics: list[SurfaceCSGRouteSupportDiagnostic] = []
    required = SurfaceHeightmapCSGEvidenceReport(()).required_route_kinds
    by_kind = {kind: [row for row in rows if row.route_kind == kind] for kind in required}
    for route_kind, kind_rows in by_kind.items():
        if not kind_rows:
            diagnostics.append(
                SurfaceCSGRouteSupportDiagnostic(
                    code="missing-route",
                    operation="union",
                    left_family="heightmap",
                    right_family="heightmap",
                    pair_class="sampled-boundary",
                    message=f"Heightmap CSG evidence matrix is missing route kind {route_kind}.",
                )
            )
        for row in kind_rows:
            if not row.passed or row.mesh_fallback_attempted or row.reference_state != "clean":
                diagnostics.append(
                    SurfaceCSGRouteSupportDiagnostic(
                        code="non-executable-route",
                        operation=row.operation,
                        left_family="heightmap",
                        right_family="heightmap",
                        pair_class="sampled-boundary",
                        message=f"Heightmap CSG fixture {row.fixture_id} is not clean evidence: {row.message}",
                    )
                )
    return SurfaceHeightmapCSGEvidenceReport(rows=rows, diagnostics=tuple(diagnostics))


def _displacement_csg_fixture_source_patch(*, source: str = "shared") -> PlanarSurfacePatch:
    return PlanarSurfacePatch(family="planar", metadata={"fixture_source": source})


def enumerate_displacement_csg_fixture_rows() -> tuple[SurfaceDisplacementCSGFixtureRow, ...]:
    """Build deterministic displacement CSG success/refusal/promotion/persistence evidence rows."""

    rows: list[SurfaceDisplacementCSGFixtureRow] = []
    source = _displacement_csg_fixture_source_patch()
    left = DisplacementSurfacePatch(
        family="displacement",
        source_patch=source,
        displacement_samples=np.asarray([[0.0, 0.25], [0.5, 0.75]], dtype=float),
        alpha_mask=np.ones((2, 2), dtype=bool),
        projection_bounds=(-1.0, 1.0, -1.0, 1.0),
    )
    right = DisplacementSurfacePatch(
        family="displacement",
        source_patch=source,
        displacement_samples=np.asarray([[0.5, 0.1], [0.25, 1.0]], dtype=float),
        alpha_mask=np.ones((2, 2), dtype=bool),
        projection_bounds=(-1.0, 1.0, -1.0, 1.0),
    )
    success = compose_displacement_csg_result("union", left, right)
    rows.append(
        SurfaceDisplacementCSGFixtureRow(
            fixture_id="displacement-csg/success-union",
            route_kind="success",
            passed=success.supported and success.patch is not None and success.operation_record.resampling.resample_kernel == "none",
            operation="union",
            message="Displacement CSG union produced a surface-native displacement result.",
        )
    )

    other_source = _displacement_csg_fixture_source_patch(source="other")
    mismatch = DisplacementSurfacePatch(
        family="displacement",
        source_patch=other_source,
        displacement_samples=np.ones((2, 2), dtype=float),
        alpha_mask=np.ones((2, 2), dtype=bool),
        projection_bounds=(-1.0, 1.0, -1.0, 1.0),
    )
    refusal = displacement_source_mismatch_refusal_record("union", left, mismatch)
    rows.append(
        SurfaceDisplacementCSGFixtureRow(
            fixture_id="displacement-csg/source-mismatch-refusal",
            route_kind="source-refusal",
            passed=refusal.supported_refusal and refusal.reason_code == "source-mismatch",
            operation="union",
            message=refusal.message,
        )
    )

    promotion = plan_displacement_promotion_route("union", left, mismatch)
    rows.append(
        SurfaceDisplacementCSGFixtureRow(
            fixture_id="displacement-csg/promotion-implicit",
            route_kind="promotion",
            passed=promotion.supported and promotion.target_family == "implicit",
            operation="union",
            target_family=promotion.target_family,
            message="Displacement CSG source mismatch promotion selected an implicit target.",
        )
    )

    if success.body is None:
        persistence_passed = False
        persistence_message = "Displacement CSG success fixture produced no body."
    else:
        from impression.io import verify_displacement_csg_impress_round_trip

        persistence = verify_displacement_csg_impress_round_trip(success.body)
        persistence_passed = persistence.supported
        persistence_message = persistence.message
    rows.append(
        SurfaceDisplacementCSGFixtureRow(
            fixture_id="displacement-csg/impress-persistence",
            route_kind="persistence",
            passed=persistence_passed,
            operation="union",
            message=persistence_message,
        )
    )

    no_mesh_passed = all(not row.mesh_fallback_attempted for row in rows) and any(
        "mesh fallback" in row.message.lower() for row in rows
    )
    rows.append(
        SurfaceDisplacementCSGFixtureRow(
            fixture_id="displacement-csg/no-mesh-fallback",
            route_kind="no-mesh-fallback",
            passed=no_mesh_passed,
            operation="union",
            message="Displacement CSG fixture matrix contains no mesh fallback attempts.",
            mesh_fallback_attempted=False,
        )
    )
    return tuple(rows)


def verify_displacement_csg_fixture_evidence_matrix() -> SurfaceDisplacementCSGEvidenceReport:
    """Return a clean evidence report for displacement CSG routes."""

    rows = enumerate_displacement_csg_fixture_rows()
    diagnostics: list[SurfaceCSGRouteSupportDiagnostic] = []
    required = SurfaceDisplacementCSGEvidenceReport(()).required_route_kinds
    by_kind = {kind: [row for row in rows if row.route_kind == kind] for kind in required}
    for route_kind, kind_rows in by_kind.items():
        if not kind_rows:
            diagnostics.append(
                SurfaceCSGRouteSupportDiagnostic(
                    code="missing-route",
                    operation="union",
                    left_family="displacement",
                    right_family="displacement",
                    pair_class="sampled-boundary",
                    message=f"Displacement CSG evidence matrix is missing route kind {route_kind}.",
                )
            )
        for row in kind_rows:
            if not row.passed or row.mesh_fallback_attempted or row.reference_state != "clean":
                diagnostics.append(
                    SurfaceCSGRouteSupportDiagnostic(
                        code="non-executable-route",
                        operation=row.operation,
                        left_family="displacement",
                        right_family="displacement",
                        pair_class="sampled-boundary",
                        message=f"Displacement CSG fixture {row.fixture_id} is not clean evidence: {row.message}",
                    )
                )
    return SurfaceDisplacementCSGEvidenceReport(rows=rows, diagnostics=tuple(diagnostics))


def enumerate_sampled_implicit_csg_unsupported_rows(
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
    operations: Iterable[SurfaceBooleanOperation] = SURFACE_BOOLEAN_OPERATIONS,
    route_status: SurfaceSampledImplicitCSGRouteStatus = "in-progress",
) -> tuple[SurfaceSampledImplicitCSGUnsupportedRow, ...]:
    """Enumerate sampled/implicit CSG rows that still need route promotion."""

    source = registry if registry is not None else SURFACE_CSG_SOLVER_REGISTRY
    rows: list[SurfaceSampledImplicitCSGUnsupportedRow] = []
    for operation in operations:
        for left_family in source.families:
            for right_family in source.families:
                if left_family not in SAMPLED_SURFACE_CSG_FAMILIES and right_family not in SAMPLED_SURFACE_CSG_FAMILIES:
                    continue
                support = source.support_for(operation, left_family, right_family)
                if support.supported:
                    continue
                if support.support_state != "unsupported":
                    continue
                rows.append(
                    SurfaceSampledImplicitCSGUnsupportedRow(
                        operation=operation,
                        left_family=left_family,
                        right_family=right_family,
                        support_state=support.support_state,
                        route_status=route_status,
                        route_id=SURFACE_CSG_ROUTE_ID_BY_PAIR_CLASS["sampled-boundary"],
                        required_future_capability=support.required_future_capability or "",
                    )
                )
    return tuple(rows)


def verify_sampled_implicit_csg_unsupported_row_tracker(
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
    operations: Iterable[SurfaceBooleanOperation] = SURFACE_BOOLEAN_OPERATIONS,
) -> SurfaceSampledImplicitCSGUnsupportedRowReport:
    """Verify the current sampled/implicit CSG unsupported-row tracking set."""

    source = registry if registry is not None else SURFACE_CSG_SOLVER_REGISTRY
    operation_tuple = tuple(operations)
    sampled_pairs = tuple(
        (left_family, right_family)
        for left_family in source.families
        for right_family in source.families
        if left_family in SAMPLED_SURFACE_CSG_FAMILIES or right_family in SAMPLED_SURFACE_CSG_FAMILIES
    )
    expected_rows_per_operation = len(sampled_pairs)
    expected_row_count = expected_rows_per_operation * len(operation_tuple)
    rows = enumerate_sampled_implicit_csg_unsupported_rows(registry=source, operations=operation_tuple)
    diagnostics: list[SurfaceCSGRouteSupportDiagnostic] = []

    if len(rows) != expected_row_count:
        diagnostics.append(
            SurfaceCSGRouteSupportDiagnostic(
                code="missing-route",
                operation=operation_tuple[0] if operation_tuple else "union",
                left_family="sampled-implicit-tracker",
                right_family="sampled-implicit-tracker",
                pair_class="sampled-boundary",
                message=(
                    f"Sampled/implicit CSG unsupported-row tracker found {len(rows)} rows; "
                    f"expected {expected_row_count}."
                ),
            )
        )

    for operation in operation_tuple:
        operation_rows = tuple(row for row in rows if row.operation == operation)
        if len(operation_rows) != expected_rows_per_operation:
            diagnostics.append(
                SurfaceCSGRouteSupportDiagnostic(
                    code="missing-route",
                    operation=operation,
                    left_family="sampled-implicit-tracker",
                    right_family="sampled-implicit-tracker",
                    pair_class="sampled-boundary",
                    message=(
                        f"Sampled/implicit CSG unsupported-row tracker found {len(operation_rows)} "
                        f"rows for {operation}; expected {expected_rows_per_operation}."
                    ),
                )
            )

    valid_statuses: set[SurfaceSampledImplicitCSGRouteStatus] = {
        "in-progress",
        "native-route",
        "promotion-route",
        "representation-refusal",
        "non-csg-replacement",
    }
    for row in rows:
        if row.route_status not in valid_statuses:
            diagnostics.append(
                SurfaceCSGRouteSupportDiagnostic(
                    code="missing-route",
                    operation=row.operation,
                    left_family=row.left_family,
                    right_family=row.right_family,
                    pair_class="sampled-boundary",
                    message=f"Sampled/implicit CSG row has invalid route status {row.route_status!r}.",
                )
            )
        if not row.required_future_capability:
            diagnostics.append(
                SurfaceCSGRouteSupportDiagnostic(
                    code="missing-route",
                    operation=row.operation,
                    left_family=row.left_family,
                    right_family=row.right_family,
                    pair_class="sampled-boundary",
                    message="Sampled/implicit CSG row requires a future capability or route classification.",
                )
            )
        if row.mesh_fallback_attempted:
            diagnostics.append(
                SurfaceCSGRouteSupportDiagnostic(
                    code="non-executable-route",
                    operation=row.operation,
                    left_family=row.left_family,
                    right_family=row.right_family,
                    pair_class="sampled-boundary",
                    message="Sampled/implicit CSG row attempted mesh fallback.",
                )
            )

    return SurfaceSampledImplicitCSGUnsupportedRowReport(
        rows=rows,
        expected_row_count=expected_row_count,
        expected_rows_per_operation=expected_rows_per_operation,
        diagnostics=tuple(diagnostics),
    )


def _sampled_implicit_default_promotion_target(
    left_family: str,
    right_family: str,
) -> tuple[
    SurfaceSampledImplicitPromotionTargetFamily,
    SurfaceSampledImplicitCSGRouteStatus,
    Literal["lossless", "sampled-reconstruction", "volumetric-field", "exact-reconstruction"],
    str,
]:
    families = {left_family, right_family}
    if "implicit" in families:
        return (
            "implicit",
            "promotion-route",
            "volumetric-field",
            "Implicit participation preserves the CSG predicate as an implicit field route.",
        )
    if "nurbs" in families:
        return (
            "nurbs",
            "promotion-route",
            "exact-reconstruction",
            "Sampled route promotes to a NURBS reconstruction target for downstream exact-trim ownership.",
        )
    if "bspline" in families:
        return (
            "bspline",
            "promotion-route",
            "exact-reconstruction",
            "Sampled route promotes to a B-spline reconstruction target for downstream exact-trim ownership.",
        )
    if "subdivision" in families or "sweep" in families:
        return (
            "subdivision",
            "promotion-route",
            "sampled-reconstruction",
            "Sampled route promotes to subdivision because the paired family is reconstructed as bounded charts.",
        )
    if families <= {"heightmap"}:
        return (
            "subdivision",
            "promotion-route",
            "sampled-reconstruction",
            "Static sampled matrix routes heightmap-only rows through subdivision when a native heightmap route is unavailable.",
        )
    if families <= {"displacement"}:
        return (
            "subdivision",
            "promotion-route",
            "sampled-reconstruction",
            "Static sampled matrix routes displacement-only rows through subdivision when source identity cannot be proven statically.",
        )
    return (
        "subdivision",
        "promotion-route",
        "sampled-reconstruction",
        "Sampled route promotes to subdivision as the bounded reconstructed surface target.",
    )


def select_sampled_implicit_promotion_target(
    row: SurfaceSampledImplicitCSGUnsupportedRow,
    *,
    allowed_targets: Sequence[SurfaceSampledImplicitPromotionTargetFamily] = (
        "implicit",
        "subdivision",
        "nurbs",
        "bspline",
        "representation-refusal",
        "non-csg-replacement",
    ),
) -> SurfaceSampledImplicitPromotionDecision:
    """Select a final target route for one sampled/implicit CSG tracker row."""

    allowed = tuple(str(target) for target in allowed_targets)
    target, route_status, lossiness, reason = _sampled_implicit_default_promotion_target(
        row.left_family,
        row.right_family,
    )
    diagnostics: list[SurfaceSampledImplicitPromotionDiagnostic] = []
    supported = True
    if target not in allowed:
        supported = False
        route_status = "in-progress"
        diagnostics.append(
            SurfaceSampledImplicitPromotionDiagnostic(
                code="missing-target",
                operation=row.operation,
                left_family=row.left_family,
                right_family=row.right_family,
                message=(
                    f"Sampled/implicit CSG target {target} is not allowed for this promotion matrix; "
                    "no mesh fallback was attempted."
                ),
            )
        )
    policy_row = SurfaceSampledImplicitPromotionPolicyRow(
        operation=row.operation,
        left_family=row.left_family,
        right_family=row.right_family,
        source_support_state=row.support_state,
        route_status=route_status,
        target_family=target if supported else None,
        lossiness=lossiness,
        route_id=row.route_id,
        reason=reason,
        required_future_capability=row.required_future_capability,
        mesh_fallback_attempted=row.mesh_fallback_attempted,
    )
    if policy_row.mesh_fallback_attempted:
        supported = False
        diagnostics.append(
            SurfaceSampledImplicitPromotionDiagnostic(
                code="mesh-fallback",
                operation=row.operation,
                left_family=row.left_family,
                right_family=row.right_family,
                message="Sampled/implicit CSG promotion row attempted mesh fallback.",
            )
        )
    return SurfaceSampledImplicitPromotionDecision(
        supported=supported and policy_row.complete,
        row=policy_row,
        diagnostics=tuple(diagnostics),
    )


def build_sampled_implicit_promotion_matrix(
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
    operations: Iterable[SurfaceBooleanOperation] = SURFACE_BOOLEAN_OPERATIONS,
    allowed_targets: Sequence[SurfaceSampledImplicitPromotionTargetFamily] = (
        "implicit",
        "subdivision",
        "nurbs",
        "bspline",
        "representation-refusal",
        "non-csg-replacement",
    ),
) -> SurfaceSampledImplicitPromotionMatrixReport:
    """Build the final promotion matrix for sampled/implicit CSG rows."""

    source = registry if registry is not None else SURFACE_CSG_SOLVER_REGISTRY
    operation_tuple = tuple(operations)
    tracker = verify_sampled_implicit_csg_unsupported_row_tracker(registry=source, operations=operation_tuple)
    rows: list[SurfaceSampledImplicitPromotionPolicyRow] = []
    diagnostics: list[SurfaceSampledImplicitPromotionDiagnostic] = []
    for tracker_row in tracker.rows:
        decision = select_sampled_implicit_promotion_target(tracker_row, allowed_targets=allowed_targets)
        rows.append(decision.row)
        diagnostics.extend(decision.diagnostics)
        if not decision.supported and not decision.diagnostics:
            diagnostics.append(
                SurfaceSampledImplicitPromotionDiagnostic(
                    code="incomplete-route",
                    operation=tracker_row.operation,
                    left_family=tracker_row.left_family,
                    right_family=tracker_row.right_family,
                    message="Sampled/implicit CSG promotion row remains incomplete; no mesh fallback was attempted.",
                )
            )
    if len(rows) != tracker.expected_row_count:
        diagnostics.append(
            SurfaceSampledImplicitPromotionDiagnostic(
                code="incomplete-route",
                operation=operation_tuple[0] if operation_tuple else "union",
                left_family="sampled-implicit-promotion",
                right_family="sampled-implicit-promotion",
                message=(
                    f"Sampled/implicit CSG promotion matrix found {len(rows)} rows; "
                    f"expected {tracker.expected_row_count}."
                ),
            )
        )
    return SurfaceSampledImplicitPromotionMatrixReport(
        rows=tuple(rows),
        expected_row_count=tracker.expected_row_count,
        expected_rows_per_operation=tracker.expected_rows_per_operation,
        diagnostics=tuple(diagnostics),
    )


def verify_sampled_implicit_promotion_matrix(
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
    operations: Iterable[SurfaceBooleanOperation] = SURFACE_BOOLEAN_OPERATIONS,
    allowed_targets: Sequence[SurfaceSampledImplicitPromotionTargetFamily] = (
        "implicit",
        "subdivision",
        "nurbs",
        "bspline",
        "representation-refusal",
        "non-csg-replacement",
    ),
) -> SurfaceSampledImplicitPromotionMatrixReport:
    """Verify sampled/implicit CSG rows have final non-mesh route targets."""

    report = build_sampled_implicit_promotion_matrix(
        registry=registry,
        operations=operations,
        allowed_targets=allowed_targets,
    )
    diagnostics = list(report.diagnostics)
    for row in report.rows:
        if row.route_status == "in-progress" or row.target_family is None:
            diagnostics.append(
                SurfaceSampledImplicitPromotionDiagnostic(
                    code="incomplete-route",
                    operation=row.operation,
                    left_family=row.left_family,
                    right_family=row.right_family,
                    message="Sampled/implicit CSG row remains in-progress after promotion selection.",
                )
            )
        if row.mesh_fallback_attempted:
            diagnostics.append(
                SurfaceSampledImplicitPromotionDiagnostic(
                    code="mesh-fallback",
                    operation=row.operation,
                    left_family=row.left_family,
                    right_family=row.right_family,
                    message="Sampled/implicit CSG promotion matrix attempted mesh fallback.",
                )
            )
    return SurfaceSampledImplicitPromotionMatrixReport(
        rows=report.rows,
        expected_row_count=report.expected_row_count,
        expected_rows_per_operation=report.expected_rows_per_operation,
        diagnostics=tuple(diagnostics),
    )


def normalize_sampled_implicit_promotion_tolerance(tolerance: float) -> float:
    """Normalize promotion provenance tolerance and reject unsafe values."""

    normalized = float(tolerance)
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError("Sampled/implicit promotion tolerance must be finite and non-negative.")
    return normalized


def sampled_implicit_promotion_lossiness_record(
    row: SurfaceSampledImplicitPromotionPolicyRow,
    *,
    tolerance: float = 1e-9,
) -> SurfaceSampledImplicitPromotionLossinessRecord:
    """Return normalized lossiness metadata for a sampled/implicit promotion row."""

    if row.target_family is None:
        raise ValueError("Sampled/implicit promotion lossiness requires a selected target family.")
    normalized_tolerance = normalize_sampled_implicit_promotion_tolerance(tolerance)
    if row.target_family == "implicit":
        sampling_policy: Literal["none", "bounded-samples", "field-sampling", "chart-reconstruction"] = "field-sampling"
        reconstruction_kind: Literal["none", "implicit-field", "subdivision-chart", "nurbs-fit", "bspline-fit"] = "implicit-field"
    elif row.target_family == "subdivision":
        sampling_policy = "chart-reconstruction"
        reconstruction_kind = "subdivision-chart"
    elif row.target_family == "nurbs":
        sampling_policy = "bounded-samples"
        reconstruction_kind = "nurbs-fit"
    elif row.target_family == "bspline":
        sampling_policy = "bounded-samples"
        reconstruction_kind = "bspline-fit"
    else:
        sampling_policy = "none"
        reconstruction_kind = "none"
    return SurfaceSampledImplicitPromotionLossinessRecord(
        target_family=row.target_family,
        lossiness=row.lossiness,
        tolerance=normalized_tolerance,
        sampling_policy=sampling_policy,
        reconstruction_kind=reconstruction_kind,
    )


def build_sampled_implicit_promotion_provenance_record(
    row: SurfaceSampledImplicitPromotionPolicyRow,
    *,
    operand_ids: Sequence[str] | None = None,
    tolerance: float = 1e-9,
) -> SurfaceSampledImplicitPromotionProvenanceRecord:
    """Build durable sampled/implicit CSG promotion provenance for result metadata."""

    diagnostics: list[SurfaceSampledImplicitPromotionProvenanceDiagnostic] = []
    source_operand_ids: tuple[str, str]
    if operand_ids is None:
        source_operand_ids = (f"{row.left_family}:left", f"{row.right_family}:right")
    elif len(operand_ids) != 2 or any(not str(operand_id).strip() for operand_id in operand_ids):
        source_operand_ids = ("", "")
        diagnostics.append(
            SurfaceSampledImplicitPromotionProvenanceDiagnostic(
                code="invalid-operands",
                message="Sampled/implicit promotion provenance requires two non-empty source operand ids.",
            )
        )
    else:
        source_operand_ids = (str(operand_ids[0]).strip(), str(operand_ids[1]).strip())
    lossiness: SurfaceSampledImplicitPromotionLossinessRecord | None = None
    try:
        if row.complete:
            lossiness = sampled_implicit_promotion_lossiness_record(row, tolerance=tolerance)
        else:
            diagnostics.append(
                SurfaceSampledImplicitPromotionProvenanceDiagnostic(
                    code="incomplete-route",
                    message="Sampled/implicit promotion provenance cannot be supported for an incomplete route.",
                )
            )
            normalize_sampled_implicit_promotion_tolerance(tolerance)
    except ValueError as exc:
        diagnostics.append(
            SurfaceSampledImplicitPromotionProvenanceDiagnostic(
                code="invalid-tolerance",
                message=f"{exc}; no mesh fallback was attempted.",
            )
        )
    return SurfaceSampledImplicitPromotionProvenanceRecord(
        operation=row.operation,
        source_families=(row.left_family, row.right_family),
        source_operand_ids=source_operand_ids,
        route_status=row.route_status,
        route_id=row.route_id,
        target_family=row.target_family,
        lossiness=lossiness,
        source_support_state=row.source_support_state,
        required_future_capability=row.required_future_capability,
        diagnostics=tuple(diagnostics),
        no_mesh_fallback=not row.mesh_fallback_attempted,
    )


def sampled_implicit_promotion_metadata_payload(
    row: SurfaceSampledImplicitPromotionPolicyRow,
    *,
    operand_ids: Sequence[str] | None = None,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    """Return serializable promotion provenance metadata for a result patch or body."""

    return build_sampled_implicit_promotion_provenance_record(
        row,
        operand_ids=operand_ids,
        tolerance=tolerance,
    ).canonical_payload()


def sampled_implicit_reconstruction_criteria(
    target_family: SurfaceSampledImplicitPromotionTargetFamily,
) -> SurfaceSampledImplicitReconstructionCriteriaRecord:
    """Return default reconstruction criteria for a sampled/implicit promotion target."""

    if target_family == "implicit":
        return SurfaceSampledImplicitReconstructionCriteriaRecord(
            target_family=target_family,
            max_sample_count=1_000_000,
            max_residual=None,
            requires_exact_reconstruction=False,
        )
    if target_family == "subdivision":
        return SurfaceSampledImplicitReconstructionCriteriaRecord(
            target_family=target_family,
            max_sample_count=1_000_000,
            max_residual=1e-3,
            requires_exact_reconstruction=False,
        )
    if target_family == "nurbs":
        return SurfaceSampledImplicitReconstructionCriteriaRecord(
            target_family=target_family,
            max_sample_count=250_000,
            max_residual=1e-6,
            requires_exact_reconstruction=True,
        )
    if target_family == "bspline":
        return SurfaceSampledImplicitReconstructionCriteriaRecord(
            target_family=target_family,
            max_sample_count=250_000,
            max_residual=1e-6,
            requires_exact_reconstruction=True,
        )
    return SurfaceSampledImplicitReconstructionCriteriaRecord(
        target_family=target_family,
        max_sample_count=0,
        max_residual=0.0,
        requires_complete_provenance=True,
        requires_exact_reconstruction=True,
    )


def evaluate_sampled_implicit_reconstruction_feasibility(
    provenance: SurfaceSampledImplicitPromotionProvenanceRecord,
    *,
    estimated_sample_count: int = 0,
    residual: float | None = None,
    criteria: SurfaceSampledImplicitReconstructionCriteriaRecord | None = None,
) -> SurfaceSampledImplicitReconstructionFeasibilityReport:
    """Evaluate whether a promoted sampled/implicit target can be reconstructed."""

    target = provenance.target_family
    selected_criteria = criteria if criteria is not None else (
        sampled_implicit_reconstruction_criteria(target) if target is not None else None
    )
    diagnostics: list[SurfaceSampledImplicitReconstructionDiagnostic] = []
    sample_count = int(estimated_sample_count)
    residual_value = None if residual is None else float(residual)
    if selected_criteria is None or target is None:
        diagnostics.append(
            SurfaceSampledImplicitReconstructionDiagnostic(
                code="unsupported-target",
                message="Sampled/implicit reconstruction requires a selected target family; no mesh fallback was attempted.",
            )
        )
    elif target in {"representation-refusal", "non-csg-replacement"}:
        diagnostics.append(
            SurfaceSampledImplicitReconstructionDiagnostic(
                code="unsupported-target",
                message=f"Target {target} is not a reconstructable surface family; no mesh fallback was attempted.",
            )
        )
    if (
        (selected_criteria is None or selected_criteria.requires_complete_provenance)
        and not provenance.supported
    ):
        diagnostics.append(
            SurfaceSampledImplicitReconstructionDiagnostic(
                code="incomplete-provenance",
                message="Sampled/implicit reconstruction requires supported promotion provenance; no mesh fallback was attempted.",
            )
        )
    if selected_criteria is not None and sample_count > selected_criteria.max_sample_count:
        diagnostics.append(
            SurfaceSampledImplicitReconstructionDiagnostic(
                code="sample-budget-exceeded",
                message=(
                    f"Sampled/implicit reconstruction needs {sample_count} samples, "
                    f"above the {selected_criteria.max_sample_count} target budget; no mesh fallback was attempted."
                ),
            )
        )
    if (
        selected_criteria is not None
        and selected_criteria.max_residual is not None
        and residual_value is not None
        and residual_value > selected_criteria.max_residual
    ):
        diagnostics.append(
            SurfaceSampledImplicitReconstructionDiagnostic(
                code="residual-exceeded",
                message=(
                    f"Sampled/implicit reconstruction residual {residual_value} exceeds "
                    f"{selected_criteria.max_residual}; no mesh fallback was attempted."
                ),
            )
        )
    return SurfaceSampledImplicitReconstructionFeasibilityReport(
        target_family=target,
        supported=not diagnostics,
        criteria=selected_criteria,
        provenance=provenance,
        estimated_sample_count=sample_count,
        residual=residual_value,
        diagnostics=tuple(diagnostics),
    )


def build_sampled_implicit_reconstruction_refusal(
    report: SurfaceSampledImplicitReconstructionFeasibilityReport,
) -> tuple[SurfaceSampledImplicitReconstructionDiagnostic, ...]:
    """Return deterministic refusal diagnostics for an ineligible reconstruction target."""

    if report.supported:
        return ()
    return report.diagnostics


def _sampled_implicit_promotion_fixture_body(row: SurfaceSampledImplicitPromotionPolicyRow) -> SurfaceBody:
    provenance = build_sampled_implicit_promotion_provenance_record(row, operand_ids=("fixture-left", "fixture-right"))
    metadata = {"kernel": {"sampled_implicit_promotion": provenance.canonical_payload()}}
    if row.target_family == "implicit":
        patch: SurfacePatch = ImplicitSurfacePatch(
            family="implicit",
            field=ImplicitFieldNode("sphere"),
            bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        )
    elif row.target_family == "subdivision":
        patch = SubdivisionSurfacePatch(family="subdivision")
    elif row.target_family == "nurbs":
        patch = NURBSSurfacePatch(family="nurbs")
    elif row.target_family == "bspline":
        patch = BSplineSurfacePatch(family="bspline")
    else:
        patch = PlanarSurfacePatch(family="planar")
    return make_surface_body((make_surface_shell((patch,), connected=False),), metadata=metadata)


def enumerate_sampled_implicit_promotion_fixture_rows() -> tuple[SurfaceSampledImplicitPromotionFixtureRow, ...]:
    """Build clean evidence rows for sampled/implicit promotion routes."""

    rows: list[SurfaceSampledImplicitPromotionFixtureRow] = []
    matrix = verify_sampled_implicit_promotion_matrix(operations=("union",))
    for target_family in ("implicit", "subdivision", "nurbs", "bspline"):
        row = next(matrix_row for matrix_row in matrix.rows if matrix_row.target_family == target_family)
        provenance = build_sampled_implicit_promotion_provenance_record(row, operand_ids=("fixture-left", "fixture-right"))
        feasibility = evaluate_sampled_implicit_reconstruction_feasibility(
            provenance,
            estimated_sample_count=16,
            residual=0.0,
        )
        rows.append(
            SurfaceSampledImplicitPromotionFixtureRow(
                fixture_id=f"sampled-implicit-promotion/{target_family}-target",
                route_kind="promotion-target",
                passed=provenance.supported and provenance.target_family == target_family,
                target_family=target_family,
                message=f"Sampled/implicit promotion selected {target_family} without mesh source truth.",
            )
        )
        rows.append(
            SurfaceSampledImplicitPromotionFixtureRow(
                fixture_id=f"sampled-implicit-promotion/{target_family}-criteria",
                route_kind="criteria",
                passed=feasibility.supported,
                target_family=target_family,
                message=(
                    "Sampled/implicit promotion reconstruction criteria accepted the target."
                    if feasibility.supported
                    else "; ".join(diagnostic.message for diagnostic in feasibility.diagnostics)
                ),
            )
        )
        body = _sampled_implicit_promotion_fixture_body(row)
        from impression.io import verify_sampled_implicit_promotion_impress_round_trip

        persistence = verify_sampled_implicit_promotion_impress_round_trip(body)
        rows.append(
            SurfaceSampledImplicitPromotionFixtureRow(
                fixture_id=f"sampled-implicit-promotion/{target_family}-persistence",
                route_kind="persistence",
                passed=persistence.supported,
                target_family=target_family,
                message=persistence.message,
            )
        )

    refused = build_sampled_implicit_promotion_matrix(operations=("union",), allowed_targets=("subdivision",))
    rows.append(
        SurfaceSampledImplicitPromotionFixtureRow(
            fixture_id="sampled-implicit-promotion/missing-target-refusal",
            route_kind="refusal",
            passed=not refused.passed and any(diagnostic.code == "missing-target" for diagnostic in refused.diagnostics),
            target_family=None,
            message="Sampled/implicit promotion missing-target fixture refused deterministically without mesh fallback.",
        )
    )
    no_mesh_passed = all(not row.mesh_fallback_attempted for row in rows) and any(
        "mesh" in row.message.lower() for row in rows
    )
    rows.append(
        SurfaceSampledImplicitPromotionFixtureRow(
            fixture_id="sampled-implicit-promotion/no-mesh-fallback",
            route_kind="no-mesh-fallback",
            passed=no_mesh_passed,
            target_family=None,
            message="Sampled/implicit promotion fixture matrix contains no mesh fallback attempts.",
            mesh_fallback_attempted=False,
        )
    )
    return tuple(rows)


def verify_sampled_implicit_promotion_fixture_evidence_matrix() -> SurfaceSampledImplicitPromotionEvidenceReport:
    """Return clean promotion fixture evidence and reject dirty or mesh-backed rows."""

    rows = enumerate_sampled_implicit_promotion_fixture_rows()
    diagnostics: list[SurfaceSampledImplicitPromotionDiagnostic] = []
    required = SurfaceSampledImplicitPromotionEvidenceReport(()).required_route_kinds
    by_kind = {kind: [row for row in rows if row.route_kind == kind] for kind in required}
    for route_kind, kind_rows in by_kind.items():
        if not kind_rows:
            diagnostics.append(
                SurfaceSampledImplicitPromotionDiagnostic(
                    code="incomplete-route",
                    operation="union",
                    left_family="sampled-implicit-promotion",
                    right_family="sampled-implicit-promotion",
                    message=f"Sampled/implicit promotion evidence is missing route kind {route_kind}.",
                )
            )
        for row in kind_rows:
            if row.mesh_fallback_attempted:
                diagnostics.append(
                    SurfaceSampledImplicitPromotionDiagnostic(
                        code="mesh-fallback",
                        operation="union",
                        left_family=str(row.target_family or "promotion"),
                        right_family=str(row.target_family or "promotion"),
                        message=f"Sampled/implicit promotion fixture {row.fixture_id} attempted mesh fallback.",
                    )
                )
            if not row.passed or row.reference_state != "clean":
                diagnostics.append(
                    SurfaceSampledImplicitPromotionDiagnostic(
                        code="incomplete-route",
                        operation="union",
                        left_family=str(row.target_family or "promotion"),
                        right_family=str(row.target_family or "promotion"),
                        message=f"Sampled/implicit promotion fixture {row.fixture_id} is not clean evidence: {row.message}",
                    )
                )
    return SurfaceSampledImplicitPromotionEvidenceReport(rows=rows, diagnostics=tuple(diagnostics))


def suggest_non_csg_replacement_workflow(
    intent: Literal["authoring-edit", "profile-transition", "sample-edit", "retry-promotion"],
    *,
    target_family: SurfaceSampledImplicitPromotionTargetFamily | None = None,
    reason: str = "",
) -> SurfaceNonCSGReplacementRecord:
    """Return a deliberate non-CSG replacement workflow hint."""

    workflow: Literal["author-new-surface", "use-loft-or-sweep", "edit-source-samples", "promote-and-retry"]
    if intent == "profile-transition":
        workflow = "use-loft-or-sweep"
    elif intent == "sample-edit":
        workflow = "edit-source-samples"
    elif intent == "retry-promotion":
        workflow = "promote-and-retry"
    else:
        workflow = "author-new-surface"
    return SurfaceNonCSGReplacementRecord(
        workflow=workflow,
        target_family=target_family,
        reason=reason or "Request is better represented as an authored modeling operation than as CSG.",
    )


def representation_refusal_from_heightmap_report(
    report: HeightmapRepresentabilityReport,
) -> SurfaceRepresentationRefusalRecord:
    """Convert a heightmap representability failure into a supported refusal record."""

    if report.representable:
        return SurfaceRepresentationRefusalRecord(
            operation=report.operation,
            source_families=("heightmap", "heightmap"),
            reason_code="missing-solver-code",
            message="Heightmap report is representable; this is not a representation refusal.",
            supported_refusal=False,
        )
    reason = report.diagnostics[0].message if report.diagnostics else "Heightmap CSG is not representable."
    return SurfaceRepresentationRefusalRecord(
        operation=report.operation,
        source_families=("heightmap", "heightmap"),
        reason_code="heightmap-overhang",
        message=f"{reason} No mesh fallback was attempted.",
        supported_refusal=True,
        replacement=suggest_non_csg_replacement_workflow(
            "retry-promotion",
            target_family="implicit",
            reason="Promote overhanging heightmap CSG to an implicit route.",
        ),
    )


def representation_refusal_from_displacement_refusal(
    refusal: DisplacementSourceMismatchRefusalRecord,
) -> SurfaceRepresentationRefusalRecord:
    """Convert displacement source/domain incompatibility into a supported refusal record."""

    return SurfaceRepresentationRefusalRecord(
        operation=refusal.operation,
        source_families=("displacement", "displacement"),
        reason_code="displacement-source-mismatch",
        message=f"{refusal.message} No mesh fallback was attempted.",
        supported_refusal=refusal.supported_refusal,
        replacement=suggest_non_csg_replacement_workflow(
            "retry-promotion",
            target_family="implicit" if refusal.replacement_hint == "promote-to-implicit" else "subdivision",
            reason="Use the declared displacement promotion route instead of mesh CSG.",
        ),
    )


def representation_refusal_from_implicit_result(
    result: ImplicitCompositionResult,
) -> SurfaceRepresentationRefusalRecord:
    """Convert unsafe implicit composition into a supported refusal record."""

    unsafe = not result.supported and any(diagnostic.code == "unsafe-result" for diagnostic in result.diagnostics)
    message = result.diagnostics[0].message if result.diagnostics else "Implicit CSG result is unsafe."
    return SurfaceRepresentationRefusalRecord(
        operation=result.operation,
        source_families=("implicit", "implicit"),
        reason_code="unsafe-implicit-field",
        message=f"{message} No mesh fallback was attempted.",
        supported_refusal=unsafe,
        replacement=suggest_non_csg_replacement_workflow(
            "retry-promotion",
            target_family="implicit",
            reason="Adjust field safety or sampling bounds and retry implicit composition.",
        ),
    )


def classify_sampled_implicit_representation_refusal(
    operation: SurfaceBooleanOperation,
    source_families: tuple[str, str],
    *,
    reason_code: Literal["non-csg-replacement", "missing-solver-code"],
    message: str,
    replacement: SurfaceNonCSGReplacementRecord | None = None,
) -> SurfaceRepresentationRefusalRecord:
    """Classify shared representation refusal or deliberate non-CSG replacement states."""

    if reason_code == "missing-solver-code":
        return SurfaceRepresentationRefusalRecord(
            operation=operation,
            source_families=source_families,
            reason_code=reason_code,
            message=f"{message} Missing solver code is not a representation refusal.",
            supported_refusal=False,
        )
    return SurfaceRepresentationRefusalRecord(
        operation=operation,
        source_families=source_families,
        reason_code=reason_code,
        message=f"{message} No mesh fallback was attempted.",
        supported_refusal=True,
        replacement=replacement
        or suggest_non_csg_replacement_workflow(
            "authoring-edit",
            reason="Use a deliberate authored workflow instead of treating this request as CSG.",
        ),
    )


def enumerate_sampled_implicit_reference_fixture_promotions() -> tuple[SurfaceSampledImplicitReferenceFixtureRow, ...]:
    """Return promoted clean reference fixtures for sampled/implicit CSG route outcomes."""

    rows: list[SurfaceSampledImplicitReferenceFixtureRow] = []
    native_sources = (
        ("implicit-csg", verify_implicit_csg_fixture_evidence_matrix()),
        ("heightmap-csg", verify_heightmap_csg_fixture_evidence_matrix()),
        ("displacement-csg", verify_displacement_csg_fixture_evidence_matrix()),
    )
    for payload_kind, report in native_sources:
        rows.append(
            SurfaceSampledImplicitReferenceFixtureRow(
                fixture_id=f"sampled-implicit-reference/{payload_kind}",
                route_kind="native",
                payload_kind=payload_kind,
                passed=report.passed,
                reference_state="clean" if report.passed else "dirty",
            )
        )
    promotion = verify_sampled_implicit_promotion_fixture_evidence_matrix()
    rows.append(
        SurfaceSampledImplicitReferenceFixtureRow(
            fixture_id="sampled-implicit-reference/promotion",
            route_kind="promoted",
            payload_kind="sampled-implicit-promotion",
            passed=promotion.passed,
            reference_state="clean" if promotion.passed else "dirty",
        )
    )
    refusal = classify_sampled_implicit_representation_refusal(
        "union",
        ("heightmap", "displacement"),
        reason_code="non-csg-replacement",
        message="Reference fixture captures a deliberate non-CSG replacement route.",
    )
    rows.append(
        SurfaceSampledImplicitReferenceFixtureRow(
            fixture_id="sampled-implicit-reference/refusal",
            route_kind="refusal",
            payload_kind="sampled-implicit-representation-refusal",
            passed=refusal.supported_refusal,
            reference_state="clean" if refusal.supported_refusal else "dirty",
        )
    )
    unsafe = compose_implicit_field_csg_result(
        "union",
        (
            adapt_surface_patch_to_implicit_field(PlanarSurfacePatch(family="planar")),
            adapt_surface_patch_to_implicit_field(PlanarSurfacePatch(family="planar")),
        ),
        samples=(4, 4, 4),
        max_sample_count=1,
    )
    unsafe_refusal = representation_refusal_from_implicit_result(unsafe)
    rows.append(
        SurfaceSampledImplicitReferenceFixtureRow(
            fixture_id="sampled-implicit-reference/unsafe-implicit",
            route_kind="unsafe",
            payload_kind="sampled-implicit-representation-refusal",
            passed=unsafe_refusal.supported_refusal,
            reference_state="clean" if unsafe_refusal.supported_refusal else "dirty",
        )
    )
    malformed = build_sampled_implicit_promotion_matrix(operations=("union",), allowed_targets=("subdivision",))
    rows.append(
        SurfaceSampledImplicitReferenceFixtureRow(
            fixture_id="sampled-implicit-reference/malformed-promotion",
            route_kind="malformed",
            payload_kind="sampled-implicit-promotion",
            passed=not malformed.passed and any(diagnostic.code == "missing-target" for diagnostic in malformed.diagnostics),
            reference_state="clean",
        )
    )
    return tuple(rows)


def verify_sampled_implicit_reference_fixture_promotions() -> SurfaceSampledImplicitReferenceFixturePromotionReport:
    """Verify sampled/implicit reference fixture promotions are clean and no-mesh."""

    rows = enumerate_sampled_implicit_reference_fixture_promotions()
    diagnostics: list[SurfaceSampledImplicitPromotionDiagnostic] = []
    required = SurfaceSampledImplicitReferenceFixturePromotionReport(()).required_route_kinds
    by_kind = {kind: [row for row in rows if row.route_kind == kind] for kind in required}
    for route_kind, kind_rows in by_kind.items():
        if not kind_rows:
            diagnostics.append(
                SurfaceSampledImplicitPromotionDiagnostic(
                    code="incomplete-route",
                    operation="union",
                    left_family="sampled-implicit-reference",
                    right_family="sampled-implicit-reference",
                    message=f"Sampled/implicit reference fixture promotion is missing route kind {route_kind}.",
                )
            )
        for row in kind_rows:
            if not row.passed or row.reference_state != "clean":
                diagnostics.append(
                    SurfaceSampledImplicitPromotionDiagnostic(
                        code="incomplete-route",
                        operation="union",
                        left_family=row.payload_kind,
                        right_family=row.payload_kind,
                        message=f"Sampled/implicit reference fixture {row.fixture_id} is not clean evidence.",
                    )
                )
            if not row.no_mesh_fallback:
                diagnostics.append(
                    SurfaceSampledImplicitPromotionDiagnostic(
                        code="mesh-fallback",
                        operation="union",
                        left_family=row.payload_kind,
                        right_family=row.payload_kind,
                        message=f"Sampled/implicit reference fixture {row.fixture_id} attempted mesh fallback.",
                    )
                )
    return SurfaceSampledImplicitReferenceFixturePromotionReport(rows=rows, diagnostics=tuple(diagnostics))


def _surface_csg_fixture_category(pair_class: SurfaceCSGRoutePairClass) -> Literal[
    "crossing", "tangent", "coincident", "boundary", "singular", "refusal"
]:
    if pair_class in {"spline-nurbs-pair"}:
        return "coincident"
    if pair_class in {"sweep-pair", "subdivision-pair"}:
        return "boundary"
    return "crossing"


def collect_higher_order_csg_residual(
    route: SurfaceCSGRouteRegistryRow,
    *,
    max_residual: float,
    tolerance: float,
    iteration_count: int,
    converged: bool,
    patch_ids: Iterable[str] = (),
) -> SurfaceCSGResidualRecord:
    """Collect a declared-tolerance residual witness without re-solving the route."""

    return SurfaceCSGResidualRecord(
        route_id=route.route_id,
        operation=route.operation,
        left_family=route.left_family,
        right_family=route.right_family,
        max_residual=max_residual,
        tolerance=tolerance,
        iteration_count=iteration_count,
        converged=bool(converged),
        patch_ids=tuple(patch_ids),
    )


def classify_higher_order_csg_degeneracies(
    residual: SurfaceCSGResidualRecord,
    *,
    ambiguous: bool = False,
    overlap: bool = False,
    singularity: bool = False,
    budget_exhausted: bool = False,
    location: str | None = None,
) -> tuple[SurfaceCSGDegeneracyRecord, ...]:
    """Classify residual and authored-route state into deterministic diagnostics."""

    records: list[SurfaceCSGDegeneracyRecord] = []
    base = {
        "route_id": residual.route_id,
        "operation": residual.operation,
        "left_family": residual.left_family,
        "right_family": residual.right_family,
        "location": location,
    }
    if not residual.converged:
        records.append(
            SurfaceCSGDegeneracyRecord(
                code="non-convergence",
                message="Higher-order CSG route did not converge within its declared iteration budget.",
                residual=residual.max_residual,
                **base,
            )
        )
    elif residual.max_residual > residual.tolerance:
        records.append(
            SurfaceCSGDegeneracyRecord(
                code="high-residual",
                message="Higher-order CSG route residual exceeds declared tolerance.",
                residual=residual.max_residual,
                **base,
            )
        )
    if ambiguous:
        records.append(
            SurfaceCSGDegeneracyRecord(
                code="ambiguous-route",
                message="Higher-order CSG route has unresolved authored ambiguity.",
                **base,
            )
        )
    if overlap:
        records.append(
            SurfaceCSGDegeneracyRecord(
                code="overlap",
                message="Higher-order CSG route produced an overlap or coincident-region condition.",
                blocking=False,
                **base,
            )
        )
    if singularity:
        records.append(
            SurfaceCSGDegeneracyRecord(
                code="singularity",
                message="Higher-order CSG route encountered a singular evaluator or local parameter event.",
                **base,
            )
        )
    if budget_exhausted:
        records.append(
            SurfaceCSGDegeneracyRecord(
                code="budget-refusal",
                message="Higher-order CSG route exhausted its declared safety or refinement budget.",
                residual=residual.max_residual,
                **base,
            )
        )
    return tuple(records)


def format_higher_order_csg_route_diagnostics(
    route: SurfaceCSGRouteRegistryRow,
    *,
    residual: SurfaceCSGResidualRecord | None = None,
    degeneracies: Iterable[SurfaceCSGDegeneracyRecord] = (),
) -> tuple[SurfaceCSGRouteSupportDiagnostic | SurfaceCSGAmbiguityDiagnostic, ...]:
    """Format route, residual, and degeneracy state into user-facing diagnostics."""

    diagnostics: list[SurfaceCSGRouteSupportDiagnostic | SurfaceCSGAmbiguityDiagnostic] = []
    if route.diagnostic is not None:
        diagnostics.append(route.diagnostic)
    if residual is not None and not residual.within_tolerance:
        diagnostics.append(
            SurfaceCSGRouteSupportDiagnostic(
                code="non-executable-route",
                operation=route.operation,
                left_family=route.left_family,
                right_family=route.right_family,
                pair_class=route.pair_class,
                message=(
                    f"Surface CSG route {route.route_id} residual {residual.max_residual} "
                    f"exceeds tolerance {residual.tolerance} or did not converge."
                ),
            )
        )
    for degeneracy in degeneracies:
        if degeneracy.code == "ambiguous-route":
            diagnostics.append(
                SurfaceCSGAmbiguityDiagnostic(
                    route_id=degeneracy.route_id,
                    operation=degeneracy.operation,
                    left_family=degeneracy.left_family,
                    right_family=degeneracy.right_family,
                    message=degeneracy.message,
                    location=degeneracy.location,
                    blocking=degeneracy.blocking,
                )
            )
        elif degeneracy.blocking:
            diagnostics.append(
                SurfaceCSGRouteSupportDiagnostic(
                    code="non-executable-route",
                    operation=degeneracy.operation,
                    left_family=degeneracy.left_family,
                    right_family=degeneracy.right_family,
                    pair_class=route.pair_class,
                    message=degeneracy.message,
                )
            )
    return tuple(diagnostics)


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
    elif left_family in SAMPLED_SURFACE_CSG_FAMILIES or right_family in SAMPLED_SURFACE_CSG_FAMILIES:
        phase = "operand-family-eligibility"
        support_state = "unsupported"
        required_future_capability = _surface_boolean_required_future_capability(
            operation, left_family, right_family
        )
    elif left_capability.support_phase == "planned" or right_capability.support_phase == "planned":
        phase = "operand-family-eligibility"
        support_state = "not-yet-implemented"
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


def _sampled_patch_point_to_uv(
    patch: SurfacePatch,
    point: Sequence[float],
    *,
    sample_count: int = 17,
) -> tuple[float, float]:
    target = np.asarray(point, dtype=float).reshape(3)
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    best_uv = (float(u0), float(v0))
    best_distance = float("inf")
    for u in np.linspace(u0, u1, max(2, int(sample_count))):
        for v in np.linspace(v0, v1, max(2, int(sample_count))):
            candidate = patch.point_at(float(u), float(v))
            distance = float(np.linalg.norm(candidate - target))
            if distance < best_distance:
                best_distance = distance
                best_uv = (float(u), float(v))
    return best_uv


def map_surface_csg_curve_to_patch_local(
    curve: SurfaceCSGCurvePrimitive,
    patch_ref: SurfaceBooleanPatchRef,
    patch: SurfacePatch,
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
        elif isinstance(patch, (RuledSurfacePatch, BSplineSurfacePatch, NURBSSurfacePatch, SweepSurfacePatch, SubdivisionSurfacePatch)):
            points_uv.append(_sampled_patch_point_to_uv(patch, point))
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


def map_surface_csg_curve_to_affected_patches(
    curve: SurfaceCSGCurvePrimitive,
    patch_bindings: Sequence[tuple[SurfaceBooleanPatchRef, SurfacePatch]],
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> SurfaceCSGIntersectionMappingResult:
    """Map one intersection curve into every affected patch-local domain."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    mappings = tuple(
        map_surface_csg_curve_to_patch_local(curve, patch_ref, patch, policy=normalized_policy)
        for patch_ref, patch in patch_bindings
    )
    diagnostics: list[SurfaceCSGCurveMappingDiagnostic] = []
    if len(mappings) < 2:
        patch_ref = patch_bindings[0][0] if patch_bindings else SurfaceBooleanPatchRef(operand_index=-1, patch_index=-1)
        diagnostics.append(
            _surface_csg_mapping_diagnostic(
                "ambiguous-curve",
                "CSG intersection curve mapping requires both affected patches.",
                patch_ref=patch_ref,
                curve=curve,
                policy=normalized_policy,
            )
        )
    if diagnostics:
        failed = SurfaceCSGPatchLocalCurveMappingResult(
            source_curve=curve,
            patch=diagnostics[0].patch,
            diagnostics=tuple(diagnostics),
        )
        return SurfaceCSGIntersectionMappingResult(curve_mappings=(*mappings, failed))
    return SurfaceCSGIntersectionMappingResult(curve_mappings=mappings)


def map_surface_csg_coincident_region_loop(
    source_region_id: str,
    patch_ref: SurfaceBooleanPatchRef,
    patch: SurfacePatch,
    points_uv: Sequence[Sequence[float]],
    *,
    source_curve_digests: Sequence[str] = (),
) -> SurfaceCSGPatchLocalRegionMappingResult:
    """Map a coincident region loop into one patch-local trim domain."""

    try:
        loop = TrimLoop(points_uv, category="outer").normalized()
        loop.validate_against_domain(patch.domain)
    except ValueError as exc:
        diagnostic = SurfaceCSGCurveMappingDiagnostic(
            code="outside-domain" if "outside" in str(exc).lower() else "degenerate-curve",
            message=f"Coincident CSG region {source_region_id!r} could not be mapped to patch-local trim space: {exc}",
            patch=patch_ref,
            source_curve_digest=str(source_region_id),
        )
        return SurfaceCSGPatchLocalRegionMappingResult(
            source_region_id=str(source_region_id),
            patch=patch_ref,
            diagnostics=(diagnostic,),
        )
    return SurfaceCSGPatchLocalRegionMappingResult(
        source_region_id=str(source_region_id),
        patch=patch_ref,
        region_loop=SurfaceCSGPatchLocalRegionLoop(
            source_region_id=str(source_region_id),
            patch=patch_ref,
            loop=loop,
            source_curve_digests=tuple(source_curve_digests),
        ),
    )


def _surface_csg_patch_local_curve_from_parameters(
    source_curve: SurfaceCSGCurvePrimitive,
    patch_ref: SurfaceBooleanPatchRef,
    patch: SurfacePatch,
    parameters: Sequence[Sequence[float]],
    *,
    policy: SurfaceCSGTolerancePolicy,
) -> SurfaceCSGPatchLocalCurveMappingResult:
    source_digest = surface_csg_curve_digest(source_curve, policy=policy)
    try:
        local_curve = SurfaceCSGPatchLocalCurve(
            source_curve_digest=source_digest,
            patch=patch_ref,
            points_uv=tuple((float(uv[0]), float(uv[1])) for uv in parameters),
            domain_bounds=(
                float(patch.domain.u_range[0]),
                float(patch.domain.u_range[1]),
                float(patch.domain.v_range[0]),
                float(patch.domain.v_range[1]),
            ),
        )
    except (IndexError, TypeError, ValueError) as exc:
        diagnostic = SurfaceCSGCurveMappingDiagnostic(
            code="ambiguous-curve",
            message=f"CSG intersection parameters could not be emitted as patch-local curve: {exc}",
            patch=patch_ref,
            source_curve_digest=source_digest,
        )
        return SurfaceCSGPatchLocalCurveMappingResult(
            source_curve=source_curve,
            patch=patch_ref,
            diagnostics=(diagnostic,),
        )
    diagnostics = validate_surface_csg_patch_local_curve_domain(local_curve, policy=policy)
    if diagnostics:
        return SurfaceCSGPatchLocalCurveMappingResult(
            source_curve=source_curve,
            patch=patch_ref,
            diagnostics=diagnostics,
        )
    return SurfaceCSGPatchLocalCurveMappingResult(source_curve=source_curve, patch=patch_ref, curve=local_curve)


def intersect_analytic_bspline_patch_pair(
    first_ref: SurfaceBooleanPatchRef,
    first_patch: PlanarSurfacePatch | RuledSurfacePatch | RevolutionSurfacePatch | BSplineSurfacePatch,
    second_ref: SurfaceBooleanPatchRef,
    second_patch: PlanarSurfacePatch | RuledSurfacePatch | RevolutionSurfacePatch | BSplineSurfacePatch,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
    sample_count: int = 17,
) -> SurfaceCSGAnalyticBSplineIntersectionRecord:
    """Intersect one analytic patch with one B-spline patch for surface CSG."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    first_is_bspline = isinstance(first_patch, BSplineSurfacePatch)
    second_is_bspline = isinstance(second_patch, BSplineSurfacePatch)
    first_is_analytic = isinstance(first_patch, (PlanarSurfacePatch, RuledSurfacePatch, RevolutionSurfacePatch))
    second_is_analytic = isinstance(second_patch, (PlanarSurfacePatch, RuledSurfacePatch, RevolutionSurfacePatch))
    placeholder_request = make_surface_intersection_request(
        first_patch,
        second_patch,
        first_patch_ref=first_ref,
        second_patch_ref=second_ref,
        consumer="surface-csg",
    )
    if (first_is_bspline == second_is_bspline) or (first_is_analytic == second_is_analytic):
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer="surface-csg",
            family_pair=placeholder_request.normalized_family_pair,
            message="analytic-to-B-spline CSG requires exactly one analytic patch and one B-spline patch",
        )
        result, report = solve_analytic_spline_surface_intersection(placeholder_request, sample_count=sample_count)
        return SurfaceCSGAnalyticBSplineIntersectionRecord(
            first_patch=first_ref,
            second_patch=second_ref,
            intersection=result,
            residual_report=report,
            diagnostics=(diagnostic, *report.diagnostics),
        )

    result, report = solve_analytic_spline_surface_intersection(placeholder_request, sample_count=sample_count)
    diagnostics: list[SurfaceIntersectionSupportDiagnostic | SurfaceCSGCurveMappingDiagnostic] = list(report.diagnostics)
    curves: list[SurfaceCSGCurvePrimitive] = []
    patch_local_curves: list[SurfaceCSGPatchLocalCurve] = []
    for curve_record in result.curves:
        try:
            curve = make_surface_csg_curve(curve_record.kind, curve_record.points_3d, policy=normalized_policy)
        except ValueError as exc:
            source_digest = hashlib.sha256(repr(curve_record.points_3d).encode("utf-8")).hexdigest()
            diagnostics.append(
                SurfaceCSGCurveMappingDiagnostic(
                    code="degenerate-curve",
                    message=f"Analytic-to-B-spline CSG curve could not be emitted: {exc}",
                    patch=first_ref,
                    source_curve_digest=source_digest,
                )
            )
            continue
        curves.append(curve)
        first_parameters = curve_record.first_parameters
        second_parameters = curve_record.second_parameters
        first_mapping = (
            _surface_csg_patch_local_curve_from_parameters(
                curve,
                first_ref,
                first_patch,
                first_parameters,
                policy=normalized_policy,
            )
            if first_parameters
            else map_surface_csg_curve_to_patch_local(curve, first_ref, first_patch, policy=normalized_policy)
        )
        second_mapping = (
            _surface_csg_patch_local_curve_from_parameters(
                curve,
                second_ref,
                second_patch,
                second_parameters,
                policy=normalized_policy,
            )
            if second_parameters
            else map_surface_csg_curve_to_patch_local(curve, second_ref, second_patch, policy=normalized_policy)
        )
        for mapping in (first_mapping, second_mapping):
            diagnostics.extend(mapping.diagnostics)
            if mapping.curve is not None:
                patch_local_curves.append(mapping.curve)
    return SurfaceCSGAnalyticBSplineIntersectionRecord(
        first_patch=first_ref,
        second_patch=second_ref,
        intersection=result,
        residual_report=report,
        curves=tuple(curves),
        patch_local_curves=tuple(patch_local_curves),
        diagnostics=tuple(diagnostics),
    )


def validate_nurbs_csg_patch_weights(patch: NURBSSurfacePatch) -> tuple[NURBSWeightValidationDiagnostic, ...]:
    """Return rational weight diagnostics for a NURBS patch entering CSG."""

    return validate_nurbs_weights(patch.weights, control_net_shape=patch.control_net.shape[:2])


def intersect_analytic_nurbs_patch_pair(
    first_ref: SurfaceBooleanPatchRef,
    first_patch: PlanarSurfacePatch | RuledSurfacePatch | RevolutionSurfacePatch | NURBSSurfacePatch,
    second_ref: SurfaceBooleanPatchRef,
    second_patch: PlanarSurfacePatch | RuledSurfacePatch | RevolutionSurfacePatch | NURBSSurfacePatch,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
    sample_count: int = 17,
) -> SurfaceCSGAnalyticNURBSIntersectionRecord:
    """Intersect one analytic patch with one NURBS patch for surface CSG."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    first_is_nurbs = isinstance(first_patch, NURBSSurfacePatch)
    second_is_nurbs = isinstance(second_patch, NURBSSurfacePatch)
    first_is_analytic = isinstance(first_patch, (PlanarSurfacePatch, RuledSurfacePatch, RevolutionSurfacePatch))
    second_is_analytic = isinstance(second_patch, (PlanarSurfacePatch, RuledSurfacePatch, RevolutionSurfacePatch))
    request = make_surface_intersection_request(
        first_patch,
        second_patch,
        first_patch_ref=first_ref,
        second_patch_ref=second_ref,
        consumer="surface-csg",
    )
    nurbs_patch = first_patch if first_is_nurbs else second_patch if second_is_nurbs else None
    weight_diagnostics = (
        validate_nurbs_csg_patch_weights(nurbs_patch)
        if isinstance(nurbs_patch, NURBSSurfacePatch)
        else ()
    )
    if (first_is_nurbs == second_is_nurbs) or (first_is_analytic == second_is_analytic):
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer="surface-csg",
            family_pair=request.normalized_family_pair,
            message="analytic-to-NURBS CSG requires exactly one analytic patch and one NURBS patch",
        )
        result, report = solve_analytic_spline_surface_intersection(request, sample_count=sample_count)
        return SurfaceCSGAnalyticNURBSIntersectionRecord(
            first_patch=first_ref,
            second_patch=second_ref,
            intersection=result,
            residual_report=report,
            weight_diagnostics=weight_diagnostics,
            diagnostics=(diagnostic, *weight_diagnostics, *report.diagnostics),
        )

    result, report = solve_analytic_spline_surface_intersection(request, sample_count=sample_count)
    diagnostics: list[
        SurfaceIntersectionSupportDiagnostic | SurfaceCSGCurveMappingDiagnostic | NURBSWeightValidationDiagnostic
    ] = [*weight_diagnostics, *report.diagnostics]
    curves: list[SurfaceCSGCurvePrimitive] = []
    patch_local_curves: list[SurfaceCSGPatchLocalCurve] = []
    if weight_diagnostics:
        return SurfaceCSGAnalyticNURBSIntersectionRecord(
            first_patch=first_ref,
            second_patch=second_ref,
            intersection=result,
            residual_report=report,
            weight_diagnostics=weight_diagnostics,
            diagnostics=tuple(diagnostics),
        )
    for curve_record in result.curves:
        try:
            curve = make_surface_csg_curve(curve_record.kind, curve_record.points_3d, policy=normalized_policy)
        except ValueError as exc:
            source_digest = hashlib.sha256(repr(curve_record.points_3d).encode("utf-8")).hexdigest()
            diagnostics.append(
                SurfaceCSGCurveMappingDiagnostic(
                    code="degenerate-curve",
                    message=f"Analytic-to-NURBS CSG curve could not be emitted: {exc}",
                    patch=first_ref,
                    source_curve_digest=source_digest,
                )
            )
            continue
        curves.append(curve)
        first_mapping = (
            _surface_csg_patch_local_curve_from_parameters(
                curve,
                first_ref,
                first_patch,
                curve_record.first_parameters,
                policy=normalized_policy,
            )
            if curve_record.first_parameters
            else map_surface_csg_curve_to_patch_local(curve, first_ref, first_patch, policy=normalized_policy)
        )
        second_mapping = (
            _surface_csg_patch_local_curve_from_parameters(
                curve,
                second_ref,
                second_patch,
                curve_record.second_parameters,
                policy=normalized_policy,
            )
            if curve_record.second_parameters
            else map_surface_csg_curve_to_patch_local(curve, second_ref, second_patch, policy=normalized_policy)
        )
        for mapping in (first_mapping, second_mapping):
            diagnostics.extend(mapping.diagnostics)
            if mapping.curve is not None:
                patch_local_curves.append(mapping.curve)
    return SurfaceCSGAnalyticNURBSIntersectionRecord(
        first_patch=first_ref,
        second_patch=second_ref,
        intersection=result,
        residual_report=report,
        weight_diagnostics=weight_diagnostics,
        curves=tuple(curves),
        patch_local_curves=tuple(patch_local_curves),
        diagnostics=tuple(diagnostics),
    )


def intersect_spline_nurbs_patch_pair(
    first_ref: SurfaceBooleanPatchRef,
    first_patch: BSplineSurfacePatch | NURBSSurfacePatch,
    second_ref: SurfaceBooleanPatchRef,
    second_patch: BSplineSurfacePatch | NURBSSurfacePatch,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
    sample_count: int = 13,
) -> SurfaceCSGSplinePairIntersectionRecord:
    """Intersect B-spline/NURBS patch pairs for surface CSG curve routes."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    request = make_surface_intersection_request(
        first_patch,
        second_patch,
        first_patch_ref=first_ref,
        second_patch_ref=second_ref,
        consumer="surface-csg",
    )
    valid_pair = isinstance(first_patch, (BSplineSurfacePatch, NURBSSurfacePatch)) and isinstance(
        second_patch,
        (BSplineSurfacePatch, NURBSSurfacePatch),
    )
    if not valid_pair:
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer="surface-csg",
            family_pair=request.normalized_family_pair,
            message="spline/NURBS CSG requires two B-spline or NURBS patches",
        )
        result, report = solve_spline_spline_surface_intersection(request, sample_count=sample_count)
        return SurfaceCSGSplinePairIntersectionRecord(
            first_patch=first_ref,
            second_patch=second_ref,
            intersection=result,
            residual_report=report,
            diagnostics=(diagnostic, *report.diagnostics),
        )

    result, report = solve_spline_spline_surface_intersection(request, sample_count=sample_count)
    diagnostics: list[SurfaceIntersectionSupportDiagnostic | SurfaceCSGCurveMappingDiagnostic] = list(report.diagnostics)
    curves: list[SurfaceCSGCurvePrimitive] = []
    patch_local_curves: list[SurfaceCSGPatchLocalCurve] = []
    tangent_events: list[SurfaceCSGDegeneracyRecord] = []
    route = surface_csg_route_lookup("intersection", first_patch.family, second_patch.family)
    for curve_record in result.curves:
        try:
            curve = make_surface_csg_curve(curve_record.kind, curve_record.points_3d, policy=normalized_policy)
        except ValueError as exc:
            source_digest = hashlib.sha256(repr(curve_record.points_3d).encode("utf-8")).hexdigest()
            diagnostics.append(
                SurfaceCSGCurveMappingDiagnostic(
                    code="degenerate-curve",
                    message=f"Spline/NURBS CSG curve could not be emitted: {exc}",
                    patch=first_ref,
                    source_curve_digest=source_digest,
                )
            )
            continue
        curves.append(curve)
        first_mapping = _surface_csg_patch_local_curve_from_parameters(
            curve,
            first_ref,
            first_patch,
            curve_record.first_parameters,
            policy=normalized_policy,
        )
        second_mapping = _surface_csg_patch_local_curve_from_parameters(
            curve,
            second_ref,
            second_patch,
            curve_record.second_parameters,
            policy=normalized_policy,
        )
        for mapping in (first_mapping, second_mapping):
            diagnostics.extend(mapping.diagnostics)
            if mapping.curve is not None:
                patch_local_curves.append(mapping.curve)
    if report.converged and report.iterations:
        residual = collect_higher_order_csg_residual(
            route,
            max_residual=report.iterations[-1].max_residual,
            tolerance=DEFAULT_SURFACE_CSG_TOLERANCE_POLICY.degeneracy_tolerance,
            iteration_count=len(report.iterations),
            converged=True,
            patch_ids=(f"{first_ref.operand_index}:{first_ref.patch_index}", f"{second_ref.operand_index}:{second_ref.patch_index}"),
        )
        tangent_events.extend(
            classify_higher_order_csg_degeneracies(
                residual,
                singularity=result.quality == "degenerate",
                location="spline-pair:csg-intersection",
            )
        )
    return SurfaceCSGSplinePairIntersectionRecord(
        first_patch=first_ref,
        second_patch=second_ref,
        intersection=result,
        residual_report=report,
        curves=tuple(curves),
        patch_local_curves=tuple(patch_local_curves),
        tangent_events=tuple(tangent_events),
        diagnostics=tuple(diagnostics),
    )


def _surface_patch_domain_loop(patch: SurfacePatch) -> tuple[tuple[float, float], ...]:
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    return ((float(u0), float(v0)), (float(u1), float(v0)), (float(u1), float(v1)), (float(u0), float(v1)))


def detect_spline_nurbs_coincident_regions(
    first_ref: SurfaceBooleanPatchRef,
    first_patch: BSplineSurfacePatch | NURBSSurfacePatch,
    second_ref: SurfaceBooleanPatchRef,
    second_patch: BSplineSurfacePatch | NURBSSurfacePatch,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
    sample_count: int = 5,
) -> SurfaceCSGSplineCoincidentRegionRecord:
    """Detect full-domain coincident B-spline/NURBS regions for CSG overlap handling."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    request = make_surface_intersection_request(
        first_patch,
        second_patch,
        first_patch_ref=first_ref,
        second_patch_ref=second_ref,
        consumer="surface-csg",
    )
    valid_pair = isinstance(first_patch, (BSplineSurfacePatch, NURBSSurfacePatch)) and isinstance(
        second_patch,
        (BSplineSurfacePatch, NURBSSurfacePatch),
    )
    if not valid_pair:
        diagnostic = SurfaceIntersectionSupportDiagnostic(
            code="unsupported-family-pair",
            consumer="surface-csg",
            family_pair=request.normalized_family_pair,
            message="spline/NURBS coincident-region CSG requires two B-spline or NURBS patches",
        )
        return SurfaceCSGSplineCoincidentRegionRecord(
            first_patch=first_ref,
            second_patch=second_ref,
            intersection=normalize_surface_intersection_result(request, quality="unsupported"),
            diagnostics=(diagnostic,),
        )

    first_samples = _sampled_region_points(first_patch, sample_count)
    second_samples = _sampled_region_points(second_patch, sample_count)
    distances = tuple(
        float(np.linalg.norm(np.asarray(first_point, dtype=float) - np.asarray(second_point, dtype=float)))
        for first_point, second_point in zip(first_samples, second_samples, strict=True)
    )
    max_distance = max(distances, default=float("inf"))
    if max_distance > normalized_policy.equality_tolerance:
        diagnostic = SurfaceCSGArrangementDiagnostic(
            code="ambiguous-overlap",
            message=(
                "Spline/NURBS patches are not coincident within CSG equality tolerance; "
                f"max sampled separation was {max_distance}."
            ),
            patch=first_ref,
        )
        return SurfaceCSGSplineCoincidentRegionRecord(
            first_patch=first_ref,
            second_patch=second_ref,
            intersection=normalize_surface_intersection_result(
                request,
                max_residual=max_distance if np.isfinite(max_distance) else 0.0,
                quality="unsupported",
            ),
            diagnostics=(diagnostic,),
        )

    first_loop = _surface_patch_domain_loop(first_patch)
    second_loop = _surface_patch_domain_loop(second_patch)
    overlap = SurfaceIntersectionOverlapRegionRecord(
        region_id="spline-coincident-region-0",
        first_loop_uv=first_loop,
        second_loop_uv=second_loop,
    )
    result = normalize_surface_intersection_result(
        request,
        overlap_regions=(overlap,),
        max_residual=max_distance,
        quality="within-tolerance",
    )
    first_mapping = map_surface_csg_coincident_region_loop(overlap.region_id, first_ref, first_patch, overlap.first_loop_uv)
    second_mapping = map_surface_csg_coincident_region_loop(overlap.region_id, second_ref, second_patch, overlap.second_loop_uv)
    diagnostics = tuple(diagnostic for mapping in (first_mapping, second_mapping) for diagnostic in mapping.diagnostics)
    return SurfaceCSGSplineCoincidentRegionRecord(
        first_patch=first_ref,
        second_patch=second_ref,
        intersection=result,
        region_mappings=(first_mapping, second_mapping),
        diagnostics=diagnostics,
    )


def intersect_sweep_csg_patch_pair(
    first_ref: SurfaceBooleanPatchRef,
    first_patch: SurfacePatch,
    second_ref: SurfaceBooleanPatchRef,
    second_patch: SurfacePatch,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
    sample_count: int = 13,
) -> SurfaceCSGSweepPairIntersectionRecord:
    """Intersect sweep-participating patch pairs for surface CSG curve routes."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    request = make_surface_intersection_request(
        first_patch,
        second_patch,
        first_patch_ref=first_ref,
        second_patch_ref=second_ref,
        consumer="surface-csg",
    )
    result, report = solve_sweep_surface_intersection_adapter(request, sample_count=sample_count)
    diagnostics: list[SurfaceIntersectionSupportDiagnostic | SurfaceCSGCurveMappingDiagnostic] = list(
        diagnostic for diagnostic in report.diagnostics if isinstance(diagnostic, SurfaceIntersectionSupportDiagnostic)
    )
    ambiguity_diagnostics = tuple(
        SurfaceCSGAmbiguityDiagnostic(
            route_id=surface_csg_route_lookup("intersection", first_patch.family, second_patch.family).route_id,
            operation="intersection",
            left_family=first_patch.family,
            right_family=second_patch.family,
            message=diagnostic.message,
            location=f"sweep-parameter:{diagnostic.parameter}",
            blocking=diagnostic.blocking,
        )
        for diagnostic in report.diagnostics
        if not isinstance(diagnostic, SurfaceIntersectionSupportDiagnostic)
    )
    curves: list[SurfaceCSGCurvePrimitive] = []
    patch_local_curves: list[SurfaceCSGPatchLocalCurve] = []
    for curve_record in result.curves:
        try:
            curve = make_surface_csg_curve(curve_record.kind, curve_record.points_3d, policy=normalized_policy)
        except ValueError as exc:
            source_digest = hashlib.sha256(repr(curve_record.points_3d).encode("utf-8")).hexdigest()
            diagnostics.append(
                SurfaceCSGCurveMappingDiagnostic(
                    code="degenerate-curve",
                    message=f"Sweep CSG curve could not be emitted: {exc}",
                    patch=first_ref,
                    source_curve_digest=source_digest,
                )
            )
            continue
        curves.append(curve)
        first_mapping = _surface_csg_patch_local_curve_from_parameters(
            curve,
            first_ref,
            first_patch,
            curve_record.first_parameters,
            policy=normalized_policy,
        )
        second_mapping = _surface_csg_patch_local_curve_from_parameters(
            curve,
            second_ref,
            second_patch,
            curve_record.second_parameters,
            policy=normalized_policy,
        )
        for mapping in (first_mapping, second_mapping):
            diagnostics.extend(mapping.diagnostics)
            if mapping.curve is not None:
                patch_local_curves.append(mapping.curve)
    return SurfaceCSGSweepPairIntersectionRecord(
        first_patch=first_ref,
        second_patch=second_ref,
        intersection=result,
        residual_report=report,
        curves=tuple(curves),
        patch_local_curves=tuple(patch_local_curves),
        ambiguity_diagnostics=ambiguity_diagnostics,
        diagnostics=tuple(diagnostics),
    )


def intersect_subdivision_csg_patch_pair(
    first_ref: SurfaceBooleanPatchRef,
    first_patch: SurfacePatch,
    second_ref: SurfaceBooleanPatchRef,
    second_patch: SurfacePatch,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
    sample_count: int = 5,
) -> SurfaceCSGSubdivisionPairIntersectionRecord:
    """Intersect subdivision-participating patch pairs for surface CSG curve routes."""

    normalized_policy = normalize_surface_csg_tolerance_policy(policy)
    request = make_surface_intersection_request(
        first_patch,
        second_patch,
        first_patch_ref=first_ref,
        second_patch_ref=second_ref,
        consumer="surface-csg",
    )
    result, report = solve_subdivision_surface_intersection_adapter(request, sample_count=sample_count)
    diagnostics: list[SurfaceIntersectionSupportDiagnostic | SurfaceCSGCurveMappingDiagnostic] = list(report.diagnostics)
    curves: list[SurfaceCSGCurvePrimitive] = []
    patch_local_curves: list[SurfaceCSGPatchLocalCurve] = []
    for curve_record in result.curves:
        try:
            curve = make_surface_csg_curve(curve_record.kind, curve_record.points_3d, policy=normalized_policy)
        except ValueError as exc:
            source_digest = hashlib.sha256(repr(curve_record.points_3d).encode("utf-8")).hexdigest()
            diagnostics.append(
                SurfaceCSGCurveMappingDiagnostic(
                    code="degenerate-curve",
                    message=f"Subdivision CSG curve could not be emitted: {exc}",
                    patch=first_ref,
                    source_curve_digest=source_digest,
                )
            )
            continue
        curves.append(curve)
        first_mapping = _surface_csg_patch_local_curve_from_parameters(
            curve,
            first_ref,
            first_patch,
            curve_record.first_parameters,
            policy=normalized_policy,
        )
        second_mapping = _surface_csg_patch_local_curve_from_parameters(
            curve,
            second_ref,
            second_patch,
            curve_record.second_parameters,
            policy=normalized_policy,
        )
        for mapping in (first_mapping, second_mapping):
            diagnostics.extend(mapping.diagnostics)
            if mapping.curve is not None:
                patch_local_curves.append(mapping.curve)
    return SurfaceCSGSubdivisionPairIntersectionRecord(
        first_patch=first_ref,
        second_patch=second_ref,
        intersection=result,
        adapter_report=report,
        curves=tuple(curves),
        patch_local_curves=tuple(patch_local_curves),
        diagnostics=tuple(diagnostics),
    )


def _sampled_region_points(patch: SurfacePatch, sample_count: int) -> tuple[tuple[float, float, float], ...]:
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    count = max(2, int(sample_count))
    return tuple(
        tuple(float(component) for component in patch.point_at(float(u), float(v)))
        for u in np.linspace(u0, u1, count)
        for v in np.linspace(v0, v1, count)
    )


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


def _surface_csg_classification_for_support(support: SurfaceBooleanFamilyPairSupport) -> str:
    if support.supported:
        if support.support_state == "exact":
            return "supported-exact"
        return "supported-declared"
    if support.left_family not in PATCH_FAMILY_CAPABILITY_MATRIX or support.right_family not in PATCH_FAMILY_CAPABILITY_MATRIX:
        return "unsupported-family"
    higher_order = classify_higher_order_csg_pair(support.operation, support.left_family, support.right_family)
    if higher_order.solver_boundary == "higher-order-exact-solver":
        return "higher-order-refusal"
    if higher_order.solver_boundary == "sampled-tessellation-boundary":
        return "sampled-boundary-refusal"
    return "non-csg-or-unsupported"


def available_family_csg_classification_rows(
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
    families: Iterable[str] | None = None,
) -> tuple[SurfaceCSGFamilyClassificationRow, ...]:
    """Return explicit CSG classification rows for every requested family pair."""

    source = registry if registry is not None else SURFACE_CSG_SOLVER_REGISTRY
    family_keys = tuple(sorted(PATCH_FAMILY_CAPABILITY_MATRIX if families is None else tuple(families)))
    rows: list[SurfaceCSGFamilyClassificationRow] = []
    for operation in SURFACE_BOOLEAN_OPERATIONS:
        for left_family in family_keys:
            for right_family in family_keys:
                support = source.support_for(operation, left_family, right_family)
                classification = _surface_csg_classification_for_support(support)
                diagnostic = "" if support.supported else build_surface_boolean_unsupported_family_diagnostic(support).message
                rows.append(
                    SurfaceCSGFamilyClassificationRow(
                        operation=operation,
                        left_family=left_family,
                        right_family=right_family,
                        supported=support.supported,
                        classification=classification,
                        support_state=support.support_state,
                        diagnostic=diagnostic,
                    )
                )
    return tuple(rows)


def verify_available_family_csg_classification_rows(
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
    families: Iterable[str] | None = None,
) -> SurfaceCSGFamilyClassificationReport:
    """Verify complete CSG classification rows for every operation/family pair."""

    source = registry if registry is not None else SURFACE_CSG_SOLVER_REGISTRY
    family_keys = tuple(sorted(PATCH_FAMILY_CAPABILITY_MATRIX if families is None else tuple(families)))
    support_matrix = {
        (record.operation, record.left_family, record.right_family): record
        for record in source.support_records
        if record.left_family in family_keys and record.right_family in family_keys
    }
    registry_report = build_surface_csg_solver_registry(
        support_matrix=support_matrix,
        family_capability_matrix={family: PATCH_FAMILY_CAPABILITY_MATRIX.get(family) for family in family_keys},
    )
    rows = available_family_csg_classification_rows(registry=source, families=family_keys)
    diagnostics = list(registry_report.diagnostics)
    expected_count = len(SURFACE_BOOLEAN_OPERATIONS) * len(family_keys) * len(family_keys)
    if len(rows) != expected_count:
        diagnostics.append(
            SurfaceCSGSolverRegistryDiagnostic(
                code="missing-pair",
                operation="union",
                left_family="availability-report",
                right_family="availability-report",
                message=f"CSG classification row count {len(rows)} did not match expected {expected_count}.",
            )
        )
    missing_classification = [row for row in rows if not row.classification]
    for row in missing_classification:
        diagnostics.append(
            SurfaceCSGSolverRegistryDiagnostic(
                code="missing-future-capability",
                operation=row.operation,
                left_family=row.left_family,
                right_family=row.right_family,
                message=f"CSG classification row {row.operation} {row.left_family}/{row.right_family} is empty.",
            )
        )
    return SurfaceCSGFamilyClassificationReport(
        passed=not diagnostics,
        rows=rows,
        diagnostics=tuple(diagnostics),
    )


def collect_surface_csg_no_mesh_fallback_evidence(
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
    families: Iterable[str] | None = None,
) -> tuple[SurfaceCSGNoMeshFallbackEvidenceRecord, ...]:
    """Collect CSG evidence proving unsupported pairs refuse rather than mesh fallback."""

    rows = available_family_csg_classification_rows(registry=registry, families=families)
    evidence: list[SurfaceCSGNoMeshFallbackEvidenceRecord] = []
    for row in rows:
        if row.supported:
            evidence.append(
                SurfaceCSGNoMeshFallbackEvidenceRecord(
                    operation=row.operation,
                    left_family=row.left_family,
                    right_family=row.right_family,
                    result_kind="supported-surface",
                    message="Surface CSG row is supported by the surface solver registry.",
                )
            )
            continue
        evidence.append(
            SurfaceCSGNoMeshFallbackEvidenceRecord(
                operation=row.operation,
                left_family=row.left_family,
                right_family=row.right_family,
                result_kind="diagnostic-refusal",
                message=row.diagnostic,
                mesh_fallback_attempted=False,
            )
        )
    return tuple(evidence)


def verify_surface_csg_no_mesh_fallback_evidence(
    *,
    registry: SurfaceCSGSolverRegistryRecord | None = None,
    families: Iterable[str] | None = None,
) -> SurfaceCSGNoMeshFallbackReport:
    """Verify CSG unsupported rows expose diagnostics and never hidden mesh fallback."""

    evidence = collect_surface_csg_no_mesh_fallback_evidence(registry=registry, families=families)
    diagnostics: list[SurfaceCSGPlanDiagnostic] = []
    for record in evidence:
        if record.mesh_fallback_attempted:
            diagnostics.append(
                SurfaceCSGPlanDiagnostic(
                    code="unsupported-family-pair",
                    operation=record.operation,
                    left_family=record.left_family,
                    right_family=record.right_family,
                    message="CSG evidence recorded a hidden mesh fallback attempt.",
                )
            )
        if record.result_kind == "diagnostic-refusal" and not record.message:
            diagnostics.append(
                SurfaceCSGPlanDiagnostic(
                    code="unsupported-family-pair",
                    operation=record.operation,
                    left_family=record.left_family,
                    right_family=record.right_family,
                    message="Unsupported CSG evidence requires an explicit diagnostic message.",
                )
            )
    return SurfaceCSGNoMeshFallbackReport(
        passed=not diagnostics,
        evidence=evidence,
        diagnostics=tuple(diagnostics),
    )


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
    if pair_support.supported:
        return SurfaceCSGHigherOrderSupportRecord(
            operation=operation,
            left_family=left_family,
            right_family=right_family,
            supported=True,
            solver_boundary="higher-order-declared-tolerance",
            support_state=pair_support.support_state,
            required_future_capability=None,
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


def record_surface_csg_continuity_handoff(
    seam_rebuild: SurfaceCSGSeamRebuildRecord,
    *,
    requested_continuity: Sequence[str] = ("C0", "G0"),
) -> SurfaceCSGContinuityHandoffRecord:
    """Record the continuity level seam rebuild can honestly hand to enforcement."""

    normalized_requested = tuple(str(item).upper() for item in requested_continuity)
    enforceable = tuple(item for item in normalized_requested if item in {"C0", "G0"})
    diagnostics: list[SurfaceCSGContinuityHandoffDiagnostic] = []
    if not seam_rebuild.supported:
        diagnostics.append(
            SurfaceCSGContinuityHandoffDiagnostic(
                code="invalid-seam-rebuild",
                message="Surface CSG continuity handoff requires a supported seam rebuild.",
            )
        )
    diagnostics.extend(
        SurfaceCSGContinuityHandoffDiagnostic(
            code="unsupported-continuity-enforcement",
            continuity=item,
            message=(
                f"Surface CSG seam adjacency records {item} as requested, but "
                "that continuity level requires the dedicated continuity enforcement route."
            ),
        )
        for item in normalized_requested
        if item not in {"C0", "G0"}
    )
    return SurfaceCSGContinuityHandoffRecord(
        seam_rebuild=seam_rebuild,
        requested_continuity=normalized_requested,
        enforceable_continuity=enforceable,
        diagnostics=tuple(diagnostics),
    )


def _surface_boolean_cleanup_body(body: SurfaceBody) -> SurfaceBody:
    cleaned_shells = tuple(rebuild_surface_csg_shell_seams(shell).shell for shell in body.iter_shells(world=True))
    return make_surface_body(cleaned_shells, metadata=body.metadata)


def detect_surface_csg_dangling_trims(body: SurfaceBody) -> tuple[SurfaceCSGValidityDiagnostic, ...]:
    """Detect trim loops that cannot be validated against their owning patch domain."""

    diagnostics: list[SurfaceCSGValidityDiagnostic] = []
    for shell_index, shell in enumerate(body.iter_shells(world=True)):
        for patch_index, patch in enumerate(shell.iter_patches(world=True)):
            for loop_index, trim_loop in enumerate(getattr(patch, "trim_loops", ())):
                try:
                    trim_loop.validate_against_domain(patch.domain)
                except ValueError as exc:
                    diagnostics.append(
                        SurfaceCSGValidityDiagnostic(
                            code="dangling-trim",
                            message=(
                                "Surface CSG runtime validity found dangling trim loop "
                                f"{loop_index} on shell {shell_index} patch {patch_index}: {exc}"
                            ),
                        )
                    )
    return tuple(diagnostics)


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
    diagnostics.extend(detect_surface_csg_dangling_trims(cleaned_body))

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


def check_surface_csg_runtime_result_validity(
    operation: SurfaceBooleanOperation,
    operands: SurfaceBooleanOperands,
    result: object,
    *,
    unresolved_diagnostics: Sequence[object] = (),
) -> SurfaceCSGRuntimeValidityReport:
    """Validate the object crossing the runtime CSG return boundary."""

    diagnostics: list[SurfaceCSGValidityDiagnostic] = []
    diagnostics.extend(
        SurfaceCSGValidityDiagnostic(
            code="unresolved-diagnostic",
            message=f"Surface CSG runtime result has unresolved diagnostic: {diagnostic}",
        )
        for diagnostic in unresolved_diagnostics
    )
    if isinstance(result, (Mesh, MeshGroup)):
        diagnostics.append(
            SurfaceCSGValidityDiagnostic(
                code="mesh-backed-fragment",
                message="Surface CSG runtime result attempted to return a mesh-backed fragment.",
            )
        )
        return SurfaceCSGRuntimeValidityReport(
            operation=operation,
            status="invalid",
            diagnostics=tuple(diagnostics),
        )
    if not isinstance(result, SurfaceBody):
        diagnostics.append(
            SurfaceCSGValidityDiagnostic(
                code="non-surface-result",
                message="Surface CSG runtime result must be a SurfaceBody.",
            )
        )
        return SurfaceCSGRuntimeValidityReport(
            operation=operation,
            status="invalid",
            diagnostics=tuple(diagnostics),
        )
    diagnostics.extend(detect_surface_csg_dangling_trims(result))
    if diagnostics:
        return SurfaceCSGRuntimeValidityReport(
            operation=operation,
            status="invalid",
            diagnostics=tuple(diagnostics),
        )
    gate = finalize_surface_csg_validity_gate(operation, operands, result)
    if not gate.accepted:
        diagnostics.extend(gate.diagnostics)
    if diagnostics:
        return SurfaceCSGRuntimeValidityReport(
            operation=operation,
            status="invalid",
            diagnostics=tuple(diagnostics),
            validity_gate=gate,
        )
    return SurfaceCSGRuntimeValidityReport(
        operation=operation,
        status="succeeded",
        result=gate.body,
        validity_gate=gate,
    )


def verify_surface_csg_persistence_tessellation_evidence(
    body: SurfaceBody,
    *,
    fixture_id: str,
    reference_state: Literal["clean", "dirty", "missing"] = "missing",
) -> SurfaceCSGReferencePromotionReport:
    """Verify `.impress`, tessellation-boundary, and reference promotion evidence."""

    from impression.io import dumps_impress_json, loads_impress_json, make_impress_document_payload
    from .tessellation import export_tessellation_request, tessellate_surface_body

    diagnostics: list[str] = []
    try:
        payload = make_impress_document_payload([body])
        loaded = loads_impress_json(dumps_impress_json(payload))
        loaded_body = loaded.bodies[0]
        persistence = SurfaceCSGPersistenceEvidenceRecord(
            fixture_id=fixture_id,
            passed=loaded_body.patch_count == body.patch_count,
            body_id=body.stable_identity,
            loaded_body_id=loaded_body.stable_identity,
            message="CSG surface body round-tripped through `.impress`.",
        )
        if not persistence.passed:
            diagnostics.append("CSG `.impress` round trip changed body patch count.")
    except Exception as exc:
        persistence = SurfaceCSGPersistenceEvidenceRecord(
            fixture_id=fixture_id,
            passed=False,
            body_id=body.stable_identity,
            message=f"CSG `.impress` round trip failed: {exc}",
        )
        diagnostics.append(persistence.message)
    try:
        tessellation = tessellate_surface_body(body, export_tessellation_request(require_watertight=False))
        face_count = int(tessellation.mesh.n_faces)
        tessellation_evidence = SurfaceCSGTessellationBoundaryEvidenceRecord(
            fixture_id=fixture_id,
            passed=face_count > 0,
            body_id=body.stable_identity,
            face_count=face_count,
            message="CSG surface body tessellated at the explicit boundary.",
        )
        if not tessellation_evidence.passed:
            diagnostics.append("CSG tessellation boundary produced no faces.")
    except Exception as exc:
        tessellation_evidence = SurfaceCSGTessellationBoundaryEvidenceRecord(
            fixture_id=fixture_id,
            passed=False,
            body_id=body.stable_identity,
            message=f"CSG tessellation boundary failed: {exc}",
        )
        diagnostics.append(tessellation_evidence.message)
    if reference_state != "clean":
        diagnostics.append(f"CSG reference fixture {fixture_id!r} is {reference_state}; clean evidence is required.")
    return SurfaceCSGReferencePromotionReport(
        fixture_id=fixture_id,
        persistence=persistence,
        tessellation=tessellation_evidence,
        reference_state=reference_state,
        diagnostics=tuple(diagnostics),
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


def build_surface_csg_fragments_from_arrangement(
    arrangement: SurfaceCSGPatchLocalArrangementGraph,
    patch: PlanarSurfacePatch,
) -> SurfaceCSGFragmentBuildResult:
    """Convert arrangement face candidates into explicit surface-native fragments."""

    diagnostics: list[SurfaceCSGFragmentBuildDiagnostic] = []
    diagnostics.extend(
        SurfaceCSGFragmentBuildDiagnostic(
            code="invalid-arrangement",
            message=diagnostic.message,
            patch=diagnostic.patch,
        )
        for diagnostic in arrangement.diagnostics
    )
    if not arrangement.face_candidates:
        diagnostics.append(
            SurfaceCSGFragmentBuildDiagnostic(
                code="missing-face-candidate",
                message="Surface CSG fragment builder requires at least one arrangement face candidate.",
                patch=arrangement.patch,
            )
        )
    fragments: list[SurfaceCSGSurfaceFragment] = []
    for index, candidate in enumerate(arrangement.face_candidates):
        try:
            candidate.loop.validate_against_domain(patch.domain)
        except ValueError as exc:
            diagnostics.append(
                SurfaceCSGFragmentBuildDiagnostic(
                    code="invalid-fragment-loop",
                    message=f"Surface CSG fragment loop is invalid for patch domain: {exc}",
                    patch=candidate.patch,
                    face_id=candidate.face_id,
                )
            )
            continue
        fragments.append(
            SurfaceCSGSurfaceFragment(
                fragment_id=f"{candidate.face_id}:fragment{index}",
                source_patch=candidate.patch,
                patch=replace(patch, trim_loops=(candidate.loop,)),
                loop=candidate.loop,
                source_face_id=candidate.face_id,
                cut_curve_ids=tuple(sorted(candidate.cut_curve_ids)),
            )
        )
    return SurfaceCSGFragmentBuildResult(
        arrangement=arrangement,
        fragments=tuple(fragments),
        diagnostics=tuple(diagnostics),
    )


def resolve_surface_csg_coincident_fragment_ownership(
    fragment: SurfaceCSGSurfaceFragment,
    classification: SurfaceCSGFragmentClassificationRecord,
) -> SurfaceCSGCoincidentOwnershipRecord:
    """Resolve ownership for fragments classified on the opposing boundary."""

    if classification.relation != "on":
        return SurfaceCSGCoincidentOwnershipRecord(
            fragment_id=fragment.fragment_id,
            patch=fragment.source_patch,
            relation=classification.relation,
        )
    if not classification.cut_curve_ids:
        diagnostic = SurfaceCSGCoincidentOwnershipDiagnostic(
            code="missing-cut-provenance",
            message=(
                "Surface CSG coincident fragment ownership requires cut-curve "
                "provenance for boundary-classified fragments."
            ),
            fragment_id=fragment.fragment_id,
            patch=fragment.source_patch,
        )
        return SurfaceCSGCoincidentOwnershipRecord(
            fragment_id=fragment.fragment_id,
            patch=fragment.source_patch,
            relation=classification.relation,
            policy="refuse-without-cut-provenance",
            diagnostics=(diagnostic,),
        )
    return SurfaceCSGCoincidentOwnershipRecord(
        fragment_id=fragment.fragment_id,
        patch=fragment.source_patch,
        relation=classification.relation,
        owner_patch=fragment.source_patch,
        policy="source-patch-with-cut-provenance",
    )


def classify_surface_csg_fragments_against_body(
    fragments: Sequence[SurfaceCSGSurfaceFragment],
    opposing_body: SurfaceBody,
    *,
    policy: SurfaceCSGTolerancePolicy | Mapping[str, float] | None = None,
) -> SurfaceCSGClassifiedFragmentSet:
    """Classify arrangement-built fragments and collect coincident ownership records."""

    classifications: list[SurfaceCSGFragmentClassificationRecord] = []
    ownership: list[SurfaceCSGCoincidentOwnershipRecord] = []
    for fragment in fragments:
        classification = classify_surface_csg_fragment_against_body(
            fragment.source_patch,
            fragment.patch,
            opposing_body,
            trim_loop=fragment.loop,
            cut_curve_ids=fragment.cut_curve_ids,
            policy=policy,
        )
        classifications.append(classification)
        ownership.append(resolve_surface_csg_coincident_fragment_ownership(fragment, classification))
    return SurfaceCSGClassifiedFragmentSet(
        fragments=tuple(fragments),
        classifications=tuple(classifications),
        coincident_ownership=tuple(ownership),
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


def classify_surface_csg_cap_eligibility(
    selection: SurfaceCSGOperationSelectionRecord,
    source_patch: SurfacePatch | None = None,
) -> SurfaceCSGCapEligibilityRecord:
    """Classify whether a selected fragment can generate a surface-native cap."""

    if not selection.cut_cap.required:
        return SurfaceCSGCapEligibilityRecord(
            selection=selection,
            required=False,
            eligible=True,
            reason="operation selection does not require a cut cap",
        )
    if source_patch is None:
        diagnostic = SurfaceCSGUnsupportedCapDiagnostic(
            code="missing-source-patch",
            patch=selection.patch,
            message=(
                "Surface CSG cap eligibility requires the source patch for "
                f"operand {selection.patch.operand_index} patch {selection.patch.patch_index}."
            ),
        )
        return SurfaceCSGCapEligibilityRecord(
            selection=selection,
            required=True,
            eligible=False,
            reason="missing source patch",
            diagnostics=(diagnostic,),
        )
    if not isinstance(source_patch, PlanarSurfacePatch):
        diagnostic = SurfaceCSGUnsupportedCapDiagnostic(
            code="unsupported-cap-family",
            patch=selection.patch,
            cap_family=source_patch.family,
            message=(
                f"Surface CSG cap eligibility supports planar cap payloads; "
                f"source family {source_patch.family!r} requires a surface-native cap producer."
            ),
        )
        return SurfaceCSGCapEligibilityRecord(
            selection=selection,
            required=True,
            eligible=False,
            cap_family=source_patch.family,
            reason="unsupported cap family",
            diagnostics=(diagnostic,),
        )
    return SurfaceCSGCapEligibilityRecord(
        selection=selection,
        required=True,
        eligible=True,
        cap_family="planar",
        reason="planar source patch can generate a surface-native cap",
    )


def select_surface_csg_operation_fragment_set(
    operation: SurfaceBooleanOperation,
    classified_fragments: SurfaceCSGClassifiedFragmentSet,
    *,
    source_patches: Mapping[SurfaceBooleanPatchRef, SurfacePatch] | None = None,
) -> SurfaceCSGOperationFragmentSelectionSet:
    """Select operation fragments and classify cap policy for a classified fragment set."""

    diagnostics: list[SurfaceCSGOperationSelectionDiagnostic] = []
    for classification in classified_fragments.classifications:
        if not classification.supported:
            diagnostics.append(
                SurfaceCSGOperationSelectionDiagnostic(
                    code="unsupported-classification",
                    message=(
                        "Surface CSG operation selection cannot execute while "
                        "fragment classification has blocking diagnostics."
                    ),
                    patch=classification.patch,
                )
            )
    for ownership in classified_fragments.coincident_ownership:
        if not ownership.supported:
            diagnostics.append(
                SurfaceCSGOperationSelectionDiagnostic(
                    code="ambiguous-coincident-ownership",
                    message=(
                        "Surface CSG operation selection cannot execute while "
                        "coincident fragment ownership is unresolved."
                    ),
                    patch=ownership.patch,
                )
            )
    selections = select_surface_csg_operation_fragments(operation, classified_fragments.classifications)
    patch_lookup = source_patches or {}
    cap_eligibility = tuple(
        classify_surface_csg_cap_eligibility(selection, patch_lookup.get(selection.patch))
        for selection in selections
    )
    return SurfaceCSGOperationFragmentSelectionSet(
        operation=operation,
        selections=selections,
        cap_eligibility=cap_eligibility,
        diagnostics=tuple(diagnostics),
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


def _arrangement_vertices_and_edges(
    patch_ref: SurfaceBooleanPatchRef,
    points_uv: Sequence[Sequence[float]],
    *,
    source: Literal["trim-loop", "cut-curve", "coincident-region"],
    prefix: str,
    cut_curve_ids: tuple[str, ...] = (),
    closed: bool,
) -> tuple[tuple[SurfaceCSGArrangementVertex, ...], tuple[SurfaceCSGArrangementEdge, ...]]:
    vertices = tuple(
        SurfaceCSGArrangementVertex(
            vertex_id=f"{prefix}:v{index}",
            patch=patch_ref,
            point_uv=(float(point[0]), float(point[1])),
            source=source,
        )
        for index, point in enumerate(points_uv)
    )
    edge_count = len(vertices) if closed and len(vertices) > 1 else max(0, len(vertices) - 1)
    edges = tuple(
        SurfaceCSGArrangementEdge(
            edge_id=f"{prefix}:e{index}",
            patch=patch_ref,
            start_vertex_id=vertices[index].vertex_id,
            end_vertex_id=vertices[(index + 1) % len(vertices)].vertex_id,
            source=source,
            cut_curve_ids=cut_curve_ids,
        )
        for index in range(edge_count)
    )
    return vertices, edges


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
    vertices: list[SurfaceCSGArrangementVertex] = []
    edges: list[SurfaceCSGArrangementEdge] = []
    for curve_index, curve in enumerate(local_curves):
        curve_vertices, curve_edges = _arrangement_vertices_and_edges(
            patch_ref,
            curve.points_uv,
            source="cut-curve",
            prefix=f"curve{curve_index}:{curve.source_curve_digest}",
            cut_curve_ids=(curve.source_curve_digest,),
            closed=False,
        )
        vertices.extend(curve_vertices)
        edges.extend(curve_edges)
    loops = tuple(trim_loops)
    if generated_loop is not None:
        loops = (*loops, generated_loop)
    if not loops:
        loops = patch.trim_loops
    split_loops: list[SurfaceCSGSplitTrimLoopRecord] = []
    face_candidates: list[SurfaceCSGArrangementFaceCandidate] = []
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
        loop_index = len(split_loops)
        loop_vertices, loop_edges = _arrangement_vertices_and_edges(
            patch_ref,
            normalized_loop.points_uv,
            source="trim-loop",
            prefix=f"loop{loop_index}:{normalized_loop.category}",
            cut_curve_ids=tuple(sorted(cut_curve_ids)),
            closed=True,
        )
        vertices.extend(loop_vertices)
        edges.extend(loop_edges)
        split_loops.append(
            SurfaceCSGSplitTrimLoopRecord(
                patch=patch_ref,
                loop=normalized_loop,
                source_category=normalized_loop.category,
                cut_curve_ids=tuple(sorted(cut_curve_ids)),
            )
        )
        face_candidates.append(
            SurfaceCSGArrangementFaceCandidate(
                face_id=f"face{loop_index}:{normalized_loop.category}",
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
        vertices=tuple(vertices),
        edges=tuple(edges),
        face_candidates=tuple(face_candidates),
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


def orient_surface_csg_selected_fragment(
    operation: SurfaceBooleanOperation,
    fragment: SurfaceBooleanTrimmedPatchFragment,
    selection: SurfaceCSGOperationSelectionRecord,
) -> SurfaceCSGOrientedFragmentRecord:
    """Apply operation-specific orientation metadata to one selected fragment."""

    orientation: Literal["preserve", "reverse"] = "preserve"
    if operation == "difference" and selection.patch.operand_index != 0 and selection.role == "cut_cap":
        orientation = "reverse"
    metadata = dict(fragment.patch.metadata)
    kernel = dict(fragment.patch.kernel_metadata())
    consumer = dict(fragment.patch.consumer_metadata())
    kernel.update(
        {
            "boolean_operation": operation,
            "csg_fragment_orientation": orientation,
            "csg_fragment_role": selection.role,
            "source_operand_index": fragment.source_patch.operand_index,
            "source_patch_index": fragment.source_patch.patch_index,
            "cut_curve_ids": tuple(sorted(fragment.cut_curve_ids)),
        }
    )
    metadata["kernel"] = kernel
    if consumer:
        metadata["consumer"] = consumer
    return SurfaceCSGOrientedFragmentRecord(
        source_patch=fragment.source_patch,
        patch=replace(fragment.patch, metadata=metadata),
        role=selection.role,
        orientation=orientation,
        operation=operation,
        cut_curve_ids=tuple(sorted(fragment.cut_curve_ids)),
    )


def orient_surface_csg_selected_fragments(
    operation: SurfaceBooleanOperation,
    fragments: Sequence[SurfaceBooleanTrimmedPatchFragment],
    selections: Sequence[SurfaceCSGOperationSelectionRecord],
) -> tuple[SurfaceCSGOrientedFragmentRecord, ...]:
    """Apply orientation to all fragments that have matching operation selections."""

    selection_by_patch = {selection.patch: selection for selection in selections}
    oriented: list[SurfaceCSGOrientedFragmentRecord] = []
    for fragment in fragments:
        selection = selection_by_patch.get(fragment.source_patch)
        if selection is None:
            continue
        oriented.append(orient_surface_csg_selected_fragment(operation, fragment, selection))
    return tuple(
        sorted(
            oriented,
            key=lambda record: (*_surface_csg_patch_ref_sort_key(record.source_patch), record.role, record.orientation),
        )
    )


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
