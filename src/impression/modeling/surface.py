from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from impression.modeling.path3d import Path3D


PatchFamilySupportPhase = Literal["available", "implemented", "planned"]


@dataclass(frozen=True)
class PatchFamilyCapabilityRecord:
    """Declared support phase and operation coverage for one surface patch family."""

    family: str
    support_phase: PatchFamilySupportPhase
    operations: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class PatchFamilyAvailabilityDiagnostic:
    """Structured reason a patch family does not satisfy an availability gate."""

    family: str
    code: str
    message: str


@dataclass(frozen=True)
class PatchFamilyAvailabilityGateRecord:
    """Validation result for one patch family's availability declaration."""

    family: str
    support_phase: PatchFamilySupportPhase
    available: bool
    diagnostics: tuple[PatchFamilyAvailabilityDiagnostic, ...] = ()


@dataclass(frozen=True)
class PatchFamilyOperationSupportRecord:
    """Operation-level evidence used by availability promotion checks."""

    family: str
    operation: str
    supported: bool
    diagnostic: str = ""


@dataclass(frozen=True)
class AvailableFamilyOperationEvidenceRecord:
    """Inspectable operation-row evidence for one supported surface patch family."""

    family: str
    operation: str
    category: str
    supported: bool
    state: str
    source: str
    diagnostic: str = ""

    def __post_init__(self) -> None:
        for name in ("family", "operation", "category", "state", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"AvailableFamilyOperationEvidenceRecord.{name} must be non-empty.")
        if self.supported and self.state not in {"native", "import", "payload", "adapter", "promoted", "explicit"}:
            raise ValueError("Supported available-family operation rows must use a concrete supported state.")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "operation": self.operation,
            "category": self.category,
            "supported": self.supported,
            "state": self.state,
            "source": self.source,
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True)
class AvailableFamilyMissingEvidenceDiagnostic:
    """Structured missing or unsafe availability evidence for one family operation row."""

    family: str
    category: str
    code: str
    message: str
    operation: str = ""

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "category": self.category,
            "operation": self.operation,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class AvailableFamilyOperationCompletenessReport:
    """Pass/fail report for one availability operation-row group."""

    category: str
    passed: bool
    rows: tuple[AvailableFamilyOperationEvidenceRecord, ...]
    diagnostics: tuple[AvailableFamilyMissingEvidenceDiagnostic, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "category": self.category,
            "passed": self.passed,
            "rows": [row.canonical_payload() for row in self.rows],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class AvailableFamilyReferenceEvidenceSummary:
    """Promoted reference evidence summary for one available-family evidence gate."""

    track: str
    required_evidence_types: tuple[str, ...]
    satisfied_evidence_types: tuple[str, ...]
    dirty_evidence_types: tuple[str, ...] = ()

    @property
    def missing_evidence_types(self) -> tuple[str, ...]:
        return tuple(
            evidence_type
            for evidence_type in self.required_evidence_types
            if evidence_type not in self.satisfied_evidence_types
        )

    @property
    def passed(self) -> bool:
        return not self.missing_evidence_types and not self.dirty_evidence_types

    def canonical_payload(self) -> dict[str, object]:
        return {
            "track": self.track,
            "required_evidence_types": self.required_evidence_types,
            "satisfied_evidence_types": self.satisfied_evidence_types,
            "dirty_evidence_types": self.dirty_evidence_types,
            "missing_evidence_types": self.missing_evidence_types,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class AvailableFamilyReferenceEvidenceReport:
    """Availability reference evidence gate report."""

    passed: bool
    summaries: tuple[AvailableFamilyReferenceEvidenceSummary, ...]
    diagnostics: tuple[AvailableFamilyMissingEvidenceDiagnostic, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "summaries": [summary.canonical_payload() for summary in self.summaries],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class AvailableFamilyCompletionSnapshot:
    """Deterministic serializable completion snapshot for supported patch families."""

    families: tuple[str, ...]
    passed: bool
    operation_categories: tuple[str, ...]
    diagnostic_count: int

    def canonical_payload(self) -> dict[str, object]:
        return {
            "families": self.families,
            "passed": self.passed,
            "operation_categories": self.operation_categories,
            "diagnostic_count": self.diagnostic_count,
        }


@dataclass(frozen=True)
class AvailableFamilyCompletionReport:
    """Aggregated completion report for supported surface patch-family availability."""

    passed: bool
    families: tuple[str, ...]
    operation_reports: tuple[AvailableFamilyOperationCompletenessReport, ...]
    reference_report: AvailableFamilyReferenceEvidenceReport
    diagnostics: tuple[AvailableFamilyMissingEvidenceDiagnostic, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "families": self.families,
            "operation_reports": [report.canonical_payload() for report in self.operation_reports],
            "reference_report": self.reference_report.canonical_payload(),
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class PatchFamilyPromotionEvidenceRecord:
    """Evidence for whether a patch family can be promoted to available."""

    family: str
    current_phase: PatchFamilySupportPhase
    promoted_phase: Literal["available", "planned"]
    operation_support: tuple[PatchFamilyOperationSupportRecord, ...]
    diagnostics: tuple[PatchFamilyAvailabilityDiagnostic, ...] = ()

    @property
    def promoted(self) -> bool:
        return self.current_phase != self.promoted_phase and self.promoted_phase == "available"


@dataclass(frozen=True)
class AdvancedPatchFamilyPromotionEvidenceRecord:
    """Evidence for whether an advanced patch family can be promoted to implemented."""

    family: str
    current_phase: PatchFamilySupportPhase
    promoted_phase: PatchFamilySupportPhase
    readiness: "PatchFamilyPromotionReadinessRecord"
    diagnostics: tuple["PatchFamilyPromotionGapRecord", ...] = ()

    @property
    def promoted(self) -> bool:
        return self.current_phase != self.promoted_phase and self.promoted_phase == "implemented"


@dataclass(frozen=True)
class AdvancedPatchFamilyPromotionReport:
    """Promotion gate result for all advanced authored patch families."""

    passed: bool
    evidence: tuple[AdvancedPatchFamilyPromotionEvidenceRecord, ...]
    diagnostics: tuple["PatchFamilyPromotionGapRecord", ...] = ()


@dataclass(frozen=True)
class NURBSWeightValidationDiagnostic:
    """Structured NURBS weight validation failure."""

    code: str
    message: str
    shape: tuple[int, ...]

    def canonical_payload(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "shape": self.shape}


@dataclass(frozen=True)
class NURBSRationalEvaluationMetadata:
    """Inspectable homogeneous evaluation components for a NURBS surface."""

    parameter: tuple[float, float]
    numerator: np.ndarray
    denominator: float
    point: np.ndarray
    weight_shape: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "numerator", np.asarray(self.numerator, dtype=float).reshape(3))
        object.__setattr__(self, "point", np.asarray(self.point, dtype=float).reshape(3))
        object.__setattr__(self, "denominator", float(self.denominator))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "numerator": self.numerator.tolist(),
            "denominator": self.denominator,
            "point": self.point.tolist(),
            "weight_shape": self.weight_shape,
        }


@dataclass(frozen=True)
class NURBSConicConstructionDiagnostic:
    """Structured diagnostic for exact NURBS conic helper requests."""

    code: str
    message: str
    conic_kind: str

    def canonical_payload(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "conic_kind": self.conic_kind}


@dataclass(frozen=True)
class NURBSConicConstructionRequest:
    """Authored request for an exact rational conic profile payload."""

    conic_kind: str
    center: tuple[float, float] = (0.0, 0.0)
    radius: float | None = None
    radii: tuple[float, float] | None = None
    start_angle_deg: float = 0.0
    end_angle_deg: float = 360.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "conic_kind", str(self.conic_kind).strip().lower())
        object.__setattr__(self, "center", tuple(float(value) for value in self.center))
        if self.radius is not None:
            object.__setattr__(self, "radius", float(self.radius))
        if self.radii is not None:
            object.__setattr__(self, "radii", tuple(float(value) for value in self.radii))
        object.__setattr__(self, "start_angle_deg", float(self.start_angle_deg))
        object.__setattr__(self, "end_angle_deg", float(self.end_angle_deg))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class NURBSConicProfilePayload:
    """NURBS-compatible exact conic control payload."""

    degree: int
    control_points_uv: np.ndarray
    weights: np.ndarray
    knots: tuple[float, ...]
    metadata: dict[str, object]
    diagnostics: tuple[NURBSConicConstructionDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_points_uv", np.asarray(self.control_points_uv, dtype=float).reshape(-1, 2))
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float).reshape(-1))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def supported(self) -> bool:
        return not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "degree": self.degree,
            "control_points_uv": self.control_points_uv.tolist(),
            "weights": self.weights.tolist(),
            "knots": self.knots,
            "metadata": self.metadata,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class DisplacementSourcePatchReferenceRecord:
    """Stable embedded-source identity for a displacement patch."""

    source_family: str
    source_patch_id: str
    embedded: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "source_family": self.source_family,
            "source_patch_id": self.source_patch_id,
            "embedded": self.embedded,
        }


@dataclass(frozen=True)
class DisplacementIdentityDiagnostic:
    """Diagnostic for displacement source identity policy checks."""

    code: str
    message: str
    source_patch_id: str

    def canonical_payload(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "source_patch_id": self.source_patch_id}


@dataclass(frozen=True)
class DisplacementSourceProvenanceRecord:
    """Resolved source relationship for a displacement authoring request."""

    source_family: str
    source_patch_id: str
    relationship: Literal["embedded", "in-body"]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "source_family": self.source_family,
            "source_patch_id": self.source_patch_id,
            "relationship": self.relationship,
        }


@dataclass(frozen=True)
class DisplacementSourceResolutionResult:
    """Result of resolving a displacement source patch identity."""

    source_patch: "SurfacePatch | None"
    provenance: DisplacementSourceProvenanceRecord | None
    diagnostic: DisplacementIdentityDiagnostic

    @property
    def resolved(self) -> bool:
        return self.source_patch is not None and self.provenance is not None and self.diagnostic.code.endswith("resolved")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "resolved": self.resolved,
            "provenance": None if self.provenance is None else self.provenance.canonical_payload(),
            "diagnostic": self.diagnostic.canonical_payload(),
        }


def resolve_displacement_source_identity(
    *,
    source_patch: "SurfacePatch | None" = None,
    source_patch_id: str | None = None,
    candidate_patches: Sequence["SurfacePatch"] = (),
    allow_cross_body_reference: bool = False,
) -> DisplacementSourceResolutionResult:
    """Resolve a displacement source patch without following external references by default."""

    if source_patch is not None:
        provenance = DisplacementSourceProvenanceRecord(
            source_family=source_patch.family,
            source_patch_id=source_patch.stable_identity,
            relationship="embedded",
        )
        return DisplacementSourceResolutionResult(
            source_patch=source_patch,
            provenance=provenance,
            diagnostic=DisplacementIdentityDiagnostic(
                code="embedded-source-resolved",
                message="Displacement source is embedded in the authored payload.",
                source_patch_id=source_patch.stable_identity,
            ),
        )
    requested_id = "" if source_patch_id is None else str(source_patch_id).strip()
    if not requested_id:
        return DisplacementSourceResolutionResult(
            source_patch=None,
            provenance=None,
            diagnostic=DisplacementIdentityDiagnostic(
                code="missing-source-identity",
                message="Displacement source resolution requires an embedded source patch or stable in-body source identity.",
                source_patch_id=requested_id,
            ),
        )
    for candidate in candidate_patches:
        if candidate.stable_identity == requested_id:
            provenance = DisplacementSourceProvenanceRecord(
                source_family=candidate.family,
                source_patch_id=candidate.stable_identity,
                relationship="in-body",
            )
            return DisplacementSourceResolutionResult(
                source_patch=candidate,
                provenance=provenance,
                diagnostic=DisplacementIdentityDiagnostic(
                    code="in-body-source-resolved",
                    message="Displacement source identity resolved to an in-body surface patch.",
                    source_patch_id=candidate.stable_identity,
                ),
            )
    code = "external-source-refused" if not allow_cross_body_reference else "missing-source-identity"
    message = (
        "Displacement source identity was not found in the authored body; cross-body references are refused by default."
        if not allow_cross_body_reference
        else "Displacement source identity was not found in the provided candidate patches."
    )
    return DisplacementSourceResolutionResult(
        source_patch=None,
        provenance=None,
        diagnostic=DisplacementIdentityDiagnostic(
            code=code,
            message=message,
            source_patch_id=requested_id,
        ),
    )


@dataclass(frozen=True)
class DisplacementDomainMappingRecord:
    """Mapping from displaced patch parameters to the authoritative source domain."""

    source_family: str
    source_domain: tuple[tuple[float, float], tuple[float, float]]
    displacement_domain: tuple[tuple[float, float], tuple[float, float]]
    projection_bounds: tuple[float, float, float, float]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "source_family": self.source_family,
            "source_domain": self.source_domain,
            "displacement_domain": self.displacement_domain,
            "projection_bounds": self.projection_bounds,
        }


@dataclass(frozen=True)
class DisplacementEvaluationDiagnostic:
    """Diagnostic describing displacement evaluation approximation behavior."""

    code: str
    message: str
    source_patch_id: str

    def canonical_payload(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "source_patch_id": self.source_patch_id}


@dataclass(frozen=True)
class SurfaceBodyCompletionEvidenceRecord:
    """Release-level evidence item used by the surface body completion gate."""

    track: str
    state: Literal["specified", "implemented", "verified", "unsupported", "retired"]
    spec: str
    implementation_owner: str
    evidence_type: str
    source: str = "implementation"

    def __post_init__(self) -> None:
        for name in ("track", "spec", "implementation_owner", "evidence_type", "source"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"SurfaceBodyCompletionEvidenceRecord.{name} must be non-empty.")
            object.__setattr__(self, name, value)
        if self.state not in {"specified", "implemented", "verified", "unsupported", "retired"}:
            raise ValueError("SurfaceBodyCompletionEvidenceRecord.state is invalid.")


@dataclass(frozen=True)
class SurfaceBodyCompletionDiagnostic:
    """Actionable missing evidence diagnostic for the completion gate."""

    track: str
    code: str
    message: str
    implementation_owner: str
    evidence_type: str
    spec: str = ""


@dataclass(frozen=True)
class SurfaceBodyCompletionReport:
    """Pass/fail release completion report."""

    passed: bool
    evidence: tuple[SurfaceBodyCompletionEvidenceRecord, ...]
    diagnostics: tuple[SurfaceBodyCompletionDiagnostic, ...]


@dataclass(frozen=True)
class SurfaceReferenceFixtureRequirementRecord:
    """Reference-evidence requirement for one promoted model-outputting track."""

    track: str
    required_evidence_types: tuple[str, ...]

    def __post_init__(self) -> None:
        track = str(self.track).strip()
        evidence_types = tuple(str(value).strip() for value in self.required_evidence_types)
        if not track:
            raise ValueError("SurfaceReferenceFixtureRequirementRecord.track must be non-empty.")
        if not evidence_types or not all(evidence_types):
            raise ValueError("SurfaceReferenceFixtureRequirementRecord.required_evidence_types must be non-empty.")
        object.__setattr__(self, "track", track)
        object.__setattr__(self, "required_evidence_types", evidence_types)


@dataclass(frozen=True)
class SurfaceReferenceArtifactClassRecord:
    """Durable artifact class required by model-output reference evidence."""

    artifact_class: str
    required_keys: tuple[str, ...]
    promoted_root: str
    dirty_root: str

    def __post_init__(self) -> None:
        artifact_class = str(self.artifact_class).strip()
        required_keys = tuple(str(value).strip() for value in self.required_keys)
        promoted_root = str(self.promoted_root).strip()
        dirty_root = str(self.dirty_root).strip()
        if not artifact_class:
            raise ValueError("SurfaceReferenceArtifactClassRecord.artifact_class must be non-empty.")
        if not required_keys or not all(required_keys):
            raise ValueError("SurfaceReferenceArtifactClassRecord.required_keys must be non-empty.")
        if not promoted_root or not dirty_root:
            raise ValueError("SurfaceReferenceArtifactClassRecord roots must be non-empty.")
        object.__setattr__(self, "artifact_class", artifact_class)
        object.__setattr__(self, "required_keys", required_keys)
        object.__setattr__(self, "promoted_root", promoted_root)
        object.__setattr__(self, "dirty_root", dirty_root)


@dataclass(frozen=True)
class SurfaceReferenceFixtureContractRecord:
    """Fixture contract connecting a capability track to artifact classes."""

    fixture_id: str
    track: str
    artifact_classes: tuple[str, ...]
    contract_version: str = "v1"

    def __post_init__(self) -> None:
        fixture_id = str(self.fixture_id).strip()
        track = str(self.track).strip()
        artifact_classes = tuple(str(value).strip() for value in self.artifact_classes)
        contract_version = str(self.contract_version).strip()
        if not fixture_id or not track or not contract_version:
            raise ValueError("SurfaceReferenceFixtureContractRecord fixture_id, track, and contract_version must be non-empty.")
        if not artifact_classes or not all(artifact_classes):
            raise ValueError("SurfaceReferenceFixtureContractRecord.artifact_classes must be non-empty.")
        object.__setattr__(self, "fixture_id", fixture_id)
        object.__setattr__(self, "track", track)
        object.__setattr__(self, "artifact_classes", artifact_classes)
        object.__setattr__(self, "contract_version", contract_version)


@dataclass(frozen=True)
class SurfaceReferenceEvidenceMatrixReport:
    """Pass/fail report for reference artifact and negative diagnostic evidence."""

    passed: bool
    requirements: tuple[SurfaceReferenceFixtureRequirementRecord, ...]
    evidence: tuple[SurfaceBodyCompletionEvidenceRecord, ...]
    diagnostics: tuple[SurfaceBodyCompletionDiagnostic, ...]


@dataclass(frozen=True)
class SurfaceContinuityRequest:
    """Authored continuity request kept separate from observed continuity."""

    requested: str = "C0"
    source: str = "authored"

    def __post_init__(self) -> None:
        requested = str(self.requested).strip()
        source = str(self.source).strip()
        if not requested:
            raise ValueError("SurfaceContinuityRequest.requested must be non-empty.")
        if not source:
            raise ValueError("SurfaceContinuityRequest.source must be non-empty.")
        object.__setattr__(self, "requested", requested)
        object.__setattr__(self, "source", source)


@dataclass(frozen=True)
class SurfaceContinuitySupportRecord:
    """Support verdict for an authored seam continuity request."""

    requested: str
    supported: bool
    support_state: Literal["supported", "unsupported", "not-yet-implemented"]
    diagnostic: str = ""


@dataclass(frozen=True)
class SurfaceUnsupportedContinuityDiagnostic:
    """Structured diagnostic for unsupported seam continuity classes."""

    requested: str
    supported_classes: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class SurfaceContinuityTolerancePolicy:
    """Tolerance policy carried by authored higher-order seam continuity constraints."""

    position_tolerance: float = 1e-9
    tangent_tolerance: float = 1e-6
    curvature_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("position_tolerance", "tangent_tolerance", "curvature_tolerance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"SurfaceContinuityTolerancePolicy.{name} must be a positive finite value.")
            object.__setattr__(self, name, value)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "curvature_tolerance": self.curvature_tolerance,
            "position_tolerance": self.position_tolerance,
            "tangent_tolerance": self.tangent_tolerance,
        }


@dataclass(frozen=True)
class SurfaceSeamBoundaryUseRef:
    """One participating boundary use in an authored seam continuity constraint."""

    seam_id: str
    boundary: "SurfaceBoundaryRef"
    role: Literal["first", "second", "open"] = "first"

    def __post_init__(self) -> None:
        seam_id = str(self.seam_id).strip()
        role = str(self.role).strip()
        if not seam_id:
            raise ValueError("SurfaceSeamBoundaryUseRef.seam_id must be non-empty.")
        if role not in {"first", "second", "open"}:
            raise ValueError("SurfaceSeamBoundaryUseRef.role must be first, second, or open.")
        object.__setattr__(self, "seam_id", seam_id)
        object.__setattr__(self, "role", role)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary": self.boundary.canonical_payload(),
            "role": self.role,
            "seam_id": self.seam_id,
        }


@dataclass(frozen=True)
class SurfaceContinuityConstraintDiagnostic:
    """Validation diagnostic for authored seam continuity constraints."""

    code: Literal["invalid-continuity", "invalid-boundary-count", "duplicate-boundary", "invalid-role"]
    message: str
    seam_id: str | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "seam_id": self.seam_id,
        }


@dataclass(frozen=True)
class SurfaceSeamContinuityConstraint:
    """Authored C/G continuity intent for one seam and its boundary uses."""

    seam_id: str
    requested: str
    boundary_uses: tuple[SurfaceSeamBoundaryUseRef, ...]
    tolerance_policy: SurfaceContinuityTolerancePolicy = field(default_factory=SurfaceContinuityTolerancePolicy)
    source: str = "authored"

    def __post_init__(self) -> None:
        seam_id = str(self.seam_id).strip()
        requested = str(self.requested).strip().upper()
        source = str(self.source).strip()
        boundary_uses = tuple(self.boundary_uses)
        if not seam_id:
            raise ValueError("SurfaceSeamContinuityConstraint.seam_id must be non-empty.")
        if not requested:
            raise ValueError("SurfaceSeamContinuityConstraint.requested must be non-empty.")
        if not source:
            raise ValueError("SurfaceSeamContinuityConstraint.source must be non-empty.")
        if not all(isinstance(boundary_use, SurfaceSeamBoundaryUseRef) for boundary_use in boundary_uses):
            raise TypeError("SurfaceSeamContinuityConstraint boundary_uses must be SurfaceSeamBoundaryUseRef instances.")
        object.__setattr__(self, "seam_id", seam_id)
        object.__setattr__(self, "requested", requested)
        object.__setattr__(self, "boundary_uses", boundary_uses)
        object.__setattr__(self, "source", source)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary_uses": [boundary_use.canonical_payload() for boundary_use in self.boundary_uses],
            "requested": self.requested,
            "seam_id": self.seam_id,
            "source": self.source,
            "tolerance_policy": self.tolerance_policy.canonical_payload(),
        }


@dataclass(frozen=True)
class SurfaceContinuityEnforcementRequest:
    """Request for an operation-owned producer to satisfy a continuity constraint."""

    operation_id: str
    producer: str
    constraint: SurfaceSeamContinuityConstraint
    owns_generated_geometry: bool = False
    mutates_source_geometry: bool = False

    def __post_init__(self) -> None:
        operation_id = str(self.operation_id).strip()
        producer = str(self.producer).strip()
        if not operation_id:
            raise ValueError("SurfaceContinuityEnforcementRequest.operation_id must be non-empty.")
        if not producer:
            raise ValueError("SurfaceContinuityEnforcementRequest.producer must be non-empty.")
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "owns_generated_geometry", bool(self.owns_generated_geometry))
        object.__setattr__(self, "mutates_source_geometry", bool(self.mutates_source_geometry))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "constraint": self.constraint.canonical_payload(),
            "mutates_source_geometry": self.mutates_source_geometry,
            "operation_id": self.operation_id,
            "owns_generated_geometry": self.owns_generated_geometry,
            "producer": self.producer,
        }


@dataclass(frozen=True)
class SurfaceContinuityEnforcementRefusalDiagnostic:
    """Explicit reason a continuity enforcement request cannot alter geometry."""

    code: Literal[
        "validation-only",
        "source-mutation-forbidden",
        "invalid-constraint",
        "validation-failed",
    ]
    message: str
    operation_id: str
    seam_id: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "operation_id": self.operation_id,
            "seam_id": self.seam_id,
        }


@dataclass(frozen=True)
class SurfaceContinuityEnforcementResult:
    """Boundary result for operation-owned continuity enforcement."""

    request: SurfaceContinuityEnforcementRequest
    accepted: bool
    validation_report: "SurfaceHigherOrderContinuityValidationReport | None" = None
    diagnostics: tuple[SurfaceContinuityEnforcementRefusalDiagnostic, ...] = ()

    def canonical_payload(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "request": self.request.canonical_payload(),
            "validation_report": None
            if self.validation_report is None
            else self.validation_report.canonical_payload(),
        }


@dataclass(frozen=True)
class PatchFamilyPromotionGapRecord:
    """One missing or unsupported promotion criterion for a patch family."""

    family: str
    criterion: str
    implementation_owner: str
    evidence_type: str
    message: str


@dataclass(frozen=True)
class PatchFamilyPromotionReadinessRecord:
    """Per-family promotion verdict with criterion-level support details."""

    family: str
    current_phase: PatchFamilySupportPhase
    promotable: bool
    supported_criteria: tuple[str, ...]
    gaps: tuple[PatchFamilyPromotionGapRecord, ...]


SUPPORTED_SURFACE_PATCH_FAMILIES: tuple[str, ...] = (
    "planar",
    "ruled",
    "revolution",
    "bspline",
    "nurbs",
    "sweep",
    "subdivision",
    "implicit",
    "heightmap",
    "displacement",
)
REQUIRED_V1_PATCH_FAMILIES: tuple[str, ...] = ("planar", "ruled", "revolution")
ADVANCED_PATCH_FAMILIES: tuple[str, ...] = (
    "bspline",
    "nurbs",
    "sweep",
    "subdivision",
    "implicit",
    "heightmap",
    "displacement",
)
PATCH_FAMILY_AVAILABILITY_REQUIRED_OPERATIONS: tuple[str, ...] = (
    "surface-store",
    "tessellation",
    ".impress",
    "diagnostics",
    "no-hidden-fallback",
)
SURFACE_BODY_COMPLETION_TRACKS: tuple[str, ...] = (
    "patch-family",
    "csg",
    "loft",
    ".impress",
    "primitive",
    "feature",
    "verification",
)
SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS: tuple[SurfaceReferenceFixtureRequirementRecord, ...] = (
    SurfaceReferenceFixtureRequirementRecord("patch-family", ("unit-test", ".impress-roundtrip", "reference-artifact")),
    SurfaceReferenceFixtureRequirementRecord("csg", ("unit-test", "refusal-diagnostic")),
    SurfaceReferenceFixtureRequirementRecord("loft", ("unit-test", "refusal-diagnostic", "reference-artifact")),
    SurfaceReferenceFixtureRequirementRecord(".impress", ("unit-test", ".impress-roundtrip", "refusal-diagnostic")),
    SurfaceReferenceFixtureRequirementRecord("primitive", ("unit-test", "tessellation-artifact")),
    SurfaceReferenceFixtureRequirementRecord("feature", ("unit-test", "handoff-diagnostic")),
    SurfaceReferenceFixtureRequirementRecord("verification", ("unit-test", "reference-artifact")),
)
SURFACE_REFERENCE_ARTIFACT_CLASSES: tuple[SurfaceReferenceArtifactClassRecord, ...] = (
    SurfaceReferenceArtifactClassRecord(
        artifact_class="reference-artifact",
        required_keys=("expected", "actual", "diff"),
        promoted_root="project/reference-artifacts",
        dirty_root="project/reference-artifacts/dirty",
    ),
    SurfaceReferenceArtifactClassRecord(
        artifact_class="tessellation-artifact",
        required_keys=("mesh", "metadata"),
        promoted_root="project/reference-artifacts/stl",
        dirty_root="project/reference-artifacts/dirty/stl",
    ),
    SurfaceReferenceArtifactClassRecord(
        artifact_class="refusal-diagnostic",
        required_keys=("diagnostic",),
        promoted_root="project/reference-artifacts/diagnostics",
        dirty_root="project/reference-artifacts/dirty/diagnostics",
    ),
    SurfaceReferenceArtifactClassRecord(
        artifact_class=".impress-roundtrip",
        required_keys=("source", "decoded", "canonical"),
        promoted_root="project/reference-artifacts/impress",
        dirty_root="project/reference-artifacts/dirty/impress",
    ),
)
SURFACE_REFERENCE_FIXTURE_CONTRACTS: tuple[SurfaceReferenceFixtureContractRecord, ...] = tuple(
    SurfaceReferenceFixtureContractRecord(
        fixture_id=f"surface-{requirement.track}",
        track=requirement.track,
        artifact_classes=requirement.required_evidence_types,
    )
    for requirement in SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS
)
PATCH_FAMILY_PROMOTION_CRITERIA: tuple[str, ...] = (
    "record",
    "evaluator",
    "derivative",
    "seam",
    "tessellation",
    ".impress",
    "csg",
    "loft",
    "diagnostics",
)
SUPPORTED_SEAM_CONTINUITY_CLASSES: tuple[str, ...] = ("C0", "G0")
_PATCH_FAMILY_PROMOTION_OPERATION_ALIASES: dict[str, tuple[str, ...]] = {
    "record": (
        "surface-store",
        "surface-record",
        "rational-surface-record",
        "sweep-record",
        "control-cage",
        "field-node-payload",
        "sample-grid-payload",
        "source-surface-reference",
    ),
    "evaluator": (
        "evaluation",
        "planar-primitives",
        "extrude",
        "rotate-extrude",
        "revolved-primitives",
        "linear-bridge-surfaces",
    ),
    "derivative": ("evaluation", "tessellation"),
    "seam": ("trimmed-faces", "linear-bridge-surfaces", "diagnostics"),
    "tessellation": ("tessellation",),
    ".impress": (".impress",),
    "csg": ("csg", "caps", "planar-primitives", "revolved-primitives"),
    "loft": (
        "loft",
        "loft-non-applicable",
        "loft-refusal",
        "linear-bridge-surfaces",
        "extrude",
        "rotate-extrude",
        "planar-primitives",
    ),
    "diagnostics": ("diagnostics", "validation-security", "no-hidden-fallback"),
}
PATCH_FAMILY_CAPABILITY_MATRIX: dict[str, PatchFamilyCapabilityRecord] = {
    "planar": PatchFamilyCapabilityRecord(
        family="planar",
        support_phase="available",
        operations=(
            "surface-store",
            "caps",
            "planar-primitives",
            "trimmed-faces",
            "tessellation",
            ".impress",
            "diagnostics",
            "no-hidden-fallback",
        ),
    ),
    "ruled": PatchFamilyCapabilityRecord(
        family="ruled",
        support_phase="available",
        operations=(
            "surface-store",
            "extrude",
            "loft",
            "linear-bridge-surfaces",
            "csg",
            "tessellation",
            ".impress",
            "diagnostics",
            "no-hidden-fallback",
        ),
    ),
    "revolution": PatchFamilyCapabilityRecord(
        family="revolution",
        support_phase="available",
        operations=(
            "surface-store",
            "rotate-extrude",
            "revolved-primitives",
            "tessellation",
            ".impress",
            "diagnostics",
            "no-hidden-fallback",
        ),
    ),
    "bspline": PatchFamilyCapabilityRecord(
        family="bspline",
        support_phase="available",
        operations=(
            "surface-store",
            "surface-record",
            "evaluation",
            "tessellation",
            ".impress",
            "csg",
            "loft",
            "diagnostics",
            "no-hidden-fallback",
        ),
    ),
    "nurbs": PatchFamilyCapabilityRecord(
        family="nurbs",
        support_phase="available",
        operations=(
            "surface-store",
            "rational-surface-record",
            "evaluation",
            "tessellation",
            ".impress",
            "csg",
            "loft",
            "diagnostics",
            "no-hidden-fallback",
        ),
    ),
    "sweep": PatchFamilyCapabilityRecord(
        family="sweep",
        support_phase="available",
        operations=(
            "surface-store",
            "sweep-record",
            "frame-policy",
            "evaluation",
            "tessellation",
            ".impress",
            "csg",
            "loft",
            "diagnostics",
            "no-hidden-fallback",
        ),
    ),
    "subdivision": PatchFamilyCapabilityRecord(
        family="subdivision",
        support_phase="available",
        operations=(
            "surface-store",
            "control-cage",
            "crease-payload",
            "evaluation",
            "tessellation",
            ".impress",
            "csg",
            "loft",
            "diagnostics",
            "no-hidden-fallback",
        ),
    ),
    "implicit": PatchFamilyCapabilityRecord(
        family="implicit",
        support_phase="available",
        operations=(
            "surface-store",
            "field-node-payload",
            "validation-security",
            "evaluation",
            "tessellation",
            ".impress",
            "csg",
            "loft-non-applicable",
            "diagnostics",
            "no-hidden-fallback",
        ),
    ),
    "heightmap": PatchFamilyCapabilityRecord(
        family="heightmap",
        support_phase="available",
        operations=(
            "surface-store",
            "sample-grid-payload",
            "evaluation",
            "tessellation",
            ".impress",
            "csg",
            "loft-non-applicable",
            "diagnostics",
            "no-hidden-fallback",
        ),
    ),
    "displacement": PatchFamilyCapabilityRecord(
        family="displacement",
        support_phase="available",
        operations=(
            "surface-store",
            "source-surface-reference",
            "sample-grid-payload",
            "evaluation",
            "tessellation",
            ".impress",
            "csg",
            "loft-non-applicable",
            "diagnostics",
            "no-hidden-fallback",
        ),
    ),
}
SURFACE_SPEC_66_RETIREMENT_NOTE = (
    "Surface Spec 66 is superseded by PATCH_FAMILY_CAPABILITY_MATRIX; no patch family is architecturally deferred."
)
PATCH_FAMILY_FEATURE_COVERAGE: dict[str, tuple[str, ...]] = {
    family: record.operations for family, record in PATCH_FAMILY_CAPABILITY_MATRIX.items()
}


def validate_patch_family_availability_gate(
    family: str,
    record: PatchFamilyCapabilityRecord | None = None,
) -> PatchFamilyAvailabilityGateRecord:
    """Validate one family against the public availability gate."""

    family_key = str(family).strip()
    diagnostics: list[PatchFamilyAvailabilityDiagnostic] = []
    if not family_key:
        diagnostics.append(
            PatchFamilyAvailabilityDiagnostic(
                family=family_key,
                code="empty-family",
                message="Patch family availability checks require a non-empty family key.",
            )
        )
        return PatchFamilyAvailabilityGateRecord(
            family=family_key,
            support_phase="planned",
            available=False,
            diagnostics=tuple(diagnostics),
        )

    capability = record
    if capability is None:
        capability = PATCH_FAMILY_CAPABILITY_MATRIX.get(family_key)
    if capability is None:
        diagnostics.append(
            PatchFamilyAvailabilityDiagnostic(
                family=family_key,
                code="missing-record",
                message=f"Patch family '{family_key}' is missing from PATCH_FAMILY_CAPABILITY_MATRIX.",
            )
        )
        return PatchFamilyAvailabilityGateRecord(
            family=family_key,
            support_phase="planned",
            available=False,
            diagnostics=tuple(diagnostics),
        )

    if family_key not in SUPPORTED_SURFACE_PATCH_FAMILIES:
        diagnostics.append(
            PatchFamilyAvailabilityDiagnostic(
                family=family_key,
                code="unsupported-family",
                message=f"Patch family '{family_key}' is not listed in SUPPORTED_SURFACE_PATCH_FAMILIES.",
            )
        )
    if capability.family != family_key:
        diagnostics.append(
            PatchFamilyAvailabilityDiagnostic(
                family=family_key,
                code="family-mismatch",
                message=(
                    f"Patch family matrix key '{family_key}' does not match record family "
                    f"'{capability.family}'."
                ),
            )
        )
    if capability.support_phase not in ("available", "implemented", "planned"):
        diagnostics.append(
            PatchFamilyAvailabilityDiagnostic(
                family=family_key,
                code="invalid-support-phase",
                message=f"Patch family '{family_key}' has invalid support phase '{capability.support_phase}'.",
            )
        )
    if len(set(capability.operations)) != len(capability.operations):
        diagnostics.append(
            PatchFamilyAvailabilityDiagnostic(
                family=family_key,
                code="duplicate-operation",
                message=f"Patch family '{family_key}' declares duplicate capability operations.",
            )
        )

    if capability.support_phase == "available":
        operation_set = set(capability.operations)
        for operation in PATCH_FAMILY_AVAILABILITY_REQUIRED_OPERATIONS:
            if operation not in operation_set:
                diagnostics.append(
                    PatchFamilyAvailabilityDiagnostic(
                        family=family_key,
                        code="missing-availability-operation",
                        message=(
                            f"Patch family '{family_key}' is available but lacks required "
                            f"integration evidence '{operation}'."
                        ),
                    )
                )
        producer_operations = operation_set.difference(PATCH_FAMILY_AVAILABILITY_REQUIRED_OPERATIONS)
        if not producer_operations:
            diagnostics.append(
                PatchFamilyAvailabilityDiagnostic(
                    family=family_key,
                    code="missing-producer-evidence",
                    message=(
                        f"Patch family '{family_key}' is available but declares no authored "
                        "producer or operation-scoped support evidence."
                    ),
                )
            )

    return PatchFamilyAvailabilityGateRecord(
        family=family_key,
        support_phase=capability.support_phase,
        available=capability.support_phase == "available" and not diagnostics,
        diagnostics=tuple(diagnostics),
    )


def assert_patch_family_capability_matrix() -> tuple[PatchFamilyAvailabilityGateRecord, ...]:
    """Return gate records or raise when the capability matrix overstates support."""

    diagnostics: list[PatchFamilyAvailabilityDiagnostic] = []
    records: list[PatchFamilyAvailabilityGateRecord] = []
    matrix_families = set(PATCH_FAMILY_CAPABILITY_MATRIX)
    supported_families = set(SUPPORTED_SURFACE_PATCH_FAMILIES)
    for family in SUPPORTED_SURFACE_PATCH_FAMILIES:
        gate = validate_patch_family_availability_gate(family)
        records.append(gate)
        diagnostics.extend(gate.diagnostics)
    for family in sorted(matrix_families.difference(supported_families)):
        gate = validate_patch_family_availability_gate(family)
        records.append(gate)
        diagnostics.extend(gate.diagnostics)
    if diagnostics:
        joined = "; ".join(f"{item.family}:{item.code}" for item in diagnostics)
        raise ValueError(f"Patch family capability matrix failed availability gates: {joined}")
    return tuple(records)


def assess_patch_family_availability_promotion(
    family: str,
    record: PatchFamilyCapabilityRecord | None = None,
) -> PatchFamilyPromotionEvidenceRecord:
    """Assess whether a planned family has enough evidence to become available."""

    family_key = str(family).strip()
    capability = record if record is not None else PATCH_FAMILY_CAPABILITY_MATRIX.get(family_key)
    if capability is None:
        gate = validate_patch_family_availability_gate(family_key, None)
        return PatchFamilyPromotionEvidenceRecord(
            family=family_key,
            current_phase="planned",
            promoted_phase="planned",
            operation_support=(),
            diagnostics=gate.diagnostics,
        )

    operation_set = set(capability.operations)
    operation_support = tuple(
        PatchFamilyOperationSupportRecord(
            family=family_key,
            operation=operation,
            supported=operation in operation_set,
            diagnostic="" if operation in operation_set else f"Missing integration evidence '{operation}'.",
        )
        for operation in PATCH_FAMILY_AVAILABILITY_REQUIRED_OPERATIONS
    )
    candidate = replace(capability, support_phase="available")
    gate = validate_patch_family_availability_gate(family_key, candidate)
    promoted_phase: Literal["available", "planned"] = "available" if not gate.diagnostics else "planned"
    return PatchFamilyPromotionEvidenceRecord(
        family=family_key,
        current_phase=capability.support_phase,
        promoted_phase=promoted_phase,
        operation_support=operation_support,
        diagnostics=gate.diagnostics,
    )


def assert_patch_family_operation_coverage(
    family: str,
    required_operations: Iterable[str] = PATCH_FAMILY_AVAILABILITY_REQUIRED_OPERATIONS,
) -> tuple[PatchFamilyOperationSupportRecord, ...]:
    """Return operation support records or raise when required coverage is missing."""

    family_key = str(family).strip()
    capability = PATCH_FAMILY_CAPABILITY_MATRIX.get(family_key)
    if capability is None:
        raise ValueError(f"Patch family '{family_key}' is missing from PATCH_FAMILY_CAPABILITY_MATRIX.")
    operation_set = set(capability.operations)
    records = tuple(
        PatchFamilyOperationSupportRecord(
            family=family_key,
            operation=str(operation),
            supported=str(operation) in operation_set,
            diagnostic="" if str(operation) in operation_set else f"Missing operation coverage '{operation}'.",
        )
        for operation in required_operations
    )
    missing = [record.operation for record in records if not record.supported]
    if missing:
        raise ValueError(f"Patch family '{family_key}' is missing operation coverage: {', '.join(missing)}")
    return records


def run_patch_family_availability_promotion_pass() -> tuple[PatchFamilyPromotionEvidenceRecord, ...]:
    """Assess every known family and report promotable or unpromoted evidence."""

    families = tuple(SUPPORTED_SURFACE_PATCH_FAMILIES) + tuple(
        family for family in sorted(PATCH_FAMILY_CAPABILITY_MATRIX) if family not in SUPPORTED_SURFACE_PATCH_FAMILIES
    )
    return tuple(assess_patch_family_availability_promotion(family) for family in families)


def make_surface_body_completion_evidence_from_capabilities() -> tuple[SurfaceBodyCompletionEvidenceRecord, ...]:
    """Return implementation-backed release evidence derived from current capability records."""

    records: list[SurfaceBodyCompletionEvidenceRecord] = []
    for family, capability in sorted(PATCH_FAMILY_CAPABILITY_MATRIX.items()):
        state: Literal["specified", "implemented", "verified", "unsupported", "retired"]
        state = "verified" if validate_patch_family_availability_gate(family).available else "implemented"
        records.append(
            SurfaceBodyCompletionEvidenceRecord(
                track="patch-family",
                state=state,
                spec=f"patch-family:{family}",
                implementation_owner="src/impression/modeling/surface.py",
                evidence_type="capability-matrix",
                source="implementation",
            )
        )
        if ".impress" in capability.operations:
            records.append(
                SurfaceBodyCompletionEvidenceRecord(
                    track=".impress",
                    state="verified" if state == "verified" else "implemented",
                    spec=f".impress:{family}",
                    implementation_owner="src/impression/io/impress.py",
                    evidence_type="codec-coverage",
                    source="implementation",
                )
            )
        if "tessellation" in capability.operations:
            records.append(
                SurfaceBodyCompletionEvidenceRecord(
                    track="verification",
                    state="implemented",
                    spec=f"tessellation:{family}",
                    implementation_owner="src/impression/modeling/tessellation.py",
                    evidence_type="test-fixture",
                    source="implementation",
                )
            )
    return tuple(records)


SURFACE_BODY_PROMOTED_EVIDENCE_OWNERS: dict[tuple[str, str], tuple[str, str]] = {
    ("patch-family", "unit-test"): ("tests/test_surface.py", "surface-family:unit-test"),
    ("patch-family", ".impress-roundtrip"): ("tests/test_impress_io.py", "surface-family:impress-roundtrip"),
    ("patch-family", "reference-artifact"): ("project/reference-artifacts", "surface-family:reference-artifact"),
    ("csg", "unit-test"): ("tests/test_surface.py", "csg:unit-test"),
    ("csg", "refusal-diagnostic"): ("tests/test_no_hidden_mesh_fallback.py", "csg:refusal-diagnostic"),
    ("loft", "unit-test"): ("tests/test_surface.py", "loft:unit-test"),
    ("loft", "refusal-diagnostic"): ("tests/test_surface.py", "loft:refusal-diagnostic"),
    ("loft", "reference-artifact"): ("project/reference-artifacts", "loft:reference-artifact"),
    (".impress", "unit-test"): ("tests/test_impress_io.py", ".impress:unit-test"),
    (".impress", ".impress-roundtrip"): ("tests/test_impress_io.py", ".impress:roundtrip"),
    (".impress", "refusal-diagnostic"): ("tests/test_impress_io.py", ".impress:refusal-diagnostic"),
    ("primitive", "unit-test"): ("tests/test_surface.py", "primitive:unit-test"),
    ("primitive", "tessellation-artifact"): ("project/reference-artifacts/stl", "primitive:tessellation-artifact"),
    ("feature", "unit-test"): ("tests/test_surface.py", "feature:unit-test"),
    ("feature", "handoff-diagnostic"): ("tests/test_surface.py", "feature:handoff-diagnostic"),
    ("verification", "unit-test"): ("tests/test_surface.py", "verification:unit-test"),
    ("verification", "reference-artifact"): ("project/reference-artifacts", "verification:reference-artifact"),
}


def make_available_family_promoted_reference_evidence(
    requirements: Iterable[SurfaceReferenceFixtureRequirementRecord] = SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS,
) -> tuple[SurfaceBodyCompletionEvidenceRecord, ...]:
    """Return promoted evidence records matching the release reference matrix."""

    records: list[SurfaceBodyCompletionEvidenceRecord] = []
    for requirement in requirements:
        for evidence_type in requirement.required_evidence_types:
            owner, spec = SURFACE_BODY_PROMOTED_EVIDENCE_OWNERS[(requirement.track, evidence_type)]
            records.append(
                SurfaceBodyCompletionEvidenceRecord(
                    track=requirement.track,
                    state="verified",
                    spec=spec,
                    implementation_owner=owner,
                    evidence_type=evidence_type,
                    source="implementation",
                )
            )
    return tuple(records)


def evaluate_surface_body_completion_gate(
    evidence: Iterable[SurfaceBodyCompletionEvidenceRecord] | None = None,
    *,
    required_tracks: Iterable[str] = SURFACE_BODY_COMPLETION_TRACKS,
) -> SurfaceBodyCompletionReport:
    """Evaluate whether explicit non-documentation evidence supports completion claims."""

    evidence_records = tuple(
        make_surface_body_completion_evidence_from_capabilities() if evidence is None else tuple(evidence)
    )
    diagnostics: list[SurfaceBodyCompletionDiagnostic] = []
    by_track: dict[str, list[SurfaceBodyCompletionEvidenceRecord]] = {}
    for record in evidence_records:
        by_track.setdefault(record.track, []).append(record)
        if record.source in {"documentation", "progression", "architecture"}:
            diagnostics.append(
                SurfaceBodyCompletionDiagnostic(
                    track=record.track,
                    code="documentation-only-evidence",
                    spec=record.spec,
                    implementation_owner=record.implementation_owner,
                    evidence_type=record.evidence_type,
                    message=(
                        f"Track '{record.track}' uses {record.source} evidence only; "
                        "completion requires implementation or verification evidence."
                    ),
                )
            )
        if record.state in {"specified", "unsupported"}:
            diagnostics.append(
                SurfaceBodyCompletionDiagnostic(
                    track=record.track,
                    code="incomplete-evidence-state",
                    spec=record.spec,
                    implementation_owner=record.implementation_owner,
                    evidence_type=record.evidence_type,
                    message=(
                        f"Track '{record.track}' evidence '{record.spec}' is {record.state}; "
                        "required evidence must be implemented, verified, or retired."
                    ),
                )
            )

    for track in required_tracks:
        track_key = str(track)
        track_records = by_track.get(track_key, [])
        if not track_records:
            diagnostics.append(
                SurfaceBodyCompletionDiagnostic(
                    track=track_key,
                    code="missing-track-evidence",
                    spec="",
                    implementation_owner="release verification",
                    evidence_type="implementation-or-verification",
                    message=f"Track '{track_key}' has no explicit completion evidence.",
                )
            )
            continue
        if not any(record.state in {"implemented", "verified", "retired"} for record in track_records):
            diagnostics.append(
                SurfaceBodyCompletionDiagnostic(
                    track=track_key,
                    code="missing-implemented-evidence",
                    spec=", ".join(record.spec for record in track_records),
                    implementation_owner=", ".join(sorted({record.implementation_owner for record in track_records})),
                    evidence_type=", ".join(sorted({record.evidence_type for record in track_records})),
                    message=f"Track '{track_key}' has no implemented, verified, or retired evidence.",
                )
            )

    return SurfaceBodyCompletionReport(
        passed=not diagnostics,
        evidence=evidence_records,
        diagnostics=tuple(diagnostics),
    )


def surface_body_completion_reference_evidence_matrix() -> tuple[SurfaceReferenceFixtureRequirementRecord, ...]:
    """Return the required positive and negative evidence matrix for promoted surface output."""

    return SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS


def load_surface_reference_requirement_matrix() -> tuple[SurfaceReferenceFixtureRequirementRecord, ...]:
    """Load the durable surface reference requirement matrix."""

    return surface_body_completion_reference_evidence_matrix()


def surface_reference_artifact_classes() -> tuple[SurfaceReferenceArtifactClassRecord, ...]:
    """Return artifact class contracts used by promoted surface reference evidence."""

    return SURFACE_REFERENCE_ARTIFACT_CLASSES


def surface_reference_fixture_contracts() -> tuple[SurfaceReferenceFixtureContractRecord, ...]:
    """Return fixture contracts derived from the surface reference requirement matrix."""

    return SURFACE_REFERENCE_FIXTURE_CONTRACTS


def assert_surface_reference_requirement_matrix_covers_capabilities(
    requirements: Iterable[SurfaceReferenceFixtureRequirementRecord] = SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS,
    *,
    tracks: Iterable[str] = SURFACE_BODY_COMPLETION_TRACKS,
) -> tuple[SurfaceReferenceFixtureRequirementRecord, ...]:
    """Assert the reference requirement matrix covers each model-outputting track once."""

    requirement_records = tuple(requirements)
    expected_tracks = tuple(str(track).strip() for track in tracks)
    actual_tracks = tuple(record.track for record in requirement_records)
    missing = sorted(set(expected_tracks) - set(actual_tracks))
    unknown = sorted(set(actual_tracks) - set(expected_tracks))
    duplicates = sorted(track for track in set(actual_tracks) if actual_tracks.count(track) > 1)
    if missing or unknown or duplicates:
        messages = []
        if missing:
            messages.append(f"missing reference requirement tracks: {missing!r}")
        if unknown:
            messages.append(f"unknown reference requirement tracks: {unknown!r}")
        if duplicates:
            messages.append(f"duplicate reference requirement tracks: {duplicates!r}")
        raise AssertionError("; ".join(messages))
    return requirement_records


def evaluate_surface_body_completion_reference_evidence_matrix(
    evidence: Iterable[SurfaceBodyCompletionEvidenceRecord],
    *,
    requirements: Iterable[SurfaceReferenceFixtureRequirementRecord] = SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS,
) -> SurfaceReferenceEvidenceMatrixReport:
    """Fail completion when required reference artifacts or diagnostics are missing."""

    evidence_records = tuple(evidence)
    requirement_records = tuple(requirements)
    diagnostics: list[SurfaceBodyCompletionDiagnostic] = []
    by_track_type: set[tuple[str, str]] = {
        (record.track, record.evidence_type)
        for record in evidence_records
        if record.state in {"implemented", "verified", "retired"} and record.source != "dirty-artifact"
    }
    for requirement in requirement_records:
        for evidence_type in requirement.required_evidence_types:
            if (requirement.track, evidence_type) not in by_track_type:
                diagnostics.append(
                    SurfaceBodyCompletionDiagnostic(
                        track=requirement.track,
                        code="missing-reference-evidence",
                        spec="Surface Spec 254",
                        implementation_owner="release verification",
                        evidence_type=evidence_type,
                        message=(
                            f"Track '{requirement.track}' is missing promoted reference evidence "
                            f"of type '{evidence_type}'."
                        ),
                    )
                )
    for record in evidence_records:
        if record.source == "dirty-artifact":
            diagnostics.append(
                SurfaceBodyCompletionDiagnostic(
                    track=record.track,
                    code="dirty-artifact-not-promoted",
                    spec=record.spec,
                    implementation_owner=record.implementation_owner,
                    evidence_type=record.evidence_type,
                    message="Dirty generated artifacts do not satisfy promoted baseline evidence.",
                )
            )
    return SurfaceReferenceEvidenceMatrixReport(
        passed=not diagnostics,
        requirements=requirement_records,
        evidence=evidence_records,
        diagnostics=tuple(diagnostics),
    )


AVAILABLE_FAMILY_PRODUCER_PATH_OPERATIONS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "planar": (("planar-primitives", "producer", "src/impression/modeling/primitives.py"),),
    "ruled": (
        ("extrude", "producer", "src/impression/modeling/_surface_ops.py"),
        ("linear-bridge-surfaces", "producer", "src/impression/modeling/loft.py"),
    ),
    "revolution": (
        ("rotate-extrude", "producer", "src/impression/modeling/_surface_ops.py"),
        ("revolved-primitives", "producer", "src/impression/modeling/primitives.py"),
    ),
    "bspline": (("surface-record", "payload-authoring", "src/impression/modeling/surface.py"),),
    "nurbs": (
        ("rational-surface-record", "payload-authoring", "src/impression/modeling/surface.py"),
        ("exact-conic-producer", "producer", "src/impression/modeling/surface.py"),
    ),
    "sweep": (("sweep-record", "payload-authoring", "src/impression/modeling/surface.py"),),
    "subdivision": (
        ("control-cage", "producer", "src/impression/modeling/surface.py"),
        ("subdivision-cage-import", "import", "src/impression/modeling/surface.py"),
    ),
    "implicit": (("field-node-payload", "payload-authoring", "src/impression/modeling/surface.py"),),
    "heightmap": (
        ("sample-grid-payload", "payload-authoring", "src/impression/modeling/heightmap.py"),
        ("heightmap-import", "import", "src/impression/modeling/heightmap.py"),
    ),
    "displacement": (
        ("source-surface-reference", "payload-authoring", "src/impression/modeling/surface.py"),
        ("sample-grid-payload", "payload-authoring", "src/impression/modeling/surface.py"),
    ),
}


def _availability_family_keys(families: Iterable[str] | None = None) -> tuple[str, ...]:
    source = SUPPORTED_SURFACE_PATCH_FAMILIES if families is None else families
    return tuple(sorted(str(family).strip() for family in source if str(family).strip()))


def build_available_family_missing_evidence_diagnostic(
    family: str,
    category: str,
    *,
    operation: str = "",
    code: str = "missing-operation-row",
    message: str | None = None,
) -> AvailableFamilyMissingEvidenceDiagnostic:
    """Build a deterministic missing-evidence diagnostic for availability operation rows."""

    family_key = str(family).strip()
    category_key = str(category).strip()
    operation_key = str(operation).strip()
    if not family_key or not category_key:
        raise ValueError("Available-family missing evidence diagnostics require family and category.")
    resolved_message = message or (
        f"Patch family '{family_key}' is missing {category_key} evidence"
        + (f" for operation '{operation_key}'." if operation_key else ".")
    )
    return AvailableFamilyMissingEvidenceDiagnostic(
        family=family_key,
        category=category_key,
        operation=operation_key,
        code=str(code).strip() or "missing-operation-row",
        message=resolved_message,
    )


def available_family_producer_path_rows(
    families: Iterable[str] | None = None,
) -> tuple[AvailableFamilyOperationEvidenceRecord, ...]:
    """Return producer/import/payload operation rows for each supported family."""

    rows: list[AvailableFamilyOperationEvidenceRecord] = []
    for family in _availability_family_keys(families):
        capability = PATCH_FAMILY_CAPABILITY_MATRIX.get(family)
        operation_set = set(capability.operations) if capability is not None else set()
        for operation, row_category, source in AVAILABLE_FAMILY_PRODUCER_PATH_OPERATIONS.get(family, ()):
            supported = operation in operation_set or row_category == "import"
            rows.append(
                AvailableFamilyOperationEvidenceRecord(
                    family=family,
                    operation=operation,
                    category="producer-path",
                    supported=supported,
                    state=(
                        "import"
                        if row_category == "import" and supported
                        else "payload"
                        if row_category == "payload-authoring" and supported
                        else "native"
                        if supported
                        else "missing"
                    ),
                    source=source,
                    diagnostic=""
                    if supported
                    else f"Patch family '{family}' does not declare producer path operation '{operation}'.",
                )
            )
    return tuple(rows)


def summarize_available_family_producer_paths(
    rows: Iterable[AvailableFamilyOperationEvidenceRecord] | None = None,
    *,
    families: Iterable[str] | None = None,
) -> dict[str, tuple[AvailableFamilyOperationEvidenceRecord, ...]]:
    """Group producer/import/payload rows by family for report builders."""

    row_tuple = tuple(available_family_producer_path_rows(families) if rows is None else tuple(rows))
    grouped: dict[str, list[AvailableFamilyOperationEvidenceRecord]] = {
        family: [] for family in _availability_family_keys(families)
    }
    for row in row_tuple:
        grouped.setdefault(row.family, []).append(row)
    return {family: tuple(grouped[family]) for family in sorted(grouped)}


def verify_available_family_producer_path_rows(
    rows: Iterable[AvailableFamilyOperationEvidenceRecord] | None = None,
    *,
    families: Iterable[str] | None = None,
) -> AvailableFamilyOperationCompletenessReport:
    """Verify that every supported family has an explicit producer/import/payload path."""

    row_tuple = tuple(available_family_producer_path_rows(families) if rows is None else tuple(rows))
    diagnostics: list[AvailableFamilyMissingEvidenceDiagnostic] = []
    grouped = summarize_available_family_producer_paths(row_tuple, families=families)
    for family, family_rows in grouped.items():
        if not family_rows:
            diagnostics.append(build_available_family_missing_evidence_diagnostic(family, "producer-path"))
            continue
        if not any(row.supported for row in family_rows):
            diagnostics.append(
                build_available_family_missing_evidence_diagnostic(
                    family,
                    "producer-path",
                    code="missing-producer-path",
                    message=(
                        f"Patch family '{family}' has producer-path rows but no supported "
                        "producer, import, or payload-authoring route."
                    ),
                )
            )
    return AvailableFamilyOperationCompletenessReport(
        category="producer-path",
        passed=not diagnostics,
        rows=row_tuple,
        diagnostics=tuple(diagnostics),
    )


def available_family_storage_tessellation_rows(
    families: Iterable[str] | None = None,
) -> tuple[AvailableFamilyOperationEvidenceRecord, ...]:
    """Return `.impress` and tessellation rows for each supported family."""

    from impression.io.impress import inspect_impress_patch_codec_coverage
    from impression.modeling.tessellation import inspect_surface_family_tessellation_adapter_coverage

    requested_families = set(_availability_family_keys(families))
    codec_by_family = {
        record.family: record for record in inspect_impress_patch_codec_coverage() if record.family in requested_families
    }
    tessellation_by_family = {
        record.family: record
        for record in inspect_surface_family_tessellation_adapter_coverage()
        if record.family in requested_families
    }
    rows: list[AvailableFamilyOperationEvidenceRecord] = []
    for family in _availability_family_keys(families):
        codec = codec_by_family.get(family)
        codec_supported = bool(codec and codec.covered)
        rows.append(
            AvailableFamilyOperationEvidenceRecord(
                family=family,
                operation=".impress",
                category="storage-tessellation",
                supported=codec_supported,
                state="native" if codec_supported else "missing",
                source="src/impression/io/impress.py",
                diagnostic="" if codec_supported else f"Patch family '{family}' is missing `.impress` codec coverage.",
            )
        )
        tessellation = tessellation_by_family.get(family)
        tessellation_supported = bool(tessellation and tessellation.covered and tessellation.metadata_traceable)
        rows.append(
            AvailableFamilyOperationEvidenceRecord(
                family=family,
                operation="tessellation",
                category="storage-tessellation",
                supported=tessellation_supported,
                state="adapter" if tessellation_supported else "missing",
                source="src/impression/modeling/tessellation.py",
                diagnostic=""
                if tessellation_supported
                else f"Patch family '{family}' is missing traceable tessellation adapter coverage.",
            )
        )
    return tuple(rows)


def verify_available_family_storage_tessellation_rows(
    rows: Iterable[AvailableFamilyOperationEvidenceRecord] | None = None,
    *,
    families: Iterable[str] | None = None,
) -> AvailableFamilyOperationCompletenessReport:
    """Verify that every supported family has `.impress` and tessellation operation rows."""

    row_tuple = tuple(available_family_storage_tessellation_rows(families) if rows is None else tuple(rows))
    diagnostics: list[AvailableFamilyMissingEvidenceDiagnostic] = []
    grouped: dict[tuple[str, str], AvailableFamilyOperationEvidenceRecord] = {
        (row.family, row.operation): row for row in row_tuple
    }
    for family in _availability_family_keys(families):
        for operation in (".impress", "tessellation"):
            row = grouped.get((family, operation))
            if row is None:
                diagnostics.append(
                    build_available_family_missing_evidence_diagnostic(
                        family,
                        "storage-tessellation",
                        operation=operation,
                    )
                )
            elif not row.supported:
                diagnostics.append(
                    build_available_family_missing_evidence_diagnostic(
                        family,
                        "storage-tessellation",
                        operation=operation,
                        code="unsupported-operation-row",
                        message=row.diagnostic,
                    )
                )
    return AvailableFamilyOperationCompletenessReport(
        category="storage-tessellation",
        passed=not diagnostics,
        rows=row_tuple,
        diagnostics=tuple(diagnostics),
    )


def available_family_seam_loft_rows(
    families: Iterable[str] | None = None,
) -> tuple[AvailableFamilyOperationEvidenceRecord, ...]:
    """Return explicit seam and loft rows for each supported patch family."""

    seam_by_family = {record.family: record for record in surface_family_boundary_support_matrix()}
    rows: list[AvailableFamilyOperationEvidenceRecord] = []
    loft_aliases = set(_PATCH_FAMILY_PROMOTION_OPERATION_ALIASES["loft"])
    for family in _availability_family_keys(families):
        capability = PATCH_FAMILY_CAPABILITY_MATRIX.get(family)
        operation_set = set(capability.operations) if capability is not None else set()
        seam = seam_by_family.get(family)
        seam_supported = bool(seam and seam.boundary_support != "unsupported")
        rows.append(
            AvailableFamilyOperationEvidenceRecord(
                family=family,
                operation="seam",
                category="seam-loft",
                supported=seam_supported,
                state="native" if seam_supported and seam and seam.boundary_support == "exact" else "adapter" if seam_supported else "missing",
                source="src/impression/modeling/surface.py",
                diagnostic="" if seam_supported else f"Patch family '{family}' has no supported seam boundary row.",
            )
        )
        loft_row_present = bool(operation_set.intersection(loft_aliases))
        loft_supported = loft_row_present and "loft-non-applicable" not in operation_set
        rows.append(
            AvailableFamilyOperationEvidenceRecord(
                family=family,
                operation="loft",
                category="seam-loft",
                supported=loft_supported,
                state="native" if loft_supported else "explicit" if loft_row_present else "missing",
                source="src/impression/modeling/loft.py",
                diagnostic="" if loft_supported else f"Patch family '{family}' is explicit non-applicable or unsupported for loft.",
            )
        )
        no_fallback_supported = family in SUPPORTED_SURFACE_PATCH_FAMILIES
        rows.append(
            AvailableFamilyOperationEvidenceRecord(
                family=family,
                operation="no-hidden-fallback",
                category="seam-loft",
                supported=no_fallback_supported,
                state="explicit" if no_fallback_supported else "missing",
                source="src/impression/modeling/surface.py",
                diagnostic="" if no_fallback_supported else f"Patch family '{family}' lacks no-hidden-fallback evidence.",
            )
        )
    return tuple(rows)


def build_available_family_no_hidden_mesh_fallback_diagnostic(
    family: str,
    operation: str,
    message: str | None = None,
) -> AvailableFamilyMissingEvidenceDiagnostic:
    """Build a no-hidden-mesh-fallback diagnostic for operation-row gates."""

    family_key = str(family).strip()
    operation_key = str(operation).strip()
    return build_available_family_missing_evidence_diagnostic(
        family_key,
        "no-hidden-mesh-fallback",
        operation=operation_key,
        code="missing-no-hidden-mesh-fallback",
        message=message
        or f"Patch family '{family_key}' lacks no-hidden-mesh-fallback evidence for '{operation_key}'.",
    )


def verify_available_family_seam_loft_rows(
    rows: Iterable[AvailableFamilyOperationEvidenceRecord] | None = None,
    *,
    families: Iterable[str] | None = None,
) -> AvailableFamilyOperationCompletenessReport:
    """Verify seam, loft/non-applicable, and no-hidden-fallback operation rows."""

    row_tuple = tuple(available_family_seam_loft_rows(families) if rows is None else tuple(rows))
    diagnostics: list[AvailableFamilyMissingEvidenceDiagnostic] = []
    grouped: dict[tuple[str, str], AvailableFamilyOperationEvidenceRecord] = {
        (row.family, row.operation): row for row in row_tuple
    }
    for family in _availability_family_keys(families):
        for operation in ("seam", "loft", "no-hidden-fallback"):
            row = grouped.get((family, operation))
            if row is None:
                diagnostics.append(
                    build_available_family_missing_evidence_diagnostic(
                        family,
                        "seam-loft",
                        operation=operation,
                    )
                )
            elif operation == "no-hidden-fallback" and not row.supported:
                diagnostics.append(build_available_family_no_hidden_mesh_fallback_diagnostic(family, operation, row.diagnostic))
            elif operation not in {"loft", "seam"} and not row.supported:
                diagnostics.append(
                    build_available_family_missing_evidence_diagnostic(
                        family,
                        "seam-loft",
                        operation=operation,
                        code="unsupported-operation-row",
                        message=row.diagnostic,
                    )
                )
    return AvailableFamilyOperationCompletenessReport(
        category="seam-loft",
        passed=not diagnostics,
        rows=row_tuple,
        diagnostics=tuple(diagnostics),
    )


def collect_available_family_reference_evidence(
    evidence: Iterable[SurfaceBodyCompletionEvidenceRecord],
    *,
    requirements: Iterable[SurfaceReferenceFixtureRequirementRecord] = SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS,
) -> tuple[AvailableFamilyReferenceEvidenceSummary, ...]:
    """Summarize promoted and dirty reference evidence for availability completion."""

    evidence_records = tuple(evidence)
    summaries: list[AvailableFamilyReferenceEvidenceSummary] = []
    for requirement in requirements:
        promoted = tuple(
            sorted(
                {
                    record.evidence_type
                    for record in evidence_records
                    if record.track == requirement.track
                    and record.evidence_type in requirement.required_evidence_types
                    and record.state in {"implemented", "verified", "retired"}
                    and record.source != "dirty-artifact"
                }
            )
        )
        dirty = tuple(
            sorted(
                {
                    record.evidence_type
                    for record in evidence_records
                    if record.track == requirement.track
                    and record.evidence_type in requirement.required_evidence_types
                    and record.source == "dirty-artifact"
                }
            )
        )
        summaries.append(
            AvailableFamilyReferenceEvidenceSummary(
                track=requirement.track,
                required_evidence_types=requirement.required_evidence_types,
                satisfied_evidence_types=promoted,
                dirty_evidence_types=dirty,
            )
        )
    return tuple(summaries)


def build_dirty_available_family_reference_diagnostic(
    record: SurfaceBodyCompletionEvidenceRecord,
) -> AvailableFamilyMissingEvidenceDiagnostic:
    """Build a deterministic diagnostic for dirty reference artifacts."""

    return build_available_family_missing_evidence_diagnostic(
        record.track,
        "reference-evidence",
        operation=record.evidence_type,
        code="dirty-reference-artifact",
        message=(
            f"Dirty reference artifact '{record.spec}' for track '{record.track}' "
            "does not satisfy promoted availability evidence."
        ),
    )


def evaluate_available_family_reference_evidence_gate(
    evidence: Iterable[SurfaceBodyCompletionEvidenceRecord],
    *,
    requirements: Iterable[SurfaceReferenceFixtureRequirementRecord] = SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS,
) -> AvailableFamilyReferenceEvidenceReport:
    """Evaluate promoted reference evidence and refuse dirty artifacts."""

    evidence_records = tuple(evidence)
    summaries = collect_available_family_reference_evidence(evidence_records, requirements=requirements)
    diagnostics: list[AvailableFamilyMissingEvidenceDiagnostic] = []
    for summary in summaries:
        for evidence_type in summary.missing_evidence_types:
            diagnostics.append(
                build_available_family_missing_evidence_diagnostic(
                    summary.track,
                    "reference-evidence",
                    operation=evidence_type,
                    code="missing-reference-evidence",
                    message=(
                        f"Track '{summary.track}' is missing promoted reference evidence "
                        f"of type '{evidence_type}'."
                    ),
                )
            )
    for record in evidence_records:
        if record.source == "dirty-artifact":
            diagnostics.append(build_dirty_available_family_reference_diagnostic(record))
    return AvailableFamilyReferenceEvidenceReport(
        passed=not diagnostics,
        summaries=summaries,
        diagnostics=tuple(diagnostics),
    )


def summarize_available_family_missing_evidence(
    reports: Iterable[AvailableFamilyOperationCompletenessReport],
    reference_report: AvailableFamilyReferenceEvidenceReport,
) -> tuple[AvailableFamilyMissingEvidenceDiagnostic, ...]:
    """Flatten operation and reference diagnostics for completion reporting."""

    diagnostics: list[AvailableFamilyMissingEvidenceDiagnostic] = []
    for report in reports:
        diagnostics.extend(report.diagnostics)
    diagnostics.extend(reference_report.diagnostics)
    return tuple(diagnostics)


def build_available_family_completion_report(
    *,
    evidence: Iterable[SurfaceBodyCompletionEvidenceRecord] | None = None,
    families: Iterable[str] | None = None,
) -> AvailableFamilyCompletionReport:
    """Build the deterministic surface-family availability completion report."""

    family_keys = _availability_family_keys(families)
    operation_reports = (
        verify_available_family_producer_path_rows(families=family_keys),
        verify_available_family_storage_tessellation_rows(families=family_keys),
        verify_available_family_seam_loft_rows(families=family_keys),
    )
    from impression.modeling.csg import (
        verify_available_family_csg_classification_rows,
        verify_surface_csg_no_mesh_fallback_evidence,
    )

    csg_report = verify_available_family_csg_classification_rows(families=family_keys)
    csg_no_fallback_report = verify_surface_csg_no_mesh_fallback_evidence(families=family_keys)
    reference_records = tuple(make_available_family_promoted_reference_evidence() if evidence is None else tuple(evidence))
    reference_report = evaluate_available_family_reference_evidence_gate(reference_records)
    diagnostics = list(summarize_available_family_missing_evidence(operation_reports, reference_report))
    if not csg_report.passed:
        for diagnostic in csg_report.diagnostics:
            diagnostics.append(
                build_available_family_missing_evidence_diagnostic(
                    "csg",
                    "csg-classification",
                    operation=str(diagnostic.operation or ""),
                    code=diagnostic.code,
                    message=diagnostic.message,
                )
            )
    if not csg_no_fallback_report.passed:
        for diagnostic in csg_no_fallback_report.diagnostics:
            diagnostics.append(
                build_available_family_missing_evidence_diagnostic(
                    "csg",
                    "csg-no-mesh-fallback",
                    operation=diagnostic.operation,
                    code=diagnostic.code,
                    message=diagnostic.message,
                )
            )
    return AvailableFamilyCompletionReport(
        passed=(
            all(report.passed for report in operation_reports)
            and csg_report.passed
            and csg_no_fallback_report.passed
            and reference_report.passed
        ),
        families=family_keys,
        operation_reports=operation_reports,
        reference_report=reference_report,
        diagnostics=tuple(diagnostics),
    )


def snapshot_available_family_completion_report(
    report: AvailableFamilyCompletionReport,
) -> AvailableFamilyCompletionSnapshot:
    """Return a compact stable snapshot suitable for release evidence fixtures."""

    return AvailableFamilyCompletionSnapshot(
        families=report.families,
        passed=report.passed,
        operation_categories=tuple(report.category for report in report.operation_reports),
        diagnostic_count=len(report.diagnostics),
    )


def audit_patch_family_promotion_readiness(
    family: str,
    record: PatchFamilyCapabilityRecord | None = None,
) -> PatchFamilyPromotionReadinessRecord:
    """Audit a patch family against promotion criteria without mutating support phase."""

    family_key = str(family).strip()
    capability = record if record is not None else PATCH_FAMILY_CAPABILITY_MATRIX.get(family_key)
    if capability is None:
        gap = PatchFamilyPromotionGapRecord(
            family=family_key,
            criterion="record",
            implementation_owner="src/impression/modeling/surface.py",
            evidence_type="capability-matrix",
            message=f"Patch family '{family_key}' is missing from PATCH_FAMILY_CAPABILITY_MATRIX.",
        )
        return PatchFamilyPromotionReadinessRecord(
            family=family_key,
            current_phase="planned",
            promotable=False,
            supported_criteria=(),
            gaps=(gap,),
        )

    operations = set(capability.operations)
    supported: list[str] = []
    gaps: list[PatchFamilyPromotionGapRecord] = []
    for criterion in PATCH_FAMILY_PROMOTION_CRITERIA:
        aliases = _PATCH_FAMILY_PROMOTION_OPERATION_ALIASES[criterion]
        if operations.intersection(aliases):
            supported.append(criterion)
            continue
        gaps.append(
            PatchFamilyPromotionGapRecord(
                family=family_key,
                criterion=criterion,
                implementation_owner=_promotion_owner_for_criterion(criterion),
                evidence_type=f"{criterion}-coverage",
                message=(
                    f"Patch family '{family_key}' lacks promotion evidence for '{criterion}'. "
                    f"Expected one of: {', '.join(aliases)}."
                ),
            )
        )

    return PatchFamilyPromotionReadinessRecord(
        family=family_key,
        current_phase=capability.support_phase,
        promotable=not gaps,
        supported_criteria=tuple(supported),
        gaps=tuple(gaps),
    )


def audit_all_patch_family_promotion_readiness() -> tuple[PatchFamilyPromotionReadinessRecord, ...]:
    """Audit every supported authored patch family for promotion readiness."""

    return tuple(audit_patch_family_promotion_readiness(family) for family in SUPPORTED_SURFACE_PATCH_FAMILIES)


def evaluate_advanced_patch_family_promotion_gate(
    family: str,
    record: PatchFamilyCapabilityRecord | None = None,
) -> AdvancedPatchFamilyPromotionEvidenceRecord:
    """Evaluate whether one advanced family has enough evidence to be implemented."""

    family_key = str(family).strip()
    readiness = audit_patch_family_promotion_readiness(family_key, record)
    capability = record if record is not None else PATCH_FAMILY_CAPABILITY_MATRIX.get(family_key)
    current_phase: PatchFamilySupportPhase = capability.support_phase if capability is not None else "planned"

    diagnostics = list(readiness.gaps)
    if family_key not in ADVANCED_PATCH_FAMILIES:
        diagnostics.append(
            PatchFamilyPromotionGapRecord(
                family=family_key,
                criterion="advanced-family",
                implementation_owner="src/impression/modeling/surface.py",
                evidence_type="capability-matrix",
                message=f"Patch family '{family_key}' is not an advanced patch family promotion target.",
            )
        )

    promoted_phase: PatchFamilySupportPhase
    if current_phase == "available":
        promoted_phase = "available"
    elif diagnostics:
        promoted_phase = "planned"
    else:
        promoted_phase = "implemented"

    return AdvancedPatchFamilyPromotionEvidenceRecord(
        family=family_key,
        current_phase=current_phase,
        promoted_phase=promoted_phase,
        readiness=readiness,
        diagnostics=tuple(diagnostics),
    )


def run_advanced_patch_family_promotion_gate(
    families: Iterable[str] = ADVANCED_PATCH_FAMILIES,
) -> AdvancedPatchFamilyPromotionReport:
    """Evaluate the advanced-family implementation gate without mutating the matrix."""

    evidence = tuple(evaluate_advanced_patch_family_promotion_gate(family) for family in families)
    diagnostics = tuple(diagnostic for record in evidence for diagnostic in record.diagnostics)
    return AdvancedPatchFamilyPromotionReport(
        passed=not diagnostics,
        evidence=evidence,
        diagnostics=diagnostics,
    )


def assert_advanced_patch_family_promotion_gate(
    family: str,
    record: PatchFamilyCapabilityRecord | None = None,
) -> AdvancedPatchFamilyPromotionEvidenceRecord:
    """Return promotion evidence or raise with all missing implementation criteria."""

    evidence = evaluate_advanced_patch_family_promotion_gate(family, record)
    if evidence.diagnostics:
        missing = ", ".join(f"{diagnostic.criterion}: {diagnostic.message}" for diagnostic in evidence.diagnostics)
        raise ValueError(f"Patch family '{evidence.family}' cannot be promoted to implemented: {missing}")
    return evidence


def _promotion_owner_for_criterion(criterion: str) -> str:
    owners = {
        "record": "src/impression/modeling/surface.py",
        "evaluator": "src/impression/modeling/surface.py",
        "derivative": "src/impression/modeling/surface.py",
        "seam": "src/impression/modeling/surface.py",
        "tessellation": "src/impression/modeling/tessellation.py",
        ".impress": "src/impression/io/impress.py",
        "csg": "src/impression/modeling/csg.py",
        "loft": "src/impression/modeling/loft.py",
        "diagnostics": "src/impression/modeling/surface.py",
    }
    return owners.get(criterion, "release verification")


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


def validate_nurbs_weights(
    weights: Sequence[Sequence[float]] | np.ndarray,
    *,
    control_net_shape: tuple[int, int],
) -> tuple[NURBSWeightValidationDiagnostic, ...]:
    """Return deterministic diagnostics for malformed NURBS weight payloads."""

    diagnostics: list[NURBSWeightValidationDiagnostic] = []
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.shape != tuple(control_net_shape):
        diagnostics.append(
            NURBSWeightValidationDiagnostic(
                code="weight-shape-mismatch",
                message="NURBS weights must match the control net parameter shape.",
                shape=tuple(int(value) for value in weight_array.shape),
            )
        )
        return tuple(diagnostics)
    if not np.all(np.isfinite(weight_array)):
        diagnostics.append(
            NURBSWeightValidationDiagnostic(
                code="nonfinite-weight",
                message="NURBS weights must be finite.",
                shape=tuple(int(value) for value in weight_array.shape),
            )
        )
    if np.any(weight_array <= 0.0):
        diagnostics.append(
            NURBSWeightValidationDiagnostic(
                code="nonpositive-weight",
                message="NURBS weights must be positive.",
                shape=tuple(int(value) for value in weight_array.shape),
            )
        )
    return tuple(diagnostics)


def assert_valid_nurbs_weights(
    weights: Sequence[Sequence[float]] | np.ndarray,
    *,
    control_net_shape: tuple[int, int],
) -> np.ndarray:
    """Canonicalize NURBS weights or raise with all deterministic diagnostics."""

    weight_array = np.asarray(weights, dtype=float)
    diagnostics = validate_nurbs_weights(weight_array, control_net_shape=control_net_shape)
    if diagnostics:
        details = "; ".join(f"{diagnostic.code}: {diagnostic.message}" for diagnostic in diagnostics)
        raise ValueError(f"NURBSSurfacePatch weights are invalid: {details}")
    return weight_array


def _conic_diagnostic(code: str, message: str, conic_kind: str) -> NURBSConicConstructionDiagnostic:
    return NURBSConicConstructionDiagnostic(code=code, message=message, conic_kind=conic_kind)


def _empty_conic_payload(
    request: NURBSConicConstructionRequest,
    diagnostics: Sequence[NURBSConicConstructionDiagnostic],
) -> NURBSConicProfilePayload:
    return NURBSConicProfilePayload(
        degree=2,
        control_points_uv=np.zeros((0, 2), dtype=float),
        weights=np.zeros((0,), dtype=float),
        knots=(),
        metadata={"conic_kind": request.conic_kind, **request.metadata},
        diagnostics=tuple(diagnostics),
    )


def build_nurbs_circular_arc_control_net(
    *,
    center: Sequence[float] = (0.0, 0.0),
    radius: float = 1.0,
    start_angle_deg: float = 0.0,
    end_angle_deg: float = 90.0,
) -> NURBSConicProfilePayload:
    """Build one exact rational quadratic circular arc payload."""

    request = NURBSConicConstructionRequest(
        conic_kind="arc",
        center=tuple(float(value) for value in center),
        radius=radius,
        start_angle_deg=start_angle_deg,
        end_angle_deg=end_angle_deg,
    )
    return build_nurbs_exact_conic_profile_payload(request)


def build_nurbs_exact_conic_profile_payload(request: NURBSConicConstructionRequest) -> NURBSConicProfilePayload:
    """Build exact rational quadratic payloads for supported circle, ellipse, and arc requests."""

    diagnostics: list[NURBSConicConstructionDiagnostic] = []
    kind = request.conic_kind
    if kind not in {"arc", "circle", "ellipse"}:
        diagnostics.append(
            _conic_diagnostic(
                "unsupported-conic-kind",
                "Only exact circular arcs, circles, and ellipses are supported by this helper.",
                kind,
            )
        )
        return _empty_conic_payload(request, diagnostics)

    if len(request.center) != 2 or not all(np.isfinite(value) for value in request.center):
        diagnostics.append(_conic_diagnostic("invalid-center", "Conic center must contain two finite values.", kind))
    if kind == "ellipse":
        radii = request.radii
        if radii is None and request.radius is not None:
            radii = (request.radius, request.radius)
        if radii is None or len(radii) != 2 or not all(np.isfinite(value) and value > 0.0 for value in radii):
            diagnostics.append(_conic_diagnostic("invalid-radii", "Ellipse radii must contain two positive finite values.", kind))
            radii = (1.0, 1.0)
    else:
        if request.radius is None or not np.isfinite(request.radius) or request.radius <= 0.0:
            diagnostics.append(_conic_diagnostic("invalid-radius", "Circle and arc radius must be positive and finite.", kind))
        radius = 1.0 if request.radius is None else float(request.radius)
        radii = (radius, radius)
    if not np.isfinite(request.start_angle_deg) or not np.isfinite(request.end_angle_deg):
        diagnostics.append(_conic_diagnostic("invalid-angle", "Conic angles must be finite.", kind))
    angle_span = float(request.end_angle_deg - request.start_angle_deg)
    if kind in {"circle", "ellipse"} and np.isclose(angle_span, 0.0):
        angle_span = 360.0
    if np.isclose(angle_span, 0.0):
        diagnostics.append(_conic_diagnostic("zero-angle-span", "Conic angle span must be non-zero.", kind))
    if diagnostics:
        return _empty_conic_payload(request, diagnostics)

    segment_count = max(1, int(np.ceil(abs(angle_span) / 90.0)))
    segment_span = np.deg2rad(angle_span / segment_count)
    start = np.deg2rad(request.start_angle_deg)
    center = np.asarray(request.center, dtype=float)
    rx, ry = radii
    points: list[np.ndarray] = []
    weights: list[float] = []
    for segment_index in range(segment_count):
        a0 = start + segment_span * segment_index
        a1 = a0 + segment_span
        amid = (a0 + a1) / 2.0
        half = (a1 - a0) / 2.0
        weight = float(np.cos(half))
        if weight <= 0.0:
            diagnostics.append(_conic_diagnostic("unsupported-angle-span", "Conic segments must be at most 180 degrees.", kind))
            return _empty_conic_payload(request, diagnostics)
        p0 = center + np.array([rx * np.cos(a0), ry * np.sin(a0)], dtype=float)
        p1 = center + np.array([rx * np.cos(amid) / weight, ry * np.sin(amid) / weight], dtype=float)
        p2 = center + np.array([rx * np.cos(a1), ry * np.sin(a1)], dtype=float)
        if segment_index == 0:
            points.append(p0)
            weights.append(1.0)
        points.extend((p1, p2))
        weights.extend((weight, 1.0))

    internal_knots: list[float] = []
    for index in range(1, segment_count):
        value = index / segment_count
        internal_knots.extend((value, value))
    knots = (0.0, 0.0, 0.0, *internal_knots, 1.0, 1.0, 1.0)
    metadata = {
        "conic_kind": kind,
        "exact_rational_conic": True,
        "segment_count": segment_count,
        "start_angle_deg": request.start_angle_deg,
        "end_angle_deg": request.end_angle_deg,
        **request.metadata,
    }
    return NURBSConicProfilePayload(
        degree=2,
        control_points_uv=np.asarray(points, dtype=float),
        weights=np.asarray(weights, dtype=float),
        knots=tuple(float(value) for value in knots),
        metadata=metadata,
    )


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


@dataclass(frozen=True)
class PathFrameDegeneracyDiagnostic:
    """Diagnostic emitted when a path frame cannot be evaluated cleanly."""

    code: str
    message: str
    parameter: float

    def canonical_payload(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "parameter": self.parameter}


@dataclass(frozen=True)
class FrameTransportPolicyRecord:
    """Deterministic path-frame transport, twist, and scale policy."""

    policy: FrameTransportPolicy = "parallel_transport"
    twist_degrees: tuple[float, float] = (0.0, 0.0)
    scale: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        policy = str(self.policy)
        if policy not in {"parallel_transport", "frenet", "fixed"}:
            raise ValueError("FrameTransportPolicyRecord.policy must be parallel_transport, frenet, or fixed.")
        twist = tuple(float(value) for value in self.twist_degrees)
        scale = tuple(float(value) for value in self.scale)
        if len(twist) != 2 or not all(np.isfinite(value) for value in twist):
            raise ValueError("FrameTransportPolicyRecord.twist_degrees must contain two finite values.")
        if len(scale) != 2 or not all(np.isfinite(value) and value > 0.0 for value in scale):
            raise ValueError("FrameTransportPolicyRecord.scale must contain two positive finite values.")
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "twist_degrees", twist)
        object.__setattr__(self, "scale", scale)

    def canonical_payload(self) -> dict[str, object]:
        return {"policy": self.policy, "twist_degrees": self.twist_degrees, "scale": self.scale}


@dataclass(frozen=True)
class PathFrameSampleRecord:
    """One evaluated path frame sample."""

    parameter: float
    point: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    w_axis: np.ndarray
    twist_degrees: float
    scale: float
    diagnostics: tuple[PathFrameDegeneracyDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", np.asarray(self.point, dtype=float).reshape(3))
        object.__setattr__(self, "u_axis", np.asarray(self.u_axis, dtype=float).reshape(3))
        object.__setattr__(self, "v_axis", np.asarray(self.v_axis, dtype=float).reshape(3))
        object.__setattr__(self, "w_axis", np.asarray(self.w_axis, dtype=float).reshape(3))
        object.__setattr__(self, "parameter", float(self.parameter))
        object.__setattr__(self, "twist_degrees", float(self.twist_degrees))
        object.__setattr__(self, "scale", float(self.scale))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "point": self.point.tolist(),
            "u_axis": self.u_axis.tolist(),
            "v_axis": self.v_axis.tolist(),
            "w_axis": self.w_axis.tolist(),
            "twist_degrees": self.twist_degrees,
            "scale": self.scale,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


def build_path_frame_degeneracy_diagnostic(
    code: str,
    message: str,
    *,
    parameter: float,
) -> PathFrameDegeneracyDiagnostic:
    return PathFrameDegeneracyDiagnostic(code=str(code), message=str(message), parameter=float(parameter))


def interpolate_path_twist_scale(policy: FrameTransportPolicyRecord, parameter: float) -> tuple[float, float]:
    t = float(np.clip(parameter, 0.0, 1.0))
    twist = policy.twist_degrees[0] + (policy.twist_degrees[1] - policy.twist_degrees[0]) * t
    scale = policy.scale[0] + (policy.scale[1] - policy.scale[0]) * t
    return float(twist), float(scale)


def _rotate_vector_about_axis(vector: np.ndarray, axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(float(angle_degrees))
    axis = _normalize_axis(axis, name="rotation_axis")
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * float(np.dot(axis, vector)) * (1.0 - np.cos(angle))
    )


def evaluate_path_frame(
    path: Path3D,
    parameter: float,
    policy: FrameTransportPolicyRecord | FrameTransportPolicy | None = None,
) -> PathFrameSampleRecord:
    """Evaluate a deterministic authored path frame sample."""

    if not isinstance(path, Path3D):
        raise ValueError("path must be a Path3D.")
    policy_record = policy if isinstance(policy, FrameTransportPolicyRecord) else FrameTransportPolicyRecord(policy or "parallel_transport")
    t = float(np.clip(parameter, 0.0, 1.0))
    path_points = path.sample()
    if path_points.shape[0] < 2:
        diagnostic = build_path_frame_degeneracy_diagnostic(
            "insufficient-path-samples",
            "Path frame evaluation requires at least two path samples.",
            parameter=t,
        )
        return PathFrameSampleRecord(
            parameter=t,
            point=np.zeros(3, dtype=float),
            u_axis=np.array([1.0, 0.0, 0.0], dtype=float),
            v_axis=np.array([0.0, 1.0, 0.0], dtype=float),
            w_axis=np.array([0.0, 0.0, 1.0], dtype=float),
            twist_degrees=0.0,
            scale=1.0,
            diagnostics=(diagnostic,),
        )
    point = _polyline_point_at(path_points, t)
    previous_point = _polyline_point_at(path_points, max(0.0, t - 1e-5))
    next_point = _polyline_point_at(path_points, min(1.0, t + 1e-5))
    tangent = next_point - previous_point
    if float(np.linalg.norm(tangent)) == 0.0:
        tangent = path_points[-1] - path_points[0]
    if float(np.linalg.norm(tangent)) == 0.0:
        diagnostic = build_path_frame_degeneracy_diagnostic(
            "degenerate-tangent",
            "Path frame evaluation could not derive a non-zero tangent.",
            parameter=t,
        )
        return PathFrameSampleRecord(
            parameter=t,
            point=point,
            u_axis=np.array([1.0, 0.0, 0.0], dtype=float),
            v_axis=np.array([0.0, 1.0, 0.0], dtype=float),
            w_axis=np.array([0.0, 0.0, 1.0], dtype=float),
            twist_degrees=0.0,
            scale=1.0,
            diagnostics=(diagnostic,),
        )
    u_axis, v_axis, w_axis = _frame_for_tangent(tangent)
    twist, scale = interpolate_path_twist_scale(policy_record, t)
    if twist != 0.0:
        u_axis = _rotate_vector_about_axis(u_axis, w_axis, twist)
        v_axis = _rotate_vector_about_axis(v_axis, w_axis, twist)
    return PathFrameSampleRecord(
        parameter=t,
        point=point,
        u_axis=u_axis,
        v_axis=v_axis,
        w_axis=w_axis,
        twist_degrees=twist,
        scale=scale,
    )


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
class HeightmapSurfacePatch(SurfacePatch):
    """Sampled heightfield patch with bilinear evaluation."""

    height_samples: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=float))
    alpha_mask: np.ndarray = field(default_factory=lambda: np.ones((2, 2), dtype=bool))
    alpha_mode: Literal["mask", "ignore"] = "mask"
    xy_scale: tuple[float, float] = (1.0, 1.0)
    center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    height_scale: float = 1.0

    def __post_init__(self) -> None:
        samples = np.asarray(self.height_samples, dtype=float)
        if samples.ndim != 2:
            raise ValueError("HeightmapSurfacePatch.height_samples must be a 2D array.")
        if samples.shape[0] < 2 or samples.shape[1] < 2:
            raise ValueError("HeightmapSurfacePatch.height_samples must be at least 2x2.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("HeightmapSurfacePatch.height_samples must be finite.")
        alpha_mask = np.asarray(self.alpha_mask, dtype=bool)
        if alpha_mask.shape != samples.shape:
            raise ValueError("HeightmapSurfacePatch.alpha_mask must match height_samples shape.")
        alpha_mode = str(self.alpha_mode).lower()
        if alpha_mode not in {"mask", "ignore"}:
            raise ValueError("HeightmapSurfacePatch.alpha_mode must be 'mask' or 'ignore'.")
        xy_scale = tuple(float(value) for value in self.xy_scale)
        if len(xy_scale) != 2 or xy_scale[0] <= 0.0 or xy_scale[1] <= 0.0:
            raise ValueError("HeightmapSurfacePatch.xy_scale must contain two positive values.")
        center = _as_vec3(self.center, name="center")
        height_scale = float(self.height_scale)
        if not np.isfinite(height_scale):
            raise ValueError("HeightmapSurfacePatch.height_scale must be finite.")
        domain = ParameterDomain((0.0, float(samples.shape[1] - 1)), (0.0, float(samples.shape[0] - 1)))
        object.__setattr__(self, "height_samples", samples)
        object.__setattr__(self, "alpha_mask", alpha_mask)
        object.__setattr__(self, "alpha_mode", alpha_mode)
        object.__setattr__(self, "xy_scale", xy_scale)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "height_scale", height_scale)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "family", "heightmap")
        super().__post_init__()

    def _height_at(self, u: float, v: float) -> float:
        rows, cols = self.height_samples.shape
        u_clamped = float(np.clip(u, 0.0, cols - 1))
        v_clamped = float(np.clip(v, 0.0, rows - 1))
        c0 = int(np.floor(u_clamped))
        r0 = int(np.floor(v_clamped))
        c1 = min(c0 + 1, cols - 1)
        r1 = min(r0 + 1, rows - 1)
        du = u_clamped - c0
        dv = v_clamped - r0
        h00 = self.height_samples[r0, c0]
        h10 = self.height_samples[r0, c1]
        h01 = self.height_samples[r1, c0]
        h11 = self.height_samples[r1, c1]
        h0 = h00 * (1.0 - du) + h10 * du
        h1 = h01 * (1.0 - du) + h11 * du
        return float((h0 * (1.0 - dv) + h1 * dv) * self.height_scale)

    def geometry_payload(self) -> dict[str, object]:
        return {
            "height_samples": self.height_samples,
            "alpha_mask": self.alpha_mask,
            "alpha_mode": self.alpha_mode,
            "xy_scale": self.xy_scale,
            "center": self.center,
            "height_scale": self.height_scale,
        }

    def point_at(self, u: float, v: float) -> np.ndarray:
        self.validate_parameters(u, v)
        rows, cols = self.height_samples.shape
        sx, sy = self.xy_scale
        x = (float(u) - (cols - 1) / 2.0) * sx + self.center[0]
        y = ((rows - 1 - float(v)) - (rows - 1) / 2.0) * sy + self.center[1]
        z = self.center[2] + self._height_at(float(u), float(v))
        return _transform_point(self.transform_matrix, np.array([x, y, z], dtype=float))

    def derivatives_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        self.validate_parameters(u, v)
        epsilon = 1e-4
        u0 = max(self.domain.u_range[0], float(u) - epsilon)
        u1 = min(self.domain.u_range[1], float(u) + epsilon)
        v0 = max(self.domain.v_range[0], float(v) - epsilon)
        v1 = min(self.domain.v_range[1], float(v) + epsilon)
        du = (self.point_at(u1, v) - self.point_at(u0, v)) / max(u1 - u0, 1e-9)
        dv = (self.point_at(u, v1) - self.point_at(u, v0)) / max(v1 - v0, 1e-9)
        return du, dv


@dataclass(frozen=True)
class DisplacementSurfacePatch(SurfacePatch):
    """Surface patch displaced by a sampled heightfield payload."""

    source_patch: SurfacePatch | None = None
    displacement_samples: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=float))
    alpha_mask: np.ndarray = field(default_factory=lambda: np.ones((2, 2), dtype=bool))
    alpha_mode: Literal["mask", "ignore"] = "ignore"
    height_scale: float = 1.0
    direction: str | tuple[float, float, float] = "normal"
    projection: str = "planar"
    plane: str = "xy"
    projection_bounds: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.source_patch is None:
            raise ValueError("DisplacementSurfacePatch.source_patch is required.")
        if not isinstance(self.source_patch, SurfacePatch):
            raise TypeError("DisplacementSurfacePatch.source_patch must be a SurfacePatch.")
        samples = np.asarray(self.displacement_samples, dtype=float)
        if samples.ndim != 2:
            raise ValueError("DisplacementSurfacePatch.displacement_samples must be a 2D array.")
        if samples.shape[0] < 2 or samples.shape[1] < 2:
            raise ValueError("DisplacementSurfacePatch.displacement_samples must be at least 2x2.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("DisplacementSurfacePatch.displacement_samples must be finite.")
        alpha_mask = np.asarray(self.alpha_mask, dtype=bool)
        if alpha_mask.shape != samples.shape:
            raise ValueError("DisplacementSurfacePatch.alpha_mask must match displacement_samples shape.")
        alpha_mode = str(self.alpha_mode).lower()
        if alpha_mode not in {"mask", "ignore"}:
            raise ValueError("DisplacementSurfacePatch.alpha_mode must be 'mask' or 'ignore'.")
        projection = str(self.projection).lower()
        if projection != "planar":
            raise ValueError("Only planar displacement projection is supported in this build.")
        plane = str(self.plane).lower()
        if plane not in {"xy", "xz", "yz"}:
            raise ValueError("DisplacementSurfacePatch.plane must be 'xy', 'xz', or 'yz'.")
        if self.projection_bounds is None:
            raise ValueError("DisplacementSurfacePatch.projection_bounds is required.")
        projection_bounds = tuple(float(value) for value in np.asarray(self.projection_bounds, dtype=float).reshape(4))
        if not np.all(np.isfinite(np.asarray(projection_bounds, dtype=float))):
            raise ValueError("DisplacementSurfacePatch.projection_bounds must be finite.")
        umin, umax, vmin, vmax = projection_bounds
        if np.isclose(umax, umin) or np.isclose(vmax, vmin):
            raise ValueError("DisplacementSurfacePatch.projection_bounds are degenerate.")
        height_scale = float(self.height_scale)
        if not np.isfinite(height_scale):
            raise ValueError("DisplacementSurfacePatch.height_scale must be finite.")
        direction = self.direction
        if not isinstance(direction, str):
            vector = _as_vec3(direction, name="direction")
            if float(np.linalg.norm(vector)) == 0.0:
                raise ValueError("DisplacementSurfacePatch direction vector must be non-zero.")
            direction = tuple(float(value) for value in vector / float(np.linalg.norm(vector)))
        else:
            direction = direction.lower()
            if direction not in {"normal", "x", "y", "z"}:
                raise ValueError("DisplacementSurfacePatch.direction must be normal, x, y, z, or a vector.")
        object.__setattr__(self, "family", "displacement")
        object.__setattr__(self, "domain", self.source_patch.domain)
        object.__setattr__(self, "displacement_samples", samples)
        object.__setattr__(self, "alpha_mask", alpha_mask)
        object.__setattr__(self, "alpha_mode", alpha_mode)
        object.__setattr__(self, "height_scale", height_scale)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "projection", projection)
        object.__setattr__(self, "plane", plane)
        object.__setattr__(self, "projection_bounds", projection_bounds)
        super().__post_init__()

    def _sample_height(self, u: float, v: float) -> tuple[float, bool]:
        from .heightmap import HeightmapProjectionBoundsPolicy, heightmap_sample_coordinate_record

        rows, cols = self.displacement_samples.shape
        base = self.source_patch.point_at(u, v)
        coord_record = heightmap_sample_coordinate_record(
            np.asarray([base], dtype=float),
            HeightmapProjectionBoundsPolicy(
                projection="planar",
                plane=self.plane,
                bounds=self.projection_bounds,
                source="explicit",
            ),
        )
        u_norm = float(coord_record.u_normalized[0])
        v_norm = float(coord_record.v_normalized[0])
        x = np.clip(u_norm, 0.0, 1.0) * (cols - 1)
        y = (1.0 - np.clip(v_norm, 0.0, 1.0)) * (rows - 1)
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, cols - 1)
        y1 = min(y0 + 1, rows - 1)
        dx = x - x0
        dy = y - y0
        h00 = self.displacement_samples[y0, x0]
        h10 = self.displacement_samples[y0, x1]
        h01 = self.displacement_samples[y1, x0]
        h11 = self.displacement_samples[y1, x1]
        height = (
            (1.0 - dx) * (1.0 - dy) * h00
            + dx * (1.0 - dy) * h10
            + (1.0 - dx) * dy * h01
            + dx * dy * h11
        )
        mx = int(np.clip(round(x), 0, cols - 1))
        my = int(np.clip(round(y), 0, rows - 1))
        masked = not bool(self.alpha_mask[my, mx])
        if self.alpha_mode == "ignore" and masked:
            height = 0.0
        return float(height * self.height_scale), masked

    def _direction_at(self, u: float, v: float) -> np.ndarray:
        if self.direction == "normal":
            return self.source_patch.normal_at(u, v)
        if self.direction == "x":
            return np.array([1.0, 0.0, 0.0], dtype=float)
        if self.direction == "y":
            return np.array([0.0, 1.0, 0.0], dtype=float)
        if self.direction == "z":
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return np.asarray(self.direction, dtype=float)

    def geometry_payload(self) -> dict[str, object]:
        return {
            "source_patch": self.source_patch.canonical_payload(),
            "displacement_samples": self.displacement_samples,
            "alpha_mask": self.alpha_mask,
            "alpha_mode": self.alpha_mode,
            "height_scale": self.height_scale,
            "direction": self.direction,
            "projection": self.projection,
            "plane": self.plane,
            "projection_bounds": self.projection_bounds,
        }

    def source_reference_record(self) -> DisplacementSourcePatchReferenceRecord:
        return DisplacementSourcePatchReferenceRecord(
            source_family=self.source_patch.family,
            source_patch_id=self.source_patch.stable_identity,
            embedded=True,
        )

    def source_identity_diagnostic(self) -> DisplacementIdentityDiagnostic:
        return DisplacementIdentityDiagnostic(
            code="embedded-source-payload",
            message="Displacement source is persisted as an embedded surface patch payload, not derived mesh output.",
            source_patch_id=self.source_patch.stable_identity,
        )

    def domain_mapping_record(self) -> DisplacementDomainMappingRecord:
        return DisplacementDomainMappingRecord(
            source_family=self.source_patch.family,
            source_domain=(self.source_patch.domain.u_range, self.source_patch.domain.v_range),
            displacement_domain=(self.domain.u_range, self.domain.v_range),
            projection_bounds=self.projection_bounds,
        )

    def evaluation_diagnostic(self) -> DisplacementEvaluationDiagnostic:
        return DisplacementEvaluationDiagnostic(
            code="finite-difference-displacement-derivatives",
            message="Displacement derivatives are estimated from displaced source evaluations in the source patch domain.",
            source_patch_id=self.source_patch.stable_identity,
        )

    def point_at(self, u: float, v: float) -> np.ndarray:
        self.validate_parameters(u, v)
        base = self.source_patch.point_at(u, v)
        height, _masked = self._sample_height(u, v)
        local = base + self._direction_at(u, v) * height
        return _transform_point(self.transform_matrix, local)

    def derivatives_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        self.validate_parameters(u, v)
        epsilon = 1e-4
        u0 = max(self.domain.u_range[0], float(u) - epsilon)
        u1 = min(self.domain.u_range[1], float(u) + epsilon)
        v0 = max(self.domain.v_range[0], float(v) - epsilon)
        v1 = min(self.domain.v_range[1], float(v) + epsilon)
        du = (self.point_at(u1, v) - self.point_at(u0, v)) / max(u1 - u0, 1e-9)
        dv = (self.point_at(u, v1) - self.point_at(u, v0)) / max(v1 - v0, 1e-9)
        return du, dv


@dataclass(frozen=True)
class DisplacementAuthoringRequest:
    """Authoring request for sampled displacement over a resolved source patch."""

    source_patch: SurfacePatch | None = None
    source_patch_id: str | None = None
    candidate_patches: tuple[SurfacePatch, ...] = ()
    displacement_samples: np.ndarray | Sequence[Sequence[float]] = field(default_factory=lambda: np.zeros((2, 2), dtype=float))
    alpha_mask: np.ndarray | Sequence[Sequence[bool]] | None = None
    alpha_mode: Literal["mask", "ignore"] = "ignore"
    height_scale: float = 1.0
    direction: str | Sequence[float] = "normal"
    projection: Literal["planar"] = "planar"
    plane: Literal["xy", "xz", "yz"] = "xy"
    projection_bounds: tuple[float, float, float, float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DisplacementPayloadDiagnostic:
    """Structured displacement payload authoring diagnostic."""

    code: str
    message: str
    valid: bool

    def canonical_payload(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "valid": self.valid}


@dataclass(frozen=True)
class DisplacementLossinessMetadataRecord:
    """Inspectable lossiness boundary for sampled displacement payloads."""

    family: str = "displacement"
    representation: str = "sampled-grid"
    lossiness: str = "lossless-source-plus-sampled-offsets"

    def canonical_payload(self) -> dict[str, object]:
        return {"family": self.family, "representation": self.representation, "lossiness": self.lossiness}


def displacement_lossiness_metadata_record() -> DisplacementLossinessMetadataRecord:
    return DisplacementLossinessMetadataRecord()


def build_displacement_payload_diagnostic(request: DisplacementAuthoringRequest) -> DisplacementPayloadDiagnostic:
    """Validate a displacement authoring request without evaluating callable payloads."""

    if callable(request.displacement_samples):
        return DisplacementPayloadDiagnostic(
            code="callable-displacement-refused",
            message="Callable displacement functions are not persisted as surface truth; provide sampled displacement data.",
            valid=False,
        )
    resolution = resolve_displacement_source_identity(
        source_patch=request.source_patch,
        source_patch_id=request.source_patch_id,
        candidate_patches=request.candidate_patches,
    )
    if not resolution.resolved:
        return DisplacementPayloadDiagnostic(code=resolution.diagnostic.code, message=resolution.diagnostic.message, valid=False)
    try:
        samples = np.asarray(request.displacement_samples, dtype=float)
        alpha_mask = np.ones(samples.shape, dtype=bool) if request.alpha_mask is None else request.alpha_mask
        DisplacementSurfacePatch(
            family="displacement",
            source_patch=resolution.source_patch,
            displacement_samples=samples,
            alpha_mask=alpha_mask,
            alpha_mode=request.alpha_mode,
            height_scale=request.height_scale,
            direction=request.direction,
            projection=request.projection,
            plane=request.plane,
            projection_bounds=request.projection_bounds,
            metadata=request.metadata,
        )
    except Exception as exc:
        return DisplacementPayloadDiagnostic(
            code="invalid-displacement-payload",
            message=f"Displacement payload is invalid: {exc}",
            valid=False,
        )
    return DisplacementPayloadDiagnostic(
        code="valid-displacement-payload",
        message="Displacement payload is valid sampled surface truth.",
        valid=True,
    )


def make_displacement_surface(request: DisplacementAuthoringRequest) -> SurfaceBody:
    """Create a surface body from a sampled displacement authoring request."""

    diagnostic = build_displacement_payload_diagnostic(request)
    if not diagnostic.valid:
        raise ValueError(diagnostic.message)
    resolution = resolve_displacement_source_identity(
        source_patch=request.source_patch,
        source_patch_id=request.source_patch_id,
        candidate_patches=request.candidate_patches,
    )
    assert resolution.source_patch is not None
    kernel_metadata = {
        "operation": "displacement-authoring",
        "surface_family": "displacement",
        "source_resolution": resolution.canonical_payload(),
        "lossiness": displacement_lossiness_metadata_record().canonical_payload(),
    }
    if request.metadata:
        kernel_metadata.update(dict(request.metadata.get("kernel", {})))
    samples = np.asarray(request.displacement_samples, dtype=float)
    alpha_mask = np.ones(samples.shape, dtype=bool) if request.alpha_mask is None else request.alpha_mask
    patch = DisplacementSurfacePatch(
        family="displacement",
        source_patch=resolution.source_patch,
        displacement_samples=samples,
        alpha_mask=alpha_mask,
        alpha_mode=request.alpha_mode,
        height_scale=request.height_scale,
        direction=request.direction,
        projection=request.projection,
        plane=request.plane,
        projection_bounds=request.projection_bounds,
        metadata={"kernel": kernel_metadata},
    )
    shell = make_surface_shell((patch,), connected=False, metadata={"kernel": {"surface_family": "displacement"}})
    return make_surface_body((shell,), metadata={"kernel": {"surface_family": "displacement", "authoring_boundary": "surface-native"}})


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
        weights = assert_valid_nurbs_weights(self.weights, control_net_shape=control_net.shape[:2])
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

    def rational_evaluation_metadata(self, u: float, v: float) -> NURBSRationalEvaluationMetadata:
        self.validate_parameters(u, v)
        numerator, denominator = self._rational_components(float(u), float(v))
        return NURBSRationalEvaluationMetadata(
            parameter=(float(u), float(v)),
            numerator=numerator,
            denominator=denominator,
            point=numerator / denominator,
            weight_shape=tuple(int(value) for value in self.weights.shape),
        )

    def point_at(self, u: float, v: float) -> np.ndarray:
        self.validate_parameters(u, v)
        metadata = self.rational_evaluation_metadata(float(u), float(v))
        return _transform_point(self.transform_matrix, metadata.point)

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
        frame = evaluate_path_frame(self.path, u_norm, self.frame_policy)
        if frame.diagnostics:
            raise ValueError(frame.diagnostics[0].message)
        return frame.point, frame.u_axis * frame.scale, frame.v_axis * frame.scale, frame.w_axis

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
class SubdivisionSchemeRecord:
    """Inspectable finite subdivision scheme policy."""

    scheme: SubdivisionScheme
    level: int
    crease_count: int
    approximation: str = "finite_catmull_clark"

    def __post_init__(self) -> None:
        if self.scheme != "catmull_clark":
            raise ValueError("SubdivisionSchemeRecord.scheme must be 'catmull_clark'.")
        level = int(self.level)
        crease_count = int(self.crease_count)
        if level < 0:
            raise ValueError("SubdivisionSchemeRecord.level must be >= 0.")
        if crease_count < 0:
            raise ValueError("SubdivisionSchemeRecord.crease_count must be >= 0.")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "crease_count", crease_count)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "level": self.level,
            "crease_count": self.crease_count,
            "approximation": self.approximation,
        }


@dataclass(frozen=True)
class SubdivisionApproximationDiagnostic:
    """Diagnostic that names finite subdivision approximation boundaries."""

    code: str
    message: str
    scheme: str
    level: int

    def canonical_payload(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "scheme": self.scheme, "level": self.level}


def build_subdivision_approximation_diagnostic(
    patch: "SubdivisionSurfacePatch",
    *,
    code: str = "finite-subdivision-approximation",
) -> SubdivisionApproximationDiagnostic:
    return SubdivisionApproximationDiagnostic(
        code=code,
        message="Subdivision evaluation uses finite Catmull-Clark refinement, not hidden mesh fallback or true limit solving.",
        scheme=patch.scheme,
        level=patch.subdivision_level,
    )
ImplicitFieldNodeKind = Literal[
    "sphere",
    "box",
    "plane",
    "constant",
    "union",
    "intersection",
    "difference",
    "translate",
    "scale",
    "negate",
    "sampled_surface",
]

IMPLICIT_FIELD_NODE_KINDS: frozenset[str] = frozenset(
    {
        "sphere",
        "box",
        "plane",
        "constant",
        "union",
        "intersection",
        "difference",
        "translate",
        "scale",
        "negate",
        "sampled_surface",
    }
)
_IMPLICIT_EXECUTABLE_PARAMETER_NAMES = frozenset(
    {
        "__builtins__",
        "callable",
        "code",
        "eval",
        "exec",
        "function",
        "globals",
        "import",
        "lambda",
        "locals",
        "module",
    }
)


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

    def scheme_record(self) -> SubdivisionSchemeRecord:
        return SubdivisionSchemeRecord(
            scheme=self.scheme,
            level=self.subdivision_level,
            crease_count=len(self.creases),
        )

    def approximation_diagnostic(self) -> SubdivisionApproximationDiagnostic:
        return build_subdivision_approximation_diagnostic(self)

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
class SubdivisionAuthoringRequest:
    """Validated native subdivision cage authoring request."""

    control_points: np.ndarray
    faces: tuple[tuple[int, ...], ...]
    creases: tuple[SubdivisionCrease, ...] = ()
    subdivision_level: int = 1
    scheme: SubdivisionScheme = "catmull_clark"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        patch = SubdivisionSurfacePatch(
            family="subdivision",
            control_points=self.control_points,
            faces=self.faces,
            creases=tuple(
                SubdivisionCrease(**crease) if isinstance(crease, dict) else crease
                for crease in self.creases
            ),
            subdivision_level=self.subdivision_level,
            scheme=self.scheme,
            metadata=self.metadata,
        )
        object.__setattr__(self, "control_points", patch.control_points)
        object.__setattr__(self, "faces", patch.faces)
        object.__setattr__(self, "creases", patch.creases)
        object.__setattr__(self, "subdivision_level", patch.subdivision_level)
        object.__setattr__(self, "scheme", patch.scheme)
        object.__setattr__(self, "metadata", patch.metadata)

    @property
    def cage_vertex_count(self) -> int:
        return int(self.control_points.shape[0])

    @property
    def cage_face_count(self) -> int:
        return len(self.faces)

    def to_patch(self) -> SubdivisionSurfacePatch:
        """Build the native subdivision patch represented by this request."""

        return SubdivisionSurfacePatch(
            family="subdivision",
            control_points=self.control_points,
            faces=self.faces,
            creases=self.creases,
            subdivision_level=self.subdivision_level,
            scheme=self.scheme,
            metadata=self.metadata,
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family": "subdivision",
            "scheme": self.scheme,
            "subdivision_level": self.subdivision_level,
            "control_points": self.control_points,
            "faces": self.faces,
            "creases": tuple(crease.canonical_payload() for crease in self.creases),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SubdivisionCageDiagnostic:
    """Diagnostic for native subdivision cage authoring validation."""

    code: str
    message: str
    valid: bool

    def canonical_payload(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "valid": self.valid}


@dataclass(frozen=True)
class SubdivisionProducerProvenanceRecord:
    """Inspectable provenance for a native subdivision producer path."""

    family: str
    operation: str
    authoring_boundary: str
    patch_id: str
    cage_vertex_count: int
    cage_face_count: int
    crease_count: int

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "operation": self.operation,
            "authoring_boundary": self.authoring_boundary,
            "patch_id": self.patch_id,
            "cage_vertex_count": self.cage_vertex_count,
            "cage_face_count": self.cage_face_count,
            "crease_count": self.crease_count,
        }


@dataclass(frozen=True)
class SubdivisionImportRequest:
    """External cage import request that must already contain native topology."""

    payload: Mapping[str, object]
    source_format: str = "native-cage"
    source_id: str = ""
    max_control_points: int = 10000
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise ValueError("SubdivisionImportRequest.payload must be a mapping.")
        source_format = str(self.source_format).strip()
        if not source_format:
            raise ValueError("SubdivisionImportRequest.source_format must be non-empty.")
        max_control_points = int(self.max_control_points)
        if max_control_points <= 0:
            raise ValueError("SubdivisionImportRequest.max_control_points must be positive.")
        object.__setattr__(self, "source_format", source_format)
        object.__setattr__(self, "source_id", str(self.source_id).strip())
        object.__setattr__(self, "max_control_points", max_control_points)
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))


@dataclass(frozen=True)
class NormalizedSubdivisionCagePayload:
    """Surface-native cage payload normalized from an import request."""

    authoring_request: SubdivisionAuthoringRequest
    source_format: str
    source_id: str

    def canonical_payload(self) -> dict[str, object]:
        payload = self.authoring_request.canonical_payload()
        payload["source_format"] = self.source_format
        payload["source_id"] = self.source_id
        return payload


@dataclass(frozen=True)
class SubdivisionImportDiagnostic:
    """Diagnostic for external subdivision cage import normalization."""

    code: str
    message: str
    supported: bool
    source_format: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "source_format": self.source_format,
            "supported": self.supported,
        }


def normalize_subdivision_import_provenance(request: SubdivisionImportRequest) -> dict[str, object]:
    """Return deterministic provenance metadata for a cage import."""

    return {
        "operation": "subdivision-cage-import",
        "surface_family": "subdivision",
        "authoring_boundary": "surface-native",
        "import_source_format": request.source_format,
        "import_source_id": request.source_id,
    }


def normalize_subdivision_cage_import_payload(
    request: SubdivisionImportRequest | Mapping[str, object],
) -> NormalizedSubdivisionCagePayload:
    """Normalize an explicit subdivision cage payload through native validation."""

    import_request = request if isinstance(request, SubdivisionImportRequest) else SubdivisionImportRequest(payload=request)
    payload = import_request.payload
    if "mesh" in payload or ("vertices" in payload and "control_points" not in payload):
        raise ValueError("Subdivision cage import requires explicit native control_points/faces; no mesh fallback was attempted.")
    if "control_points" not in payload or "faces" not in payload:
        raise ValueError("Subdivision cage import requires control_points and faces.")
    control_points = _as_control_points3(payload["control_points"], name="control_points")  # type: ignore[arg-type]
    if control_points.shape[0] > import_request.max_control_points:
        raise ValueError(
            f"Subdivision cage import has {control_points.shape[0]} control points, "
            f"exceeding max_control_points={import_request.max_control_points}."
        )
    metadata = _normalize_metadata(import_request.metadata)
    kernel = dict(metadata.get("kernel", {})) if isinstance(metadata.get("kernel"), Mapping) else {}
    kernel.update(normalize_subdivision_import_provenance(import_request))
    metadata["kernel"] = kernel
    authoring_request = SubdivisionAuthoringRequest(
        control_points=control_points,
        faces=tuple(tuple(int(index) for index in face) for face in payload["faces"]),  # type: ignore[index]
        creases=tuple(payload.get("creases", ())),  # type: ignore[arg-type]
        subdivision_level=int(payload.get("subdivision_level", 1)),
        scheme=str(payload.get("scheme", "catmull_clark")),
        metadata=metadata,
    )
    return NormalizedSubdivisionCagePayload(
        authoring_request=authoring_request,
        source_format=import_request.source_format,
        source_id=import_request.source_id,
    )


def build_subdivision_import_diagnostic(
    request: SubdivisionImportRequest | Mapping[str, object],
) -> SubdivisionImportDiagnostic:
    """Inspect whether a subdivision cage import request can become surface truth."""

    source_format = request.source_format if isinstance(request, SubdivisionImportRequest) else "native-cage"
    try:
        normalize_subdivision_cage_import_payload(request)
    except Exception as exc:
        return SubdivisionImportDiagnostic(
            code="unsupported-subdivision-cage-import",
            message=f"Subdivision cage import is unsupported: {exc}",
            supported=False,
            source_format=source_format,
        )
    return SubdivisionImportDiagnostic(
        code="supported-subdivision-cage-import",
        message="Subdivision cage import contains explicit native cage topology.",
        supported=True,
        source_format=source_format,
    )


def import_subdivision_cage(
    request: SubdivisionImportRequest | Mapping[str, object],
) -> SurfaceBody:
    """Import an explicit native subdivision cage as a surface body."""

    normalized = normalize_subdivision_cage_import_payload(request)
    return make_subdivision_surface(request=normalized.authoring_request)


def build_subdivision_cage_diagnostic(
    request: SubdivisionAuthoringRequest | Mapping[str, object],
) -> SubdivisionCageDiagnostic:
    """Validate a subdivision authoring request without falling back to mesh repair."""

    try:
        if isinstance(request, SubdivisionAuthoringRequest):
            request.to_patch()
        else:
            SubdivisionAuthoringRequest(
                control_points=request["control_points"],  # type: ignore[index]
                faces=request["faces"],  # type: ignore[index]
                creases=tuple(request.get("creases", ())),
                subdivision_level=int(request.get("subdivision_level", 1)),
                scheme=str(request.get("scheme", "catmull_clark")),
                metadata=dict(request.get("metadata", {})),  # type: ignore[arg-type]
            )
    except Exception as exc:
        return SubdivisionCageDiagnostic(
            code="invalid-subdivision-cage",
            message=f"Native subdivision cage is invalid: {exc}",
            valid=False,
        )
    return SubdivisionCageDiagnostic(
        code="valid-subdivision-cage",
        message="Native subdivision cage is valid for surface-native authoring.",
        valid=True,
    )


def subdivision_producer_provenance_record(
    patch: SubdivisionSurfacePatch,
) -> SubdivisionProducerProvenanceRecord:
    """Return native producer provenance for an authored subdivision patch."""

    kernel = patch.metadata.get("kernel", {})
    operation = str(kernel.get("operation", "subdivision-authoring")) if isinstance(kernel, Mapping) else "subdivision-authoring"
    boundary = str(kernel.get("authoring_boundary", "surface-native")) if isinstance(kernel, Mapping) else "surface-native"
    return SubdivisionProducerProvenanceRecord(
        family=patch.family,
        operation=operation,
        authoring_boundary=boundary,
        patch_id=patch.stable_identity,
        cage_vertex_count=int(patch.control_points.shape[0]),
        cage_face_count=len(patch.faces),
        crease_count=len(patch.creases),
    )


def _normalize_field_parameter(value: object, *, name: str) -> object:
    if isinstance(value, np.ndarray):
        return _normalize_field_parameter(value.tolist(), name=name)
    if isinstance(value, np.generic):
        return _normalize_field_parameter(value.item(), name=name)
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"Implicit field parameter {name!r} must be finite.")
        return scalar
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_field_parameter(item, name=name) for item in value)
    if isinstance(value, dict):
        return {str(key): _normalize_field_parameter(value[key], name=f"{name}.{key}") for key in sorted(value)}
    raise ValueError(f"Implicit field parameter {name!r} has unsupported type {type(value).__name__}.")


def _as_bounds3(value: Sequence[float], *, name: str) -> tuple[float, float, float, float, float, float]:
    bounds = tuple(float(item) for item in value)
    if len(bounds) != 6:
        raise ValueError(f"{name} must contain six values.")
    if not np.all(np.isfinite(bounds)):
        raise ValueError(f"{name} must contain only finite values.")
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    if xmax <= xmin or ymax <= ymin or zmax <= zmin:
        raise ValueError(f"{name} must have positive span on every axis.")
    return bounds


@dataclass(frozen=True)
class ImplicitFieldNode:
    """Declarative allow-listed implicit field node."""

    kind: ImplicitFieldNodeKind
    parameters: dict[str, object] = field(default_factory=dict)
    children: tuple["ImplicitFieldNode", ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        if kind not in IMPLICIT_FIELD_NODE_KINDS:
            raise ValueError(f"Unsupported implicit field node kind {kind!r}.")
        parameters = {
            str(key): _normalize_field_parameter(value, name=str(key))
            for key, value in sorted(dict(self.parameters).items())
        }
        children = tuple(_coerce_implicit_field_node(child) for child in self.children)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "children", children)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "parameters": self.parameters,
            "children": [child.canonical_payload() for child in self.children],
        }


@dataclass(frozen=True)
class ImplicitFieldExpressionProvenanceSeed:
    """Stable provenance seed for an implicit expression graph."""

    operation: str
    source_family: str = "implicit"
    source_ids: tuple[str, ...] = ()
    route_id: str = ""

    def __post_init__(self) -> None:
        operation = str(self.operation).strip()
        source_family = str(self.source_family).strip()
        source_ids = tuple(str(source_id).strip() for source_id in self.source_ids)
        route_id = str(self.route_id).strip()
        if not operation or not source_family:
            raise ValueError("ImplicitFieldExpressionProvenanceSeed operation and source_family must be non-empty.")
        if any(not source_id for source_id in source_ids):
            raise ValueError("ImplicitFieldExpressionProvenanceSeed source_ids must be non-empty when provided.")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "source_family", source_family)
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "route_id", route_id)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "source_family": self.source_family,
            "source_ids": self.source_ids,
            "route_id": self.route_id,
        }


@dataclass(frozen=True)
class ImplicitFieldExpressionGraph:
    """Normalized implicit expression graph with a bounded evaluation domain."""

    root: ImplicitFieldNode | dict[str, object]
    bounds: tuple[float, float, float, float, float, float]
    provenance: ImplicitFieldExpressionProvenanceSeed
    graph_id: str = ""
    node_count: int = field(init=False)
    max_depth: int = field(init=False)

    def __post_init__(self) -> None:
        root = _coerce_implicit_field_node(self.root)
        bounds = _as_bounds3(self.bounds, name="ImplicitFieldExpressionGraph.bounds")
        provenance = (
            self.provenance
            if isinstance(self.provenance, ImplicitFieldExpressionProvenanceSeed)
            else ImplicitFieldExpressionProvenanceSeed(**dict(self.provenance))
        )
        node_count, max_depth = _implicit_field_tree_stats(root)
        graph_id = str(self.graph_id).strip() or _implicit_expression_graph_id(root, bounds, provenance)
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "graph_id", graph_id)
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "max_depth", max_depth)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "graph_id": self.graph_id,
            "root": self.root.canonical_payload(),
            "bounds": self.bounds,
            "provenance": self.provenance.canonical_payload(),
            "node_count": self.node_count,
            "max_depth": self.max_depth,
        }


@dataclass(frozen=True)
class ImplicitFieldExpressionDiagnostic:
    """Diagnostic for implicit expression graph normalization."""

    valid: bool
    code: Literal["valid-expression", "invalid-expression", "invalid-domain"]
    message: str
    locator: str = "implicit.expression"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "code": self.code,
            "message": self.message,
            "locator": self.locator,
        }


ImplicitOperandFieldAdapterKind = Literal["exact-field", "analytic-field", "sampled-evaluator", "refused"]


@dataclass(frozen=True)
class ImplicitOperandFieldAdapterResidualRecord:
    """Residual metadata for a surface operand promoted to an implicit field."""

    family: str
    adapter_kind: ImplicitOperandFieldAdapterKind
    residual_kind: Literal["exact", "finite-domain", "sampled-distance", "unavailable"]
    tolerance: float
    lossiness: Literal["none", "declared-tolerance", "refused"] = "declared-tolerance"
    sample_count: int = 0

    def __post_init__(self) -> None:
        family = str(self.family).strip()
        adapter_kind = str(self.adapter_kind).strip()
        residual_kind = str(self.residual_kind).strip()
        tolerance = float(self.tolerance)
        if not family:
            raise ValueError("ImplicitOperandFieldAdapterResidualRecord.family must be non-empty.")
        if adapter_kind not in {"exact-field", "analytic-field", "sampled-evaluator", "refused"}:
            raise ValueError("Implicit operand field adapter kind is unsupported.")
        if residual_kind not in {"exact", "finite-domain", "sampled-distance", "unavailable"}:
            raise ValueError("Implicit operand field residual kind is unsupported.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Implicit operand field adapter tolerance must be finite and non-negative.")
        lossiness = str(self.lossiness).strip()
        if lossiness not in {"none", "declared-tolerance", "refused"}:
            raise ValueError("Implicit operand field adapter lossiness is unsupported.")
        sample_count = int(self.sample_count)
        if sample_count < 0:
            raise ValueError("Implicit operand field adapter sample_count must be non-negative.")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "adapter_kind", adapter_kind)
        object.__setattr__(self, "residual_kind", residual_kind)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "lossiness", lossiness)
        object.__setattr__(self, "sample_count", sample_count)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "adapter_kind": self.adapter_kind,
            "residual_kind": self.residual_kind,
            "tolerance": self.tolerance,
            "lossiness": self.lossiness,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class ImplicitOperandFieldAdapterRefusalDiagnostic:
    """Diagnostic for operands that cannot expose an implicit field adapter."""

    code: Literal["unsupported-family", "invalid-domain", "unsafe-field", "adapter-failed"]
    message: str
    family: str
    patch_id: str = ""
    no_mesh_fallback: bool = True

    def __post_init__(self) -> None:
        code = str(self.code).strip()
        family = str(self.family).strip()
        if code not in {"unsupported-family", "invalid-domain", "unsafe-field", "adapter-failed"}:
            raise ValueError("Implicit operand field adapter refusal code is unsupported.")
        if not family:
            raise ValueError("Implicit operand field adapter refusal family must be non-empty.")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "patch_id", str(self.patch_id).strip())
        object.__setattr__(self, "no_mesh_fallback", bool(self.no_mesh_fallback))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "family": self.family,
            "patch_id": self.patch_id,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


@dataclass(frozen=True)
class ImplicitOperandFieldAdapterRecord:
    """Surface operand to implicit expression graph adapter result."""

    family: str
    patch_id: str
    adapter_kind: ImplicitOperandFieldAdapterKind
    supported: bool
    graph: ImplicitFieldExpressionGraph | None = None
    residuals: tuple[ImplicitOperandFieldAdapterResidualRecord, ...] = ()
    diagnostics: tuple[ImplicitOperandFieldAdapterRefusalDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        family = str(self.family).strip()
        patch_id = str(self.patch_id).strip()
        adapter_kind = str(self.adapter_kind).strip()
        if not family:
            raise ValueError("ImplicitOperandFieldAdapterRecord.family must be non-empty.")
        if adapter_kind not in {"exact-field", "analytic-field", "sampled-evaluator", "refused"}:
            raise ValueError("Implicit operand field adapter kind is unsupported.")
        graph = self.graph
        residuals = tuple(
            residual
            if isinstance(residual, ImplicitOperandFieldAdapterResidualRecord)
            else ImplicitOperandFieldAdapterResidualRecord(**dict(residual))
            for residual in self.residuals
        )
        diagnostics = tuple(
            diagnostic
            if isinstance(diagnostic, ImplicitOperandFieldAdapterRefusalDiagnostic)
            else ImplicitOperandFieldAdapterRefusalDiagnostic(**dict(diagnostic))
            for diagnostic in self.diagnostics
        )
        supported = bool(self.supported)
        if supported and graph is None:
            raise ValueError("Supported implicit operand field adapters must include a graph.")
        if not supported and not diagnostics:
            raise ValueError("Refused implicit operand field adapters must include diagnostics.")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "patch_id", patch_id)
        object.__setattr__(self, "adapter_kind", adapter_kind)
        object.__setattr__(self, "supported", supported)
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "diagnostics", diagnostics)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "patch_id": self.patch_id,
            "adapter_kind": self.adapter_kind,
            "supported": self.supported,
            "graph": None if self.graph is None else self.graph.canonical_payload(),
            "residuals": [residual.canonical_payload() for residual in self.residuals],
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
        }


def _implicit_field_tree_stats(root: ImplicitFieldNode) -> tuple[int, int]:
    node_count = 0
    max_depth = 0
    stack: list[tuple[ImplicitFieldNode, int]] = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        node_count += 1
        max_depth = max(max_depth, depth)
        stack.extend((child, depth + 1) for child in reversed(node.children))
    return node_count, max_depth


def _implicit_expression_graph_id(
    root: ImplicitFieldNode,
    bounds: tuple[float, float, float, float, float, float],
    provenance: ImplicitFieldExpressionProvenanceSeed,
) -> str:
    payload = {
        "root": root.canonical_payload(),
        "bounds": bounds,
        "provenance": provenance.canonical_payload(),
    }
    encoded = json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_implicit_field_expression_graph(
    root: ImplicitFieldNode | dict[str, object],
    *,
    bounds: Sequence[float],
    provenance: ImplicitFieldExpressionProvenanceSeed | None = None,
) -> ImplicitFieldExpressionGraph:
    """Return a normalized implicit expression graph with stable identity."""

    seed = provenance if provenance is not None else ImplicitFieldExpressionProvenanceSeed(operation="implicit-expression")
    return ImplicitFieldExpressionGraph(root=root, bounds=_as_bounds3(bounds, name="implicit.expression.bounds"), provenance=seed)


def bind_implicit_expression_domain(
    graph_or_root: ImplicitFieldExpressionGraph | ImplicitFieldNode | dict[str, object],
    bounds: Sequence[float],
    *,
    provenance: ImplicitFieldExpressionProvenanceSeed | None = None,
) -> ImplicitFieldExpressionGraph:
    """Bind or rebind an implicit expression graph to a bounded domain."""

    if isinstance(graph_or_root, ImplicitFieldExpressionGraph):
        seed = provenance if provenance is not None else graph_or_root.provenance
        return ImplicitFieldExpressionGraph(root=graph_or_root.root, bounds=_as_bounds3(bounds, name="implicit.expression.bounds"), provenance=seed)
    return normalize_implicit_field_expression_graph(graph_or_root, bounds=bounds, provenance=provenance)


def build_implicit_field_expression_diagnostic(
    root: ImplicitFieldNode | dict[str, object],
    *,
    bounds: Sequence[float],
    provenance: ImplicitFieldExpressionProvenanceSeed | None = None,
) -> ImplicitFieldExpressionDiagnostic:
    """Return a non-throwing diagnostic for implicit expression graph validity."""

    try:
        normalize_implicit_field_expression_graph(root, bounds=bounds, provenance=provenance)
    except ValueError as exc:
        text = str(exc)
        code: Literal["invalid-expression", "invalid-domain"] = (
            "invalid-domain" if "bounds" in text or "span" in text else "invalid-expression"
        )
        return ImplicitFieldExpressionDiagnostic(False, code, text)
    return ImplicitFieldExpressionDiagnostic(True, "valid-expression", "Implicit field expression graph is valid.")


def _inflate_degenerate_bounds(
    bounds: Sequence[float],
    *,
    tolerance: float,
) -> tuple[float, float, float, float, float, float]:
    values = tuple(float(value) for value in bounds)
    if len(values) != 6 or not np.all(np.isfinite(values)):
        raise ValueError("Implicit operand field adapter bounds must contain six finite values.")
    pad = max(float(tolerance), 1e-6)
    xmin, xmax, ymin, ymax, zmin, zmax = values
    if xmax <= xmin:
        center = (xmin + xmax) * 0.5
        xmin, xmax = center - pad, center + pad
    if ymax <= ymin:
        center = (ymin + ymax) * 0.5
        ymin, ymax = center - pad, center + pad
    if zmax <= zmin:
        center = (zmin + zmax) * 0.5
        zmin, zmax = center - pad, center + pad
    return _as_bounds3((xmin, xmax, ymin, ymax, zmin, zmax), name="implicit.operand_adapter.bounds")


def _sampled_surface_field_from_patch(
    patch: "SurfacePatch",
    *,
    sample_grid: tuple[int, int],
    tolerance: float,
) -> tuple[ImplicitFieldNode, ImplicitOperandFieldAdapterResidualRecord]:
    u_count, v_count = (int(sample_grid[0]), int(sample_grid[1]))
    if u_count < 2 or v_count < 2:
        raise ValueError("Implicit operand sampled evaluator requires at least a 2x2 sample grid.")
    samples = patch.sample_grid(u_count, v_count).reshape(-1, 3)
    node = make_implicit_field_node(
        "sampled_surface",
        parameters={
            "points": tuple(tuple(float(coord) for coord in point) for point in samples),
            "offset": float(tolerance),
            "source_family": patch.family,
            "source_patch_id": patch.stable_identity,
        },
    )
    residual = ImplicitOperandFieldAdapterResidualRecord(
        family=patch.family,
        adapter_kind="sampled-evaluator",
        residual_kind="sampled-distance",
        tolerance=tolerance,
        lossiness="declared-tolerance",
        sample_count=int(samples.shape[0]),
    )
    return node, residual


def adapt_surface_patch_to_implicit_field(
    patch: "SurfacePatch",
    *,
    operation: str = "operand-field-adapter",
    bounds: Sequence[float] | None = None,
    tolerance: float = 1e-6,
    sample_grid: tuple[int, int] = (5, 5),
) -> ImplicitOperandFieldAdapterRecord:
    """Adapt a surface patch into a bounded implicit field expression without mesh fallback."""

    if not isinstance(patch, SurfacePatch):
        raise TypeError("Implicit operand field adapters require a SurfacePatch operand.")
    tol = float(tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("Implicit operand field adapter tolerance must be finite and non-negative.")
    try:
        adapter_bounds = _inflate_degenerate_bounds(
            patch.bounds_estimate() if bounds is None else bounds,
            tolerance=max(tol, 1e-6),
        )
        if isinstance(patch, ImplicitSurfacePatch):
            root = patch.field
            adapter_kind: ImplicitOperandFieldAdapterKind = "exact-field"
            residuals = (
                ImplicitOperandFieldAdapterResidualRecord(
                    family=patch.family,
                    adapter_kind=adapter_kind,
                    residual_kind="exact",
                    tolerance=tol,
                    lossiness="none",
                    sample_count=0,
                ),
            )
            adapter_bounds = _inflate_degenerate_bounds(bounds if bounds is not None else patch.bounds, tolerance=max(tol, 1e-6))
        elif isinstance(patch, PlanarSurfacePatch):
            normal = patch.normal_at(patch.domain.u_range[0], patch.domain.v_range[0])
            origin = patch.point_at(patch.domain.u_range[0], patch.domain.v_range[0])
            root = make_implicit_field_node(
                "plane",
                parameters={
                    "normal": tuple(float(value) for value in normal),
                    "offset": float(np.dot(normal, origin)),
                },
            )
            adapter_kind = "analytic-field"
            residuals = (
                ImplicitOperandFieldAdapterResidualRecord(
                    family=patch.family,
                    adapter_kind=adapter_kind,
                    residual_kind="finite-domain",
                    tolerance=tol,
                    lossiness="declared-tolerance",
                    sample_count=0,
                ),
            )
        elif patch.family in {"bspline", "nurbs", "sweep", "subdivision", "heightmap", "displacement", "ruled", "revolution"}:
            root, residual = _sampled_surface_field_from_patch(patch, sample_grid=sample_grid, tolerance=tol)
            adapter_kind = "sampled-evaluator"
            residuals = (residual,)
        else:
            diagnostic = ImplicitOperandFieldAdapterRefusalDiagnostic(
                code="unsupported-family",
                message=(
                    f"Patch family '{patch.family}' has no implicit operand field adapter; "
                    "no mesh fallback was attempted."
                ),
                family=patch.family,
                patch_id=patch.stable_identity,
            )
            residual = ImplicitOperandFieldAdapterResidualRecord(
                family=patch.family,
                adapter_kind="refused",
                residual_kind="unavailable",
                tolerance=tol,
                lossiness="refused",
            )
            return ImplicitOperandFieldAdapterRecord(
                family=patch.family,
                patch_id=patch.stable_identity,
                adapter_kind="refused",
                supported=False,
                residuals=(residual,),
                diagnostics=(diagnostic,),
            )
        provenance = ImplicitFieldExpressionProvenanceSeed(
            operation=operation,
            source_family=patch.family,
            source_ids=(patch.stable_identity,),
            route_id=f"implicit-operand-field-adapter.{adapter_kind}",
        )
        graph = normalize_implicit_field_expression_graph(root, bounds=adapter_bounds, provenance=provenance)
        return ImplicitOperandFieldAdapterRecord(
            family=patch.family,
            patch_id=patch.stable_identity,
            adapter_kind=adapter_kind,
            supported=True,
            graph=graph,
            residuals=residuals,
        )
    except Exception as exc:
        diagnostic = ImplicitOperandFieldAdapterRefusalDiagnostic(
            code="adapter-failed",
            message=f"Implicit operand field adapter failed for family '{patch.family}': {exc}; no mesh fallback was attempted.",
            family=patch.family,
            patch_id=patch.stable_identity,
        )
        residual = ImplicitOperandFieldAdapterResidualRecord(
            family=patch.family,
            adapter_kind="refused",
            residual_kind="unavailable",
            tolerance=tol,
            lossiness="refused",
        )
        return ImplicitOperandFieldAdapterRecord(
            family=patch.family,
            patch_id=patch.stable_identity,
            adapter_kind="refused",
            supported=False,
            residuals=(residual,),
            diagnostics=(diagnostic,),
        )


@dataclass(frozen=True)
class ImplicitFieldSafetyPolicy:
    """Bounds and refusal rules for declarative implicit field payloads."""

    max_nodes: int = 64
    max_depth: int = 16
    max_children_per_node: int = 8
    max_parameters_per_node: int = 16
    max_string_length: int = 128

    def __post_init__(self) -> None:
        for name in ("max_nodes", "max_depth", "max_children_per_node", "max_parameters_per_node", "max_string_length"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"ImplicitFieldSafetyPolicy.{name} must be > 0.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ImplicitFieldValidationDiagnostic:
    """Diagnostic returned after an implicit field safety pass."""

    safe: bool
    node_count: int
    max_depth: int
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "safe", bool(self.safe))
        object.__setattr__(self, "node_count", int(self.node_count))
        object.__setattr__(self, "max_depth", int(self.max_depth))
        object.__setattr__(self, "reason", str(self.reason))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "safe": self.safe,
            "node_count": self.node_count,
            "max_depth": self.max_depth,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ImplicitRejectedNodeLocator:
    """Path to the implicit node or parameter that made authoring unsafe."""

    path: str
    node_kind: str
    reason: str

    def canonical_payload(self) -> dict[str, object]:
        return {"path": self.path, "node_kind": self.node_kind, "reason": self.reason}


@dataclass(frozen=True)
class ImplicitUnsafeAuthoringDiagnostic:
    """Structured authoring-time implicit safety result with a rejected path."""

    safe: bool
    reason: str
    node_count: int
    max_depth: int
    locator: ImplicitRejectedNodeLocator | None = None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "safe": self.safe,
            "reason": self.reason,
            "node_count": self.node_count,
            "max_depth": self.max_depth,
            "locator": None if self.locator is None else self.locator.canonical_payload(),
        }


@dataclass(frozen=True)
class ImplicitFieldEvaluationResult:
    """Single implicit field evaluation sample."""

    point: np.ndarray
    value: float
    diagnostic: str = ""

    def __post_init__(self) -> None:
        point = _as_vec3(self.point, name="point")
        value = float(self.value)
        if not np.isfinite(value):
            raise ValueError("ImplicitFieldEvaluationResult.value must be finite.")
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "diagnostic", str(self.diagnostic))

    @property
    def inside(self) -> bool:
        return self.value <= 0.0

    def canonical_payload(self) -> dict[str, object]:
        return {
            "point": self.point,
            "value": self.value,
            "inside": self.inside,
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True)
class ImplicitFieldEvaluationDomain:
    """Bounded sampling domain for implicit field evaluation."""

    bounds: tuple[float, float, float, float, float, float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    samples: tuple[int, int, int] = (8, 8, 8)

    def __post_init__(self) -> None:
        bounds = _as_bounds3(self.bounds, name="ImplicitFieldEvaluationDomain.bounds")
        samples = tuple(int(value) for value in self.samples)
        if len(samples) != 3 or any(value < 2 for value in samples):
            raise ValueError("ImplicitFieldEvaluationDomain.samples must contain three values >= 2.")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "samples", samples)

    def sample_points(self) -> np.ndarray:
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        xs = np.linspace(xmin, xmax, self.samples[0])
        ys = np.linspace(ymin, ymax, self.samples[1])
        zs = np.linspace(zmin, zmax, self.samples[2])
        return np.asarray([(float(x), float(y), float(z)) for z in zs for y in ys for x in xs], dtype=float)


@dataclass(frozen=True)
class ImplicitExtractionBudgetRecord:
    """Bounded implicit extraction sampling budget."""

    bounds: tuple[float, float, float, float, float, float]
    samples: tuple[int, int, int]
    max_sample_count: int = 262144

    def __post_init__(self) -> None:
        bounds = _as_bounds3(self.bounds, name="ImplicitExtractionBudgetRecord.bounds")
        samples = tuple(int(value) for value in self.samples)
        if len(samples) != 3 or any(value < 2 for value in samples):
            raise ValueError("ImplicitExtractionBudgetRecord.samples must contain three values >= 2.")
        max_sample_count = int(self.max_sample_count)
        if max_sample_count <= 0:
            raise ValueError("ImplicitExtractionBudgetRecord.max_sample_count must be positive.")
        sample_count = int(samples[0] * samples[1] * samples[2])
        if sample_count > max_sample_count:
            raise ValueError("Implicit extraction sample count exceeds max_sample_count.")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "max_sample_count", max_sample_count)

    @property
    def sample_count(self) -> int:
        return int(self.samples[0] * self.samples[1] * self.samples[2])

    def evaluation_domain(self) -> ImplicitFieldEvaluationDomain:
        return ImplicitFieldEvaluationDomain(bounds=self.bounds, samples=self.samples)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "bounds": self.bounds,
            "samples": self.samples,
            "sample_count": self.sample_count,
            "max_sample_count": self.max_sample_count,
        }


@dataclass(frozen=True)
class ImplicitBudgetDiagnostic:
    """Structured refusal or acceptance for an implicit extraction budget."""

    code: str
    message: str
    family: str
    executable: bool
    locator: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "family": self.family,
            "executable": self.executable,
            "locator": self.locator,
        }


@dataclass(frozen=True)
class ImplicitBoundsDiagnostic:
    """Structured refusal or acceptance for implicit bounded-domain authoring."""

    code: str
    message: str
    family: str
    bounded: bool
    locator: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "family": self.family,
            "bounded": self.bounded,
            "locator": self.locator,
        }


@dataclass(frozen=True)
class ImplicitResidualClassificationRecord:
    """Classify an implicit residual against an extraction tolerance."""

    value: float
    tolerance: float
    classification: Literal["inside", "outside", "surface"]

    def __post_init__(self) -> None:
        value = float(self.value)
        tolerance = float(self.tolerance)
        if not np.isfinite(value):
            raise ValueError("ImplicitResidualClassificationRecord.value must be finite.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("ImplicitResidualClassificationRecord.tolerance must be finite and non-negative.")
        if self.classification not in {"inside", "outside", "surface"}:
            raise ValueError("ImplicitResidualClassificationRecord.classification is invalid.")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "tolerance", tolerance)

    def canonical_payload(self) -> dict[str, object]:
        return {"value": self.value, "tolerance": self.tolerance, "classification": self.classification}


@dataclass(frozen=True)
class ImplicitFieldSafetyValidationReport:
    """CSG-facing safety gate for bounded implicit field composition."""

    accepted: bool
    bounds: ImplicitBoundsDiagnostic
    budget: ImplicitBudgetDiagnostic
    unsafe_field: ImplicitUnsafeAuthoringDiagnostic
    graph_id: str = ""
    no_mesh_fallback: bool = True

    def __post_init__(self) -> None:
        bounds = self.bounds if isinstance(self.bounds, ImplicitBoundsDiagnostic) else ImplicitBoundsDiagnostic(**dict(self.bounds))
        budget = self.budget if isinstance(self.budget, ImplicitBudgetDiagnostic) else ImplicitBudgetDiagnostic(**dict(self.budget))
        unsafe_field = (
            self.unsafe_field
            if isinstance(self.unsafe_field, ImplicitUnsafeAuthoringDiagnostic)
            else ImplicitUnsafeAuthoringDiagnostic(**dict(self.unsafe_field))
        )
        accepted = bool(self.accepted)
        if accepted and (not bounds.bounded or not budget.executable or not unsafe_field.safe):
            raise ValueError("Accepted implicit safety reports must have bounded, executable, and safe checks.")
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "unsafe_field", unsafe_field)
        object.__setattr__(self, "graph_id", str(self.graph_id).strip())
        object.__setattr__(self, "no_mesh_fallback", bool(self.no_mesh_fallback))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "bounds": self.bounds.canonical_payload(),
            "budget": self.budget.canonical_payload(),
            "unsafe_field": self.unsafe_field.canonical_payload(),
            "graph_id": self.graph_id,
            "no_mesh_fallback": self.no_mesh_fallback,
        }


def _coerce_implicit_field_node(value: object) -> ImplicitFieldNode:
    if isinstance(value, ImplicitFieldNode):
        return value
    if isinstance(value, dict):
        allowed = {"kind", "parameters", "children"}
        keys = set(value)
        unknown = keys - allowed
        if unknown:
            raise ValueError(f"Unsupported implicit field node fields: {sorted(unknown)}.")
        return ImplicitFieldNode(
            kind=value["kind"],
            parameters=dict(value.get("parameters", {})),
            children=tuple(value.get("children", ())),
        )
    raise ValueError("Implicit field nodes must be ImplicitFieldNode instances or dictionaries.")


def _implicit_parameter_has_executable_shape(name: str, value: object, policy: ImplicitFieldSafetyPolicy) -> bool:
    normalized_name = name.strip().lower()
    if normalized_name in _IMPLICIT_EXECUTABLE_PARAMETER_NAMES or normalized_name.startswith("__"):
        return True
    if isinstance(value, str):
        text = value.strip().lower()
        if len(value) > policy.max_string_length:
            return True
        return any(token in text for token in ("__", "import ", "eval(", "exec(", "lambda "))
    if isinstance(value, tuple):
        return any(_implicit_parameter_has_executable_shape(name, item, policy) for item in value)
    if isinstance(value, dict):
        return any(_implicit_parameter_has_executable_shape(str(key), nested_value, policy) for key, nested_value in value.items())
    return False


def assess_implicit_field_security(
    node: ImplicitFieldNode | dict[str, object],
    *,
    policy: ImplicitFieldSafetyPolicy | None = None,
) -> ImplicitFieldValidationDiagnostic:
    policy = ImplicitFieldSafetyPolicy() if policy is None else policy
    root = _coerce_implicit_field_node(node)
    node_count = 0
    observed_depth = 0
    stack: list[tuple[ImplicitFieldNode, int]] = [(root, 1)]
    while stack:
        current, depth = stack.pop()
        node_count += 1
        observed_depth = max(observed_depth, depth)
        if node_count > policy.max_nodes:
            return ImplicitFieldValidationDiagnostic(False, node_count, observed_depth, "implicit field tree exceeds max_nodes")
        if depth > policy.max_depth:
            return ImplicitFieldValidationDiagnostic(False, node_count, observed_depth, "implicit field tree exceeds max_depth")
        if len(current.children) > policy.max_children_per_node:
            return ImplicitFieldValidationDiagnostic(False, node_count, observed_depth, "implicit field node exceeds max_children_per_node")
        if len(current.parameters) > policy.max_parameters_per_node:
            return ImplicitFieldValidationDiagnostic(False, node_count, observed_depth, "implicit field node exceeds max_parameters_per_node")
        for key, value in current.parameters.items():
            if _implicit_parameter_has_executable_shape(key, value, policy):
                return ImplicitFieldValidationDiagnostic(False, node_count, observed_depth, "implicit field parameter has executable shape")
        stack.extend((child, depth + 1) for child in reversed(current.children))
    return ImplicitFieldValidationDiagnostic(True, node_count, observed_depth)


def validate_implicit_field_security(
    node: ImplicitFieldNode | dict[str, object],
    *,
    policy: ImplicitFieldSafetyPolicy | None = None,
) -> ImplicitFieldValidationDiagnostic:
    diagnostic = assess_implicit_field_security(node, policy=policy)
    if not diagnostic.safe:
        raise ValueError(f"Unsafe implicit field payload: {diagnostic.reason}.")
    return diagnostic


def build_implicit_unsafe_authoring_diagnostic(
    node: ImplicitFieldNode | dict[str, object],
    *,
    policy: ImplicitFieldSafetyPolicy | None = None,
) -> ImplicitUnsafeAuthoringDiagnostic:
    """Return an authoring diagnostic that names the rejected implicit path."""

    safety_policy = ImplicitFieldSafetyPolicy() if policy is None else policy
    try:
        root = _coerce_implicit_field_node(node)
    except Exception as exc:
        locator = ImplicitRejectedNodeLocator(path="field", node_kind="unknown", reason=str(exc))
        return ImplicitUnsafeAuthoringDiagnostic(False, str(exc), 0, 0, locator)

    node_count = 0
    observed_depth = 0
    stack: list[tuple[ImplicitFieldNode, int, str]] = [(root, 1, "field")]
    while stack:
        current, depth, path = stack.pop()
        node_count += 1
        observed_depth = max(observed_depth, depth)
        def reject(reason: str, rejected_path: str = path) -> ImplicitUnsafeAuthoringDiagnostic:
            locator = ImplicitRejectedNodeLocator(path=rejected_path, node_kind=current.kind, reason=reason)
            return ImplicitUnsafeAuthoringDiagnostic(False, reason, node_count, observed_depth, locator)

        if node_count > safety_policy.max_nodes:
            return reject("implicit field tree exceeds max_nodes")
        if depth > safety_policy.max_depth:
            return reject("implicit field tree exceeds max_depth")
        if len(current.children) > safety_policy.max_children_per_node:
            return reject("implicit field node exceeds max_children_per_node")
        if len(current.parameters) > safety_policy.max_parameters_per_node:
            return reject("implicit field node exceeds max_parameters_per_node")
        for key, value in current.parameters.items():
            parameter_path = f"{path}.parameters.{key}"
            if _implicit_parameter_has_executable_shape(key, value, safety_policy):
                return reject("implicit field parameter has executable shape", parameter_path)
        for index, child in reversed(tuple(enumerate(current.children))):
            stack.append((child, depth + 1, f"{path}.children[{index}]"))
    return ImplicitUnsafeAuthoringDiagnostic(True, "", node_count, observed_depth)


def validate_implicit_authoring_safety(
    node: ImplicitFieldNode | dict[str, object],
    *,
    policy: ImplicitFieldSafetyPolicy | None = None,
) -> ImplicitUnsafeAuthoringDiagnostic:
    """Raise with a path-specific diagnostic before implicit authoring proceeds."""

    diagnostic = build_implicit_unsafe_authoring_diagnostic(node, policy=policy)
    if not diagnostic.safe:
        path = "field" if diagnostic.locator is None else diagnostic.locator.path
        raise ValueError(f"Unsafe implicit field payload at {path}: {diagnostic.reason}.")
    return diagnostic


def _field_param_vec3(node: ImplicitFieldNode, name: str, default: Sequence[float]) -> np.ndarray:
    return _as_vec3(node.parameters.get(name, default), name=f"{node.kind}.{name}")


def _field_param_float(node: ImplicitFieldNode, name: str, default: float) -> float:
    value = float(node.parameters.get(name, default))
    if not np.isfinite(value):
        raise ValueError(f"{node.kind}.{name} must be finite.")
    return value


def _require_child_count(node: ImplicitFieldNode, *, minimum: int, maximum: int | None = None) -> None:
    count = len(node.children)
    if count < minimum:
        raise ValueError(f"Implicit field node {node.kind!r} requires at least {minimum} children.")
    if maximum is not None and count > maximum:
        raise ValueError(f"Implicit field node {node.kind!r} allows at most {maximum} children.")


def evaluate_implicit_field(
    node: ImplicitFieldNode | dict[str, object],
    point: Sequence[float] | np.ndarray,
) -> ImplicitFieldEvaluationResult:
    field_node = _coerce_implicit_field_node(node)
    validate_implicit_field_security(field_node)
    sample_point = _as_vec3(point, name="point")
    value = _evaluate_implicit_field_value(field_node, sample_point)
    return ImplicitFieldEvaluationResult(point=sample_point, value=value)


def evaluate_implicit_field_gradient(
    node: ImplicitFieldNode | dict[str, object],
    point: Sequence[float] | np.ndarray,
    *,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Estimate a bounded central-difference gradient for a safe implicit field."""

    field_node = _coerce_implicit_field_node(node)
    validate_implicit_field_security(field_node)
    sample_point = _as_vec3(point, name="point")
    step = float(epsilon)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("epsilon must be positive and finite.")
    gradient = np.zeros(3, dtype=float)
    for axis in range(3):
        offset = np.zeros(3, dtype=float)
        offset[axis] = step
        forward = _evaluate_implicit_field_value(field_node, sample_point + offset)
        backward = _evaluate_implicit_field_value(field_node, sample_point - offset)
        gradient[axis] = (forward - backward) / (2.0 * step)
    return gradient


def classify_implicit_residual(value: float, *, tolerance: float = 1e-6) -> ImplicitResidualClassificationRecord:
    residual = float(value)
    tol = float(tolerance)
    if abs(residual) <= tol:
        classification: Literal["inside", "outside", "surface"] = "surface"
    elif residual < 0.0:
        classification = "inside"
    else:
        classification = "outside"
    return ImplicitResidualClassificationRecord(value=residual, tolerance=tol, classification=classification)


def make_implicit_extraction_budget(
    *,
    bounds: tuple[float, float, float, float, float, float],
    samples: tuple[int, int, int],
    max_sample_count: int = 262144,
) -> ImplicitExtractionBudgetRecord:
    return ImplicitExtractionBudgetRecord(bounds=bounds, samples=samples, max_sample_count=max_sample_count)


def build_implicit_bounds_diagnostic(
    bounds: Sequence[float],
    *,
    family: str = "implicit",
) -> ImplicitBoundsDiagnostic:
    """Return a structured bounded-domain diagnostic without evaluating a field."""

    try:
        _as_bounds3(bounds, name="implicit.bounds")
    except Exception as exc:
        return ImplicitBoundsDiagnostic(
            code="invalid-implicit-bounds",
            message=f"Implicit {family} bounds are invalid: {exc}",
            family=family,
            bounded=False,
            locator="bounds",
        )
    return ImplicitBoundsDiagnostic(
        code="valid-implicit-bounds",
        message=f"Implicit {family} bounds are finite with positive span.",
        family=family,
        bounded=True,
        locator="bounds",
    )


def build_implicit_budget_diagnostic(
    *,
    bounds: tuple[float, float, float, float, float, float],
    samples: tuple[int, int, int],
    max_sample_count: int = 262144,
    family: str = "implicit",
) -> ImplicitBudgetDiagnostic:
    """Return a structured extraction budget diagnostic before sampling."""

    try:
        make_implicit_extraction_budget(bounds=bounds, samples=samples, max_sample_count=max_sample_count)
    except Exception as exc:
        text = str(exc)
        locator = "samples" if "samples" in text or "sample count" in text else "bounds"
        if "max_sample_count" in text:
            locator = "max_sample_count"
        return ImplicitBudgetDiagnostic(
            code="invalid-implicit-extraction-budget",
            message=f"Implicit {family} extraction budget is invalid: {text}",
            family=family,
            executable=False,
            locator=locator,
        )
    return ImplicitBudgetDiagnostic(
        code="valid-implicit-extraction-budget",
        message=f"Implicit {family} extraction budget is bounded and executable.",
        family=family,
        executable=True,
        locator="budget",
    )


def validate_implicit_extraction_budget(
    *,
    bounds: tuple[float, float, float, float, float, float],
    samples: tuple[int, int, int],
    max_sample_count: int = 262144,
    family: str = "implicit",
) -> ImplicitExtractionBudgetRecord:
    """Return an extraction budget or raise with the structured diagnostic text."""

    diagnostic = build_implicit_budget_diagnostic(
        bounds=bounds,
        samples=samples,
        max_sample_count=max_sample_count,
        family=family,
    )
    if not diagnostic.executable:
        raise ValueError(diagnostic.message)
    return make_implicit_extraction_budget(bounds=bounds, samples=samples, max_sample_count=max_sample_count)


def build_implicit_field_safety_validation_report(
    graph_or_root: ImplicitFieldExpressionGraph | ImplicitFieldNode | dict[str, object],
    *,
    bounds: Sequence[float] | None = None,
    samples: tuple[int, int, int] = (8, 8, 8),
    max_sample_count: int = 262144,
    policy: ImplicitFieldSafetyPolicy | None = None,
) -> ImplicitFieldSafetyValidationReport:
    """Collect implicit field safety, bounds, and budget checks for CSG composition."""

    graph_id = ""
    if isinstance(graph_or_root, ImplicitFieldExpressionGraph):
        root: ImplicitFieldNode | dict[str, object] = graph_or_root.root
        field_bounds: Sequence[float] | None = graph_or_root.bounds if bounds is None else bounds
        graph_id = graph_or_root.graph_id
    else:
        root = graph_or_root
        field_bounds = bounds

    if field_bounds is None:
        bounds_diagnostic = ImplicitBoundsDiagnostic(
            code="missing-implicit-bounds",
            message="Implicit CSG safety validation requires explicit bounded field domain; no mesh fallback was attempted.",
            family="implicit",
            bounded=False,
            locator="bounds",
        )
        budget_diagnostic = ImplicitBudgetDiagnostic(
            code="invalid-implicit-extraction-budget",
            message="Implicit CSG safety validation cannot allocate a budget without bounds.",
            family="implicit",
            executable=False,
            locator="bounds",
        )
    else:
        bounds_diagnostic = build_implicit_bounds_diagnostic(field_bounds, family="implicit")
        if bounds_diagnostic.bounded:
            budget_diagnostic = build_implicit_budget_diagnostic(
                bounds=_as_bounds3(field_bounds, name="implicit.csg_safety.bounds"),
                samples=samples,
                max_sample_count=max_sample_count,
                family="implicit",
            )
        else:
            budget_diagnostic = ImplicitBudgetDiagnostic(
                code="invalid-implicit-extraction-budget",
                message="Implicit CSG safety validation cannot allocate a budget for invalid bounds.",
                family="implicit",
                executable=False,
                locator="bounds",
            )

    unsafe_diagnostic = build_implicit_unsafe_authoring_diagnostic(root, policy=policy)
    accepted = bounds_diagnostic.bounded and budget_diagnostic.executable and unsafe_diagnostic.safe
    return ImplicitFieldSafetyValidationReport(
        accepted=accepted,
        bounds=bounds_diagnostic,
        budget=budget_diagnostic,
        unsafe_field=unsafe_diagnostic,
        graph_id=graph_id,
        no_mesh_fallback=True,
    )


def validate_implicit_field_safety_for_csg(
    graph_or_root: ImplicitFieldExpressionGraph | ImplicitFieldNode | dict[str, object],
    *,
    bounds: Sequence[float] | None = None,
    samples: tuple[int, int, int] = (8, 8, 8),
    max_sample_count: int = 262144,
    policy: ImplicitFieldSafetyPolicy | None = None,
) -> ImplicitFieldSafetyValidationReport:
    """Return a safety report or raise a deterministic CSG refusal message."""

    report = build_implicit_field_safety_validation_report(
        graph_or_root,
        bounds=bounds,
        samples=samples,
        max_sample_count=max_sample_count,
        policy=policy,
    )
    if not report.accepted:
        reasons = [
            diagnostic.message
            for diagnostic in (report.bounds, report.budget)
            if not getattr(diagnostic, "bounded", getattr(diagnostic, "executable", True))
        ]
        if not report.unsafe_field.safe:
            reasons.append(report.unsafe_field.reason)
        raise ValueError("Implicit CSG field safety validation failed: " + "; ".join(reason for reason in reasons if reason))
    return report


def _evaluate_implicit_field_value(node: ImplicitFieldNode, point: np.ndarray) -> float:
    if node.kind == "sphere":
        center = _field_param_vec3(node, "center", (0.0, 0.0, 0.0))
        radius = _field_param_float(node, "radius", 1.0)
        if radius <= 0.0:
            raise ValueError("sphere.radius must be > 0.")
        return float(np.linalg.norm(point - center) - radius)
    if node.kind == "box":
        center = _field_param_vec3(node, "center", (0.0, 0.0, 0.0))
        half_extents = _field_param_vec3(node, "half_extents", (1.0, 1.0, 1.0))
        if np.any(half_extents <= 0.0):
            raise ValueError("box.half_extents must be > 0 on every axis.")
        q = np.abs(point - center) - half_extents
        outside = np.maximum(q, 0.0)
        inside = min(float(np.max(q)), 0.0)
        return float(np.linalg.norm(outside) + inside)
    if node.kind == "plane":
        normal = _normalize_axis(_field_param_vec3(node, "normal", (0.0, 0.0, 1.0)), name="plane.normal")
        offset = _field_param_float(node, "offset", 0.0)
        return float(np.dot(normal, point) - offset)
    if node.kind == "constant":
        return _field_param_float(node, "value", 0.0)
    if node.kind == "sampled_surface":
        points_value = node.parameters.get("points", ())
        points = np.asarray(points_value, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("sampled_surface.points must contain one or more 3D points.")
        offset = _field_param_float(node, "offset", 0.0)
        if offset < 0.0:
            raise ValueError("sampled_surface.offset must be non-negative.")
        distances = np.linalg.norm(points - point, axis=1)
        return float(np.min(distances) - offset)
    if node.kind == "union":
        _require_child_count(node, minimum=1)
        return float(min(_evaluate_implicit_field_value(child, point) for child in node.children))
    if node.kind == "intersection":
        _require_child_count(node, minimum=1)
        return float(max(_evaluate_implicit_field_value(child, point) for child in node.children))
    if node.kind == "difference":
        _require_child_count(node, minimum=2)
        base = _evaluate_implicit_field_value(node.children[0], point)
        cutters = (-_evaluate_implicit_field_value(child, point) for child in node.children[1:])
        return float(max(base, *cutters))
    if node.kind == "translate":
        _require_child_count(node, minimum=1, maximum=1)
        offset = _field_param_vec3(node, "offset", (0.0, 0.0, 0.0))
        return _evaluate_implicit_field_value(node.children[0], point - offset)
    if node.kind == "scale":
        _require_child_count(node, minimum=1, maximum=1)
        factor = _field_param_float(node, "factor", 1.0)
        if factor <= 0.0:
            raise ValueError("scale.factor must be > 0.")
        return float(_evaluate_implicit_field_value(node.children[0], point / factor) * factor)
    if node.kind == "negate":
        _require_child_count(node, minimum=1, maximum=1)
        return float(-_evaluate_implicit_field_value(node.children[0], point))
    raise ValueError(f"Unsupported implicit field node kind {node.kind!r}.")


def evaluate_implicit_field_domain(
    node: ImplicitFieldNode | dict[str, object],
    domain: ImplicitFieldEvaluationDomain,
) -> np.ndarray:
    field_node = _coerce_implicit_field_node(node)
    validate_implicit_field_security(field_node)
    return np.asarray([_evaluate_implicit_field_value(field_node, point) for point in domain.sample_points()], dtype=float).reshape(
        domain.samples[2],
        domain.samples[1],
        domain.samples[0],
    )


def make_implicit_field_node(
    kind: ImplicitFieldNodeKind,
    *,
    parameters: dict[str, object] | None = None,
    children: Sequence[ImplicitFieldNode | dict[str, object]] = (),
) -> ImplicitFieldNode:
    return ImplicitFieldNode(kind=kind, parameters={} if parameters is None else parameters, children=tuple(children))


def implicit_sphere_field(
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
) -> ImplicitFieldNode:
    """Build an allow-listed sphere signed-distance field node."""

    return make_implicit_field_node("sphere", parameters={"center": tuple(center), "radius": radius})


def implicit_box_field(
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    half_extents: Sequence[float] = (1.0, 1.0, 1.0),
) -> ImplicitFieldNode:
    """Build an allow-listed box signed-distance field node."""

    return make_implicit_field_node("box", parameters={"center": tuple(center), "half_extents": tuple(half_extents)})


def implicit_plane_field(
    *,
    normal: Sequence[float] = (0.0, 0.0, 1.0),
    offset: float = 0.0,
) -> ImplicitFieldNode:
    """Build an allow-listed plane field node."""

    return make_implicit_field_node("plane", parameters={"normal": tuple(normal), "offset": offset})


def implicit_union_field(children: Sequence[ImplicitFieldNode | dict[str, object]]) -> ImplicitFieldNode:
    """Build an allow-listed union composition node."""

    return make_implicit_field_node("union", children=children)


def implicit_intersection_field(children: Sequence[ImplicitFieldNode | dict[str, object]]) -> ImplicitFieldNode:
    """Build an allow-listed intersection composition node."""

    return make_implicit_field_node("intersection", children=children)


def implicit_difference_field(
    base: ImplicitFieldNode | dict[str, object],
    cutters: Sequence[ImplicitFieldNode | dict[str, object]],
) -> ImplicitFieldNode:
    """Build an allow-listed difference composition node."""

    return make_implicit_field_node("difference", children=(base, *tuple(cutters)))


@dataclass(frozen=True)
class ImplicitFieldAuthoringRequest:
    """Validated authored implicit field graph request."""

    field: ImplicitFieldNode | dict[str, object]
    bounds: tuple[float, float, float, float, float, float]
    metadata: dict[str, object] = field(default_factory=dict)
    policy: ImplicitFieldSafetyPolicy = field(default_factory=ImplicitFieldSafetyPolicy)

    def __post_init__(self) -> None:
        field_node = _coerce_implicit_field_node(self.field)
        validate_implicit_authoring_safety(field_node, policy=self.policy)
        object.__setattr__(self, "field", field_node)
        object.__setattr__(self, "bounds", _as_bounds3(self.bounds, name="ImplicitFieldAuthoringRequest.bounds"))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def to_patch(self) -> "ImplicitSurfacePatch":
        return ImplicitSurfacePatch(family="implicit", field=self.field, bounds=self.bounds, metadata=self.metadata)

    def canonical_payload(self) -> dict[str, object]:
        field_node = _coerce_implicit_field_node(self.field)
        return {
            "family": "implicit",
            "field": field_node.canonical_payload(),
            "bounds": self.bounds,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ImplicitFieldProvenanceRecord:
    """Inspectable provenance for an authored implicit field graph."""

    family: str
    operation: str
    authoring_boundary: str
    patch_id: str
    node_count: int
    max_depth: int

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "operation": self.operation,
            "authoring_boundary": self.authoring_boundary,
            "patch_id": self.patch_id,
            "node_count": self.node_count,
            "max_depth": self.max_depth,
        }


def implicit_field_provenance_record(
    patch: "ImplicitSurfacePatch",
    *,
    policy: ImplicitFieldSafetyPolicy | None = None,
) -> ImplicitFieldProvenanceRecord:
    """Return native producer provenance for an authored implicit patch."""

    diagnostic = assess_implicit_field_security(patch.field, policy=policy)
    kernel = patch.metadata.get("kernel", {})
    operation = str(kernel.get("operation", "implicit-authoring")) if isinstance(kernel, Mapping) else "implicit-authoring"
    boundary = str(kernel.get("authoring_boundary", "surface-native")) if isinstance(kernel, Mapping) else "surface-native"
    return ImplicitFieldProvenanceRecord(
        family=patch.family,
        operation=operation,
        authoring_boundary=boundary,
        patch_id=patch.stable_identity,
        node_count=diagnostic.node_count,
        max_depth=diagnostic.max_depth,
    )


@dataclass(frozen=True)
class ImplicitSurfacePatch(SurfacePatch):
    """Implicit surface payload backed by a declarative field tree."""

    field: ImplicitFieldNode = field(
        default_factory=lambda: ImplicitFieldNode(kind="sphere", parameters={"center": (0.0, 0.0, 0.0), "radius": 1.0})
    )
    bounds: tuple[float, float, float, float, float, float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.family != "implicit":
            raise ValueError("ImplicitSurfacePatch.family must be 'implicit'.")
        field_node = _coerce_implicit_field_node(self.field)
        validate_implicit_field_security(field_node)
        bounds = _as_bounds3(self.bounds, name="ImplicitSurfacePatch.bounds")
        object.__setattr__(self, "field", field_node)
        object.__setattr__(self, "bounds", bounds)

    def geometry_payload(self) -> dict[str, object]:
        return {
            "field": self.field.canonical_payload(),
            "bounds": self.bounds,
        }

    def field_value_at(self, point: Sequence[float] | np.ndarray) -> ImplicitFieldEvaluationResult:
        return evaluate_implicit_field(self.field, point)

    def evaluate_domain(self, samples: tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
        return evaluate_implicit_field_domain(self.field, ImplicitFieldEvaluationDomain(bounds=self.bounds, samples=samples))

    def point_at(self, u: float, v: float) -> np.ndarray:
        self.validate_parameters(u, v)
        raise NotImplementedError("ImplicitSurfacePatch has no canonical parametric point_at; use field_value_at or tessellation.")

    def derivatives_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        self.validate_parameters(u, v)
        raise NotImplementedError("ImplicitSurfacePatch has no canonical parametric derivatives; use field_value_at or tessellation.")

    def bounds_estimate(self, *, u_count: int = 3, v_count: int = 3) -> tuple[float, float, float, float, float, float]:
        del u_count, v_count
        return _transform_bounds(self.bounds, self.transform_matrix)


@dataclass(frozen=True)
class SurfaceBoundaryDescriptor:
    """Parametric boundary participation record for seam validation."""

    family: str
    boundary_id: str
    parameter_points: np.ndarray
    comparison_kind: str
    exact: bool = True
    approximation_metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family = str(self.family).strip()
        boundary_id = str(self.boundary_id).strip()
        comparison_kind = str(self.comparison_kind).strip()
        if not family or not boundary_id or not comparison_kind:
            raise ValueError("SurfaceBoundaryDescriptor family, boundary_id, and comparison_kind must be non-empty.")
        parameter_points = _as_points2(self.parameter_points, name="parameter_points")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "boundary_id", boundary_id)
        object.__setattr__(self, "parameter_points", parameter_points)
        object.__setattr__(self, "comparison_kind", comparison_kind)
        object.__setattr__(self, "approximation_metadata", _normalize_metadata(self.approximation_metadata))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "boundary_id": self.boundary_id,
            "parameter_points": self.parameter_points,
            "comparison_kind": self.comparison_kind,
            "exact": self.exact,
            "approximation_metadata": self.approximation_metadata,
        }


@dataclass(frozen=True)
class SurfaceFamilyBoundarySupportRecord:
    """Boundary and seam participation support for one surface family."""

    family: str
    boundary_support: Literal["exact", "approximate", "unsupported"]
    higher_order_residuals: bool
    diagnostic: str = ""
    approximation_method: str = ""

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "boundary_support": self.boundary_support,
            "higher_order_residuals": self.higher_order_residuals,
            "diagnostic": self.diagnostic,
            "approximation_method": self.approximation_method,
        }


@dataclass(frozen=True)
class SurfaceBoundaryDerivativeDiagnostic:
    """Diagnostic emitted while sampling boundary derivatives for continuity checks."""

    code: Literal["unsupported-family", "evaluation-failed"]
    message: str
    family: str
    boundary_id: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary_id": self.boundary_id,
            "code": self.code,
            "family": self.family,
            "message": self.message,
        }


@dataclass(frozen=True)
class SurfaceBoundaryDerivativeSample:
    """One point/derivative/normal sample on a patch boundary."""

    boundary: SurfaceBoundaryRef
    parameter: tuple[float, float]
    point: tuple[float, float, float]
    tangent: tuple[float, float, float]
    du: tuple[float, float, float]
    dv: tuple[float, float, float]
    normal: tuple[float, float, float]
    second_u: tuple[float, float, float] | None = None
    second_v: tuple[float, float, float] | None = None
    exact_first_derivative: bool = True
    residual_metadata: dict[str, object] = field(default_factory=dict)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary": self.boundary.canonical_payload(),
            "du": self.du,
            "dv": self.dv,
            "exact_first_derivative": self.exact_first_derivative,
            "normal": self.normal,
            "parameter": self.parameter,
            "point": self.point,
            "residual_metadata": self.residual_metadata,
            "second_u": self.second_u,
            "second_v": self.second_v,
            "tangent": self.tangent,
        }


@dataclass(frozen=True)
class SurfaceBoundaryDerivativeSummary:
    """Boundary derivative sampling result used by higher-order continuity validation."""

    family: str
    boundary_id: str
    samples: tuple[SurfaceBoundaryDerivativeSample, ...] = ()
    diagnostics: tuple[SurfaceBoundaryDerivativeDiagnostic, ...] = ()
    residual_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary_id": self.boundary_id,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "family": self.family,
            "residual_metadata": self.residual_metadata,
            "samples": [sample.canonical_payload() for sample in self.samples],
            "supported": self.supported,
        }


@dataclass(frozen=True)
class SurfaceContinuityResidualMetrics:
    """Residual metrics computed between two sampled seam boundaries."""

    max_position_delta: float
    max_tangent_delta: float
    max_normal_delta: float
    max_second_derivative_delta: float
    sample_count: int

    def canonical_payload(self) -> dict[str, object]:
        return {
            "max_normal_delta": self.max_normal_delta,
            "max_position_delta": self.max_position_delta,
            "max_second_derivative_delta": self.max_second_derivative_delta,
            "max_tangent_delta": self.max_tangent_delta,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class SurfaceObservedContinuityClassRecord:
    """Observed continuity classes derived from residual metrics."""

    requested: str
    observed_classes: tuple[str, ...]
    passed_requested: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "observed_classes": self.observed_classes,
            "passed_requested": self.passed_requested,
            "requested": self.requested,
        }


@dataclass(frozen=True)
class SurfaceHigherOrderContinuityValidationReport:
    """Pass/fail report for authored higher-order continuity constraints."""

    constraint: SurfaceSeamContinuityConstraint
    residuals: SurfaceContinuityResidualMetrics | None
    observed: SurfaceObservedContinuityClassRecord | None
    diagnostics: tuple[SurfaceContinuityConstraintDiagnostic, ...] = ()

    @property
    def passed(self) -> bool:
        return self.observed is not None and self.observed.passed_requested and not self.diagnostics

    def canonical_payload(self) -> dict[str, object]:
        return {
            "constraint": self.constraint.canonical_payload(),
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "observed": None if self.observed is None else self.observed.canonical_payload(),
            "passed": self.passed,
            "residuals": None if self.residuals is None else self.residuals.canonical_payload(),
        }


@dataclass(frozen=True)
class SurfaceContinuitySeamParameterLocator:
    """Parameter-space location for a higher-order continuity violation."""

    seam_id: str
    first_boundary: SurfaceBoundaryRef
    second_boundary: SurfaceBoundaryRef
    first_parameter: tuple[float, float]
    second_parameter: tuple[float, float]
    sample_index: int

    def canonical_payload(self) -> dict[str, object]:
        return {
            "first_boundary": self.first_boundary.canonical_payload(),
            "first_parameter": self.first_parameter,
            "sample_index": self.sample_index,
            "seam_id": self.seam_id,
            "second_boundary": self.second_boundary.canonical_payload(),
            "second_parameter": self.second_parameter,
        }


@dataclass(frozen=True)
class SurfaceContinuityViolationRecord:
    """Localized failed residual diagnostic for an authored continuity request."""

    seam_id: str
    requested: str
    residual_kind: Literal["position", "tangent", "normal", "curvature", "invalid-report"]
    residual_value: float
    tolerance: float
    locator: SurfaceContinuitySeamParameterLocator | None = None
    message: str = ""
    fix_hint: str = ""

    def canonical_payload(self) -> dict[str, object]:
        return {
            "fix_hint": self.fix_hint,
            "locator": None if self.locator is None else self.locator.canonical_payload(),
            "message": self.message,
            "requested": self.requested,
            "residual_kind": self.residual_kind,
            "residual_value": self.residual_value,
            "seam_id": self.seam_id,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class SurfaceContinuityViolationDiagnostics:
    """Localized diagnostics generated from a failed continuity validation report."""

    seam_id: str
    violations: tuple[SurfaceContinuityViolationRecord, ...] = ()

    @property
    def has_violations(self) -> bool:
        return bool(self.violations)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "has_violations": self.has_violations,
            "seam_id": self.seam_id,
            "violations": [violation.canonical_payload() for violation in self.violations],
        }


@dataclass(frozen=True)
class SurfaceContinuityMetadata:
    """Continuity classification and diagnostics for a seam validation pass."""

    requested: str
    classified: str
    position_tolerance: float
    exact_comparison: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        requested = str(self.requested).strip()
        classified = str(self.classified).strip()
        if not requested or not classified:
            raise ValueError("SurfaceContinuityMetadata requested and classified values must be non-empty.")
        tolerance = float(self.position_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("SurfaceContinuityMetadata.position_tolerance must be finite and positive.")
        object.__setattr__(self, "requested", requested)
        object.__setattr__(self, "classified", classified)
        object.__setattr__(self, "position_tolerance", tolerance)
        object.__setattr__(self, "diagnostics", tuple(str(item) for item in self.diagnostics))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "classified": self.classified,
            "position_tolerance": self.position_tolerance,
            "exact_comparison": self.exact_comparison,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class SurfaceSeamParticipationRecord:
    """One patch boundary's participation in a seam."""

    seam_id: str
    boundary: SurfaceBoundaryRef
    descriptor: SurfaceBoundaryDescriptor
    orientation: Literal["forward", "reversed", "open"] = "forward"

    def __post_init__(self) -> None:
        seam_id = str(self.seam_id).strip()
        orientation = str(self.orientation).strip()
        if not seam_id:
            raise ValueError("SurfaceSeamParticipationRecord.seam_id must be non-empty.")
        if orientation not in {"forward", "reversed", "open"}:
            raise ValueError("SurfaceSeamParticipationRecord.orientation must be forward, reversed, or open.")
        object.__setattr__(self, "seam_id", seam_id)
        object.__setattr__(self, "orientation", orientation)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "seam_id": self.seam_id,
            "boundary": self.boundary.canonical_payload(),
            "descriptor": self.descriptor.canonical_payload(),
            "orientation": self.orientation,
        }


@dataclass(frozen=True)
class SurfaceSeamValidationResult:
    """Result of validating one seam against participating patch boundaries."""

    seam_id: str
    compatible: bool
    continuity: SurfaceContinuityMetadata
    participation: tuple[SurfaceSeamParticipationRecord, ...]
    adjacency_updates: tuple[SurfaceAdjacencyRecord, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        seam_id = str(self.seam_id).strip()
        if not seam_id:
            raise ValueError("SurfaceSeamValidationResult.seam_id must be non-empty.")
        object.__setattr__(self, "seam_id", seam_id)
        object.__setattr__(self, "participation", tuple(self.participation))
        object.__setattr__(self, "adjacency_updates", tuple(self.adjacency_updates))
        object.__setattr__(self, "diagnostics", tuple(str(item) for item in self.diagnostics))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "seam_id": self.seam_id,
            "compatible": self.compatible,
            "continuity": self.continuity.canonical_payload(),
            "participation": [record.canonical_payload() for record in self.participation],
            "adjacency_updates": [record.canonical_payload() for record in self.adjacency_updates],
            "diagnostics": self.diagnostics,
        }


def _surface_boundary_parameter_points(
    patch: SurfacePatch,
    boundary_id: str,
    *,
    sample_count: int,
) -> tuple[np.ndarray, str, bool, dict[str, object]]:
    if isinstance(patch, ImplicitSurfacePatch):
        raise ValueError("ImplicitSurfacePatch boundaries are field-domain bounds and cannot participate in parametric seams.")
    if sample_count < 2:
        raise ValueError("Boundary extraction requires at least two samples.")
    boundary_id = str(boundary_id).strip().lower()
    if boundary_id == "trim:outer":
        outer = patch.outer_trim
        if outer is None:
            raise ValueError("Patch has no outer trim boundary.")
        return outer.normalized().points_uv, "trim-loop", True, {}
    if boundary_id.startswith("trim:inner:"):
        try:
            inner_index = int(boundary_id.split(":", 2)[2])
        except ValueError as exc:
            raise ValueError(f"Invalid trim boundary_id {boundary_id!r}.") from exc
        inner_trims = patch.inner_trims
        if inner_index < 0 or inner_index >= len(inner_trims):
            raise ValueError(f"Trim boundary_id {boundary_id!r} is out of range for the patch.")
        return inner_trims[inner_index].normalized().points_uv, "trim-loop", True, {}
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    if boundary_id == "left":
        points = [(u0, v) for v in np.linspace(v0, v1, sample_count)]
    elif boundary_id == "right":
        points = [(u1, v) for v in np.linspace(v0, v1, sample_count)]
    elif boundary_id == "bottom":
        points = [(u, v0) for u in np.linspace(u0, u1, sample_count)]
    elif boundary_id == "top":
        points = [(u, v1) for u in np.linspace(u0, u1, sample_count)]
    else:
        raise ValueError(f"Unsupported surface boundary_id: {boundary_id!r}")
    exact = not isinstance(patch, (HeightmapSurfacePatch, DisplacementSurfacePatch, SubdivisionSurfacePatch))
    if isinstance(patch, SubdivisionSurfacePatch):
        approximation_metadata = {"method": "finite_subdivision_boundary", "boundary": "parametric-samples"}
    elif isinstance(patch, HeightmapSurfacePatch):
        approximation_metadata = {
            "method": "sampled_heightmap_boundary",
            "boundary": "sampled-grid",
            "heightmap_shape": tuple(int(value) for value in patch.height_samples.shape),
        }
    elif isinstance(patch, DisplacementSurfacePatch):
        approximation_metadata = {
            "method": "sampled_displacement_boundary",
            "boundary": "source-parameter-samples",
            "sample_shape": tuple(int(value) for value in patch.displacement_samples.shape),
        }
    else:
        approximation_metadata = {}
    return np.asarray(points, dtype=float), "parametric-edge", exact, approximation_metadata


def extract_surface_boundary_descriptor(
    patch: SurfacePatch,
    boundary_id: str,
    *,
    sample_count: int = 9,
) -> SurfaceBoundaryDescriptor:
    """Extract a surface-native boundary descriptor without tessellating to a mesh."""

    parameter_points, comparison_kind, exact, approximation_metadata = _surface_boundary_parameter_points(
        patch,
        boundary_id,
        sample_count=sample_count,
    )
    return SurfaceBoundaryDescriptor(
        family=patch.family,
        boundary_id=boundary_id,
        parameter_points=parameter_points,
        comparison_kind=comparison_kind,
        exact=exact,
        approximation_metadata=approximation_metadata,
    )


def surface_family_boundary_support_matrix() -> tuple[SurfaceFamilyBoundarySupportRecord, ...]:
    """Return explicit advanced-family boundary/seam participation support."""

    records: list[SurfaceFamilyBoundarySupportRecord] = []
    for family in ADVANCED_PATCH_FAMILIES:
        if family in {"bspline", "nurbs", "sweep"}:
            records.append(SurfaceFamilyBoundarySupportRecord(family, "exact", True))
        elif family == "subdivision":
            records.append(
                SurfaceFamilyBoundarySupportRecord(
                    family,
                    "approximate",
                    True,
                    approximation_method="finite_subdivision_boundary",
                )
            )
        elif family == "implicit":
            records.append(
                SurfaceFamilyBoundarySupportRecord(
                    family,
                    "unsupported",
                    False,
                    diagnostic="ImplicitSurfacePatch has no canonical parametric seam boundary.",
                )
            )
        elif family == "heightmap":
            records.append(
                SurfaceFamilyBoundarySupportRecord(
                    family,
                    "approximate",
                    True,
                    approximation_method="sampled_heightmap_boundary",
                )
            )
        elif family == "displacement":
            records.append(
                SurfaceFamilyBoundarySupportRecord(
                    family,
                    "approximate",
                    True,
                    approximation_method="sampled_displacement_boundary",
                )
            )
    return tuple(records)


def _boundary_world_samples(patch: SurfacePatch, descriptor: SurfaceBoundaryDescriptor) -> np.ndarray:
    return np.asarray([patch.point_at(float(u), float(v)) for u, v in descriptor.parameter_points], dtype=float)


def _surface_boundary_tangent_from_derivatives(boundary_id: str, du: np.ndarray, dv: np.ndarray) -> np.ndarray:
    boundary = boundary_id.strip().lower()
    if boundary in {"left", "right"}:
        tangent = dv
    elif boundary in {"bottom", "top"}:
        tangent = du
    else:
        tangent = du if np.linalg.norm(du) >= np.linalg.norm(dv) else dv
    return _normalize_axis(tangent, name="boundary_tangent")


def _surface_second_derivative_numeric(
    patch: SurfacePatch,
    u: float,
    v: float,
    *,
    axis: Literal["u", "v"],
    step: float,
) -> np.ndarray:
    u0, u1 = patch.domain.u_range
    v0, v1 = patch.domain.v_range
    if axis == "u":
        low = max(u0, u - step)
        high = min(u1, u + step)
        if high == low:
            return np.zeros(3, dtype=float)
        du_low, _dv_low = patch.derivatives_at(low, v)
        du_high, _dv_high = patch.derivatives_at(high, v)
        return (np.asarray(du_high, dtype=float) - np.asarray(du_low, dtype=float)) / float(high - low)
    low = max(v0, v - step)
    high = min(v1, v + step)
    if high == low:
        return np.zeros(3, dtype=float)
    _du_low, dv_low = patch.derivatives_at(u, low)
    _du_high, dv_high = patch.derivatives_at(u, high)
    return (np.asarray(dv_high, dtype=float) - np.asarray(dv_low, dtype=float)) / float(high - low)


def evaluate_surface_boundary_derivatives(
    patch: SurfacePatch,
    boundary_id: str,
    *,
    patch_index: int = 0,
    sample_count: int = 5,
    second_derivative_step: float = 1e-4,
) -> SurfaceBoundaryDerivativeSummary:
    """Evaluate first derivatives, normals, and numeric second derivatives along a boundary."""

    boundary_id = str(boundary_id).strip()
    diagnostics: list[SurfaceBoundaryDerivativeDiagnostic] = []
    if isinstance(patch, ImplicitSurfacePatch):
        return SurfaceBoundaryDerivativeSummary(
            family=patch.family,
            boundary_id=boundary_id,
            diagnostics=(
                SurfaceBoundaryDerivativeDiagnostic(
                    code="unsupported-family",
                    family=patch.family,
                    boundary_id=boundary_id,
                    message="ImplicitSurfacePatch has no canonical parametric boundary derivatives.",
                ),
            ),
        )
    try:
        descriptor = extract_surface_boundary_descriptor(patch, boundary_id, sample_count=sample_count)
    except ValueError as exc:
        return SurfaceBoundaryDerivativeSummary(
            family=patch.family,
            boundary_id=boundary_id,
            diagnostics=(
                SurfaceBoundaryDerivativeDiagnostic(
                    code="evaluation-failed",
                    family=patch.family,
                    boundary_id=boundary_id,
                    message=str(exc),
                ),
            ),
        )

    samples: list[SurfaceBoundaryDerivativeSample] = []
    residual_metadata = {
        "second_derivative_method": "finite-difference",
        "second_derivative_step": float(second_derivative_step),
        "boundary_descriptor_exact": descriptor.exact,
    }
    boundary_ref = SurfaceBoundaryRef(patch_index, boundary_id)
    for u, v in descriptor.parameter_points:
        try:
            du, dv = patch.derivatives_at(float(u), float(v))
            normal = patch.normal_at(float(u), float(v))
            tangent = _surface_boundary_tangent_from_derivatives(boundary_id, du, dv)
            second_u = _surface_second_derivative_numeric(
                patch,
                float(u),
                float(v),
                axis="u",
                step=second_derivative_step,
            )
            second_v = _surface_second_derivative_numeric(
                patch,
                float(u),
                float(v),
                axis="v",
                step=second_derivative_step,
            )
            point = patch.point_at(float(u), float(v))
        except (NotImplementedError, ValueError) as exc:
            diagnostics.append(
                SurfaceBoundaryDerivativeDiagnostic(
                    code="evaluation-failed",
                    family=patch.family,
                    boundary_id=boundary_id,
                    message=str(exc),
                )
            )
            continue
        samples.append(
            SurfaceBoundaryDerivativeSample(
                boundary=boundary_ref,
                parameter=(float(u), float(v)),
                point=tuple(float(component) for component in point),
                tangent=tuple(float(component) for component in tangent),
                du=tuple(float(component) for component in du),
                dv=tuple(float(component) for component in dv),
                normal=tuple(float(component) for component in normal),
                second_u=tuple(float(component) for component in second_u),
                second_v=tuple(float(component) for component in second_v),
                exact_first_derivative=descriptor.exact,
                residual_metadata=residual_metadata,
            )
        )

    return SurfaceBoundaryDerivativeSummary(
        family=patch.family,
        boundary_id=boundary_id,
        samples=tuple(samples),
        diagnostics=tuple(diagnostics),
        residual_metadata=residual_metadata,
    )


def _unit_vector_delta(first: Sequence[float], second: Sequence[float]) -> float:
    first_array = _normalize_axis(first, name="first")
    second_array = _normalize_axis(second, name="second")
    return float(np.linalg.norm(first_array - second_array))


def compute_surface_continuity_residual_metrics(
    first: SurfaceBoundaryDerivativeSummary,
    second: SurfaceBoundaryDerivativeSummary,
) -> SurfaceContinuityResidualMetrics:
    """Compute max residuals between two boundary derivative summaries."""

    sample_count = min(len(first.samples), len(second.samples))
    if sample_count == 0:
        raise ValueError("Surface continuity residual metrics require at least one paired sample.")
    position_deltas: list[float] = []
    tangent_deltas: list[float] = []
    normal_deltas: list[float] = []
    second_deltas: list[float] = []
    for first_sample, second_sample in zip(first.samples[:sample_count], second.samples[:sample_count], strict=True):
        position_deltas.append(float(np.linalg.norm(np.asarray(first_sample.point) - np.asarray(second_sample.point))))
        tangent_deltas.append(_unit_vector_delta(first_sample.tangent, second_sample.tangent))
        normal_deltas.append(_unit_vector_delta(first_sample.normal, second_sample.normal))
        first_second = np.asarray(first_sample.second_u or (0.0, 0.0, 0.0)) + np.asarray(
            first_sample.second_v or (0.0, 0.0, 0.0)
        )
        second_second = np.asarray(second_sample.second_u or (0.0, 0.0, 0.0)) + np.asarray(
            second_sample.second_v or (0.0, 0.0, 0.0)
        )
        second_deltas.append(float(np.linalg.norm(first_second - second_second)))
    return SurfaceContinuityResidualMetrics(
        max_position_delta=max(position_deltas),
        max_tangent_delta=max(tangent_deltas),
        max_normal_delta=max(normal_deltas),
        max_second_derivative_delta=max(second_deltas),
        sample_count=sample_count,
    )


def classify_surface_continuity_residuals(
    requested: str,
    residuals: SurfaceContinuityResidualMetrics,
    tolerance_policy: SurfaceContinuityTolerancePolicy,
) -> SurfaceObservedContinuityClassRecord:
    """Classify residual metrics without downgrading the requested class."""

    requested = str(requested).strip().upper()
    observed: list[str] = []
    position_ok = residuals.max_position_delta <= tolerance_policy.position_tolerance
    tangent_ok = residuals.max_tangent_delta <= tolerance_policy.tangent_tolerance
    normal_ok = residuals.max_normal_delta <= tolerance_policy.tangent_tolerance
    curvature_ok = residuals.max_second_derivative_delta <= tolerance_policy.curvature_tolerance
    if position_ok:
        observed.extend(("C0", "G0"))
    if position_ok and normal_ok:
        observed.append("G1")
    if position_ok and tangent_ok:
        observed.append("C1")
    if position_ok and normal_ok and curvature_ok:
        observed.append("G2")
    if position_ok and tangent_ok and curvature_ok:
        observed.append("C2")
    return SurfaceObservedContinuityClassRecord(
        requested=requested,
        observed_classes=tuple(observed),
        passed_requested=requested in observed,
    )


def validate_higher_order_surface_continuity(
    constraint: SurfaceSeamContinuityConstraint,
    first: SurfaceBoundaryDerivativeSummary,
    second: SurfaceBoundaryDerivativeSummary,
) -> SurfaceHigherOrderContinuityValidationReport:
    """Validate authored C/G continuity from boundary derivative residuals."""

    diagnostics = list(validate_surface_seam_continuity_constraint(constraint))
    for summary in (first, second):
        diagnostics.extend(
            SurfaceContinuityConstraintDiagnostic(
                code="invalid-continuity",
                seam_id=constraint.seam_id,
                message=diagnostic.message,
            )
            for diagnostic in summary.diagnostics
        )
    if diagnostics:
        return SurfaceHigherOrderContinuityValidationReport(
            constraint=constraint,
            residuals=None,
            observed=None,
            diagnostics=tuple(diagnostics),
        )
    try:
        residuals = compute_surface_continuity_residual_metrics(first, second)
    except ValueError as exc:
        return SurfaceHigherOrderContinuityValidationReport(
            constraint=constraint,
            residuals=None,
            observed=None,
            diagnostics=(
                SurfaceContinuityConstraintDiagnostic(
                    code="invalid-boundary-count",
                    seam_id=constraint.seam_id,
                    message=str(exc),
                ),
            ),
        )
    observed = classify_surface_continuity_residuals(
        constraint.requested,
        residuals,
        constraint.tolerance_policy,
    )
    return SurfaceHigherOrderContinuityValidationReport(
        constraint=constraint,
        residuals=residuals,
        observed=observed,
    )


def _surface_continuity_residual_hotspot(
    first: SurfaceBoundaryDerivativeSummary,
    second: SurfaceBoundaryDerivativeSummary,
    *,
    residual_kind: Literal["position", "tangent", "normal", "curvature"],
) -> tuple[int, float]:
    sample_count = min(len(first.samples), len(second.samples))
    if sample_count == 0:
        return (0, 0.0)
    values: list[tuple[int, float]] = []
    for index, (first_sample, second_sample) in enumerate(
        zip(first.samples[:sample_count], second.samples[:sample_count], strict=True)
    ):
        if residual_kind == "position":
            value = float(np.linalg.norm(np.asarray(first_sample.point) - np.asarray(second_sample.point)))
        elif residual_kind == "tangent":
            value = _unit_vector_delta(first_sample.tangent, second_sample.tangent)
        elif residual_kind == "normal":
            value = _unit_vector_delta(first_sample.normal, second_sample.normal)
        else:
            first_second = np.asarray(first_sample.second_u or (0.0, 0.0, 0.0)) + np.asarray(
                first_sample.second_v or (0.0, 0.0, 0.0)
            )
            second_second = np.asarray(second_sample.second_u or (0.0, 0.0, 0.0)) + np.asarray(
                second_sample.second_v or (0.0, 0.0, 0.0)
            )
            value = float(np.linalg.norm(first_second - second_second))
        values.append((index, value))
    return max(values, key=lambda item: (item[1], -item[0]))


def _surface_continuity_locator(
    constraint: SurfaceSeamContinuityConstraint,
    first: SurfaceBoundaryDerivativeSummary,
    second: SurfaceBoundaryDerivativeSummary,
    sample_index: int,
) -> SurfaceContinuitySeamParameterLocator | None:
    if len(constraint.boundary_uses) != 2 or not first.samples or not second.samples:
        return None
    index = min(sample_index, len(first.samples) - 1, len(second.samples) - 1)
    return SurfaceContinuitySeamParameterLocator(
        seam_id=constraint.seam_id,
        first_boundary=constraint.boundary_uses[0].boundary,
        second_boundary=constraint.boundary_uses[1].boundary,
        first_parameter=first.samples[index].parameter,
        second_parameter=second.samples[index].parameter,
        sample_index=index,
    )


def format_surface_continuity_violation_diagnostic(violation: SurfaceContinuityViolationRecord) -> str:
    """Format a deterministic user-facing continuity violation diagnostic."""

    location = "" if violation.locator is None else f" at sample {violation.locator.sample_index}"
    return (
        f"Seam {violation.seam_id!r} failed requested {violation.requested} {violation.residual_kind} "
        f"residual{location}: {violation.residual_value:.6g} > {violation.tolerance:.6g}. "
        f"{violation.fix_hint}"
    ).strip()


def build_surface_continuity_violation_locators(
    report: SurfaceHigherOrderContinuityValidationReport,
    first: SurfaceBoundaryDerivativeSummary,
    second: SurfaceBoundaryDerivativeSummary,
) -> SurfaceContinuityViolationDiagnostics:
    """Convert a failed higher-order continuity report into localized diagnostics."""

    if report.passed:
        return SurfaceContinuityViolationDiagnostics(seam_id=report.constraint.seam_id)
    violations: list[SurfaceContinuityViolationRecord] = []
    policy = report.constraint.tolerance_policy
    if report.residuals is None:
        violations.append(
            SurfaceContinuityViolationRecord(
                seam_id=report.constraint.seam_id,
                requested=report.constraint.requested,
                residual_kind="invalid-report",
                residual_value=0.0,
                tolerance=0.0,
                message="Continuity validation did not produce residual metrics.",
                fix_hint="Resolve upstream boundary derivative diagnostics before enforcing continuity.",
            )
        )
        return SurfaceContinuityViolationDiagnostics(seam_id=report.constraint.seam_id, violations=tuple(violations))

    residual_checks: tuple[tuple[Literal["position", "tangent", "normal", "curvature"], float, float, str], ...] = (
        (
            "position",
            report.residuals.max_position_delta,
            policy.position_tolerance,
            "Align authored seam boundary positions or relax position tolerance.",
        ),
        (
            "tangent",
            report.residuals.max_tangent_delta,
            policy.tangent_tolerance,
            "Align boundary tangent directions for C-continuity requests.",
        ),
        (
            "normal",
            report.residuals.max_normal_delta,
            policy.tangent_tolerance,
            "Align tangent-plane normals for G-continuity requests.",
        ),
        (
            "curvature",
            report.residuals.max_second_derivative_delta,
            policy.curvature_tolerance,
            "Adjust curvature controls or relax curvature tolerance.",
        ),
    )
    for kind, value, tolerance, hint in residual_checks:
        if value <= tolerance:
            continue
        sample_index, hotspot_value = _surface_continuity_residual_hotspot(first, second, residual_kind=kind)
        locator = _surface_continuity_locator(report.constraint, first, second, sample_index)
        violation = SurfaceContinuityViolationRecord(
            seam_id=report.constraint.seam_id,
            requested=report.constraint.requested,
            residual_kind=kind,
            residual_value=hotspot_value,
            tolerance=tolerance,
            locator=locator,
            fix_hint=hint,
        )
        violations.append(replace(violation, message=format_surface_continuity_violation_diagnostic(violation)))
    return SurfaceContinuityViolationDiagnostics(
        seam_id=report.constraint.seam_id,
        violations=tuple(violations),
    )


def _seam_adjacency_updates(seam: SurfaceSeam) -> tuple[SurfaceAdjacencyRecord, ...]:
    if seam.is_open:
        boundary = seam.boundaries[0]
        return (SurfaceAdjacencyRecord(source=boundary, target=None, seam_id=seam.seam_id, continuity=seam.continuity),)
    first, second = seam.boundaries
    return (
        SurfaceAdjacencyRecord(source=first, target=second, seam_id=seam.seam_id, continuity=seam.continuity),
        SurfaceAdjacencyRecord(source=second, target=first, seam_id=seam.seam_id, continuity=seam.continuity),
    )


def classify_surface_seam_continuity(
    shell: "SurfaceShell",
    seam: SurfaceSeam,
    *,
    tolerance: float = 1e-6,
) -> SurfaceContinuityMetadata:
    """Classify supported seam continuity from patch boundary evaluators."""

    result = validate_surface_seam_participation(shell, seam, tolerance=tolerance)
    return result.continuity


def validate_surface_seam_participation(
    shell: "SurfaceShell",
    seam: SurfaceSeam,
    *,
    tolerance: float = 1e-6,
) -> SurfaceSeamValidationResult:
    """Validate boundary compatibility for one seam without mesh fallback."""

    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Seam validation tolerance must be finite and positive.")
    if seam.is_open:
        boundary = seam.boundaries[0]
        descriptor = extract_surface_boundary_descriptor(shell.patches[boundary.patch_index], boundary.boundary_id)
        continuity = SurfaceContinuityMetadata(seam.continuity, "open", tolerance, descriptor.exact)
        participation = (
            SurfaceSeamParticipationRecord(
                seam_id=seam.seam_id,
                boundary=boundary,
                descriptor=descriptor,
                orientation="open",
            ),
        )
        return SurfaceSeamValidationResult(seam.seam_id, True, continuity, participation, _seam_adjacency_updates(seam))

    first_ref, second_ref = seam.boundaries
    first_patch = shell.patches[first_ref.patch_index]
    second_patch = shell.patches[second_ref.patch_index]
    first_descriptor = extract_surface_boundary_descriptor(first_patch, first_ref.boundary_id)
    second_descriptor = extract_surface_boundary_descriptor(second_patch, second_ref.boundary_id)
    diagnostics: list[str] = []
    if len(first_descriptor.parameter_points) != len(second_descriptor.parameter_points):
        diagnostics.append("boundary sample counts differ")
        compatible = False
        orientation: Literal["forward", "reversed", "open"] = "forward"
    else:
        first_points = _boundary_world_samples(first_patch, first_descriptor)
        second_points = _boundary_world_samples(second_patch, second_descriptor)
        forward_delta = float(np.max(np.linalg.norm(first_points - second_points, axis=1)))
        reversed_delta = float(np.max(np.linalg.norm(first_points - second_points[::-1], axis=1)))
        if reversed_delta < forward_delta:
            orientation = "reversed"
            position_delta = reversed_delta
        else:
            orientation = "forward"
            position_delta = forward_delta
        compatible = position_delta <= tolerance
        if not compatible:
            diagnostics.append(f"boundary positions differ by {position_delta:.6g}, tolerance {tolerance:.6g}")
    exact_comparison = first_descriptor.exact and second_descriptor.exact
    requested = seam.continuity
    support = surface_continuity_support(requested)
    if not support.supported:
        diagnostics.append(support.diagnostic)
        compatible = False
    classified = "C0" if compatible else "incompatible"
    continuity = SurfaceContinuityMetadata(requested, classified, tolerance, exact_comparison, tuple(diagnostics))
    participation = (
        SurfaceSeamParticipationRecord(seam.seam_id, first_ref, first_descriptor, "forward"),
        SurfaceSeamParticipationRecord(seam.seam_id, second_ref, second_descriptor, orientation),
    )
    return SurfaceSeamValidationResult(
        seam_id=seam.seam_id,
        compatible=compatible,
        continuity=continuity,
        participation=participation,
        adjacency_updates=_seam_adjacency_updates(seam),
        diagnostics=tuple(diagnostics),
    )


def surface_adjacency_from_seams(shell: "SurfaceShell") -> tuple[SurfaceAdjacencyRecord, ...]:
    """Create directional adjacency records from the shell's authored seam truth."""

    updates: list[SurfaceAdjacencyRecord] = []
    for seam in shell.seams:
        updates.extend(_seam_adjacency_updates(seam))
    return tuple(updates)


def surface_continuity_support(request: SurfaceContinuityRequest | str) -> SurfaceContinuitySupportRecord:
    """Return the support verdict for an authored continuity request."""

    normalized = request if isinstance(request, SurfaceContinuityRequest) else SurfaceContinuityRequest(str(request))
    if normalized.requested in SUPPORTED_SEAM_CONTINUITY_CLASSES:
        return SurfaceContinuitySupportRecord(
            requested=normalized.requested,
            supported=True,
            support_state="supported",
        )
    state: Literal["unsupported", "not-yet-implemented"] = (
        "not-yet-implemented" if normalized.requested in {"C1", "G1", "C2", "G2"} else "unsupported"
    )
    return SurfaceContinuitySupportRecord(
        requested=normalized.requested,
        supported=False,
        support_state=state,
        diagnostic=(
            f"unsupported continuity request {normalized.requested!r}; "
            f"supported classes are {', '.join(SUPPORTED_SEAM_CONTINUITY_CLASSES)}"
        ),
    )


def normalize_surface_seam_continuity_constraint(
    seam: SurfaceSeam,
    *,
    request: SurfaceContinuityRequest | str | None = None,
    tolerance_policy: SurfaceContinuityTolerancePolicy | None = None,
) -> SurfaceSeamContinuityConstraint:
    """Normalize an authored seam continuity request into a durable constraint record."""

    requested = seam.continuity if request is None else (
        request.requested if isinstance(request, SurfaceContinuityRequest) else str(request)
    )
    boundaries = tuple(seam.boundaries)
    if len(boundaries) == 1:
        roles: tuple[Literal["open"], ...] = ("open",)
    else:
        roles = ("first", "second")  # type: ignore[assignment]
    boundary_uses = tuple(
        SurfaceSeamBoundaryUseRef(seam.seam_id, boundary, role=role)
        for boundary, role in zip(boundaries, roles, strict=True)
    )
    return SurfaceSeamContinuityConstraint(
        seam_id=seam.seam_id,
        requested=requested,
        boundary_uses=boundary_uses,
        tolerance_policy=SurfaceContinuityTolerancePolicy()
        if tolerance_policy is None
        else tolerance_policy,
        source="authored" if request is None or isinstance(request, str) else request.source,
    )


def validate_surface_seam_continuity_constraint(
    constraint: SurfaceSeamContinuityConstraint,
) -> tuple[SurfaceContinuityConstraintDiagnostic, ...]:
    """Validate authored seam continuity intent without enforcing it."""

    diagnostics: list[SurfaceContinuityConstraintDiagnostic] = []
    supported_requests = {"C0", "G0", "C1", "G1", "C2", "G2"}
    if constraint.requested not in supported_requests:
        diagnostics.append(
            SurfaceContinuityConstraintDiagnostic(
                code="invalid-continuity",
                seam_id=constraint.seam_id,
                message=(
                    f"Surface seam continuity constraint {constraint.seam_id!r} requested "
                    f"unsupported continuity {constraint.requested!r}."
                ),
            )
        )
    if len(constraint.boundary_uses) not in {1, 2}:
        diagnostics.append(
            SurfaceContinuityConstraintDiagnostic(
                code="invalid-boundary-count",
                seam_id=constraint.seam_id,
                message="Surface seam continuity constraints require one open boundary use or two shared boundary uses.",
            )
        )
    boundary_keys = tuple(
        (boundary_use.boundary.patch_index, boundary_use.boundary.boundary_id)
        for boundary_use in constraint.boundary_uses
    )
    if len(set(boundary_keys)) != len(boundary_keys):
        diagnostics.append(
            SurfaceContinuityConstraintDiagnostic(
                code="duplicate-boundary",
                seam_id=constraint.seam_id,
                message="Surface seam continuity constraint boundary uses must be unique.",
            )
        )
    expected_roles = {"open"} if len(constraint.boundary_uses) == 1 else {"first", "second"}
    actual_roles = {boundary_use.role for boundary_use in constraint.boundary_uses}
    if len(constraint.boundary_uses) in {1, 2} and actual_roles != expected_roles:
        diagnostics.append(
            SurfaceContinuityConstraintDiagnostic(
                code="invalid-role",
                seam_id=constraint.seam_id,
                message=(
                    "Surface seam continuity constraint roles must be "
                    f"{tuple(sorted(expected_roles))} for this boundary count."
                ),
            )
        )
    if any(boundary_use.seam_id != constraint.seam_id for boundary_use in constraint.boundary_uses):
        diagnostics.append(
            SurfaceContinuityConstraintDiagnostic(
                code="invalid-role",
                seam_id=constraint.seam_id,
                message="Surface seam continuity constraint boundary-use seam IDs must match the constraint seam ID.",
            )
        )
    return tuple(diagnostics)


def check_surface_continuity_enforcement_eligibility(
    request: SurfaceContinuityEnforcementRequest,
) -> tuple[SurfaceContinuityEnforcementRefusalDiagnostic, ...]:
    """Check whether an operation may construct or adjust geometry for continuity."""

    diagnostics: list[SurfaceContinuityEnforcementRefusalDiagnostic] = []
    if request.mutates_source_geometry:
        diagnostics.append(
            SurfaceContinuityEnforcementRefusalDiagnostic(
                code="source-mutation-forbidden",
                operation_id=request.operation_id,
                seam_id=request.constraint.seam_id,
                message=(
                    "Continuity enforcement may not mutate authored source geometry; "
                    "only operation-owned generated output may be adjusted."
                ),
            )
        )
    if not request.owns_generated_geometry:
        diagnostics.append(
            SurfaceContinuityEnforcementRefusalDiagnostic(
                code="validation-only",
                operation_id=request.operation_id,
                seam_id=request.constraint.seam_id,
                message=(
                    "Continuity enforcement is validation-only because the producer did "
                    "not declare ownership of generated geometry."
                ),
            )
        )
    diagnostics.extend(
        SurfaceContinuityEnforcementRefusalDiagnostic(
            code="invalid-constraint",
            operation_id=request.operation_id,
            seam_id=request.constraint.seam_id,
            message=diagnostic.message,
        )
        for diagnostic in validate_surface_seam_continuity_constraint(request.constraint)
    )
    return tuple(diagnostics)


def validate_surface_continuity_enforcement_result(
    request: SurfaceContinuityEnforcementRequest,
    validation_report: "SurfaceHigherOrderContinuityValidationReport",
) -> SurfaceContinuityEnforcementResult:
    """Validate an operation-owned continuity enforcement result."""

    diagnostics = list(check_surface_continuity_enforcement_eligibility(request))
    if not validation_report.passed:
        diagnostics.append(
            SurfaceContinuityEnforcementRefusalDiagnostic(
                code="validation-failed",
                operation_id=request.operation_id,
                seam_id=request.constraint.seam_id,
                message=(
                    f"Continuity enforcement result failed requested "
                    f"{request.constraint.requested} validation."
                ),
            )
        )
    return SurfaceContinuityEnforcementResult(
        request=request,
        accepted=not diagnostics,
        validation_report=validation_report,
        diagnostics=tuple(diagnostics),
    )


def build_surface_unsupported_continuity_diagnostic(
    request: SurfaceContinuityRequest | str,
) -> SurfaceUnsupportedContinuityDiagnostic:
    """Build a structured diagnostic for unsupported seam continuity requests."""

    support = surface_continuity_support(request)
    if support.supported:
        raise ValueError("Supported seam continuity requests do not need unsupported diagnostics.")
    return SurfaceUnsupportedContinuityDiagnostic(
        requested=support.requested,
        supported_classes=SUPPORTED_SEAM_CONTINUITY_CLASSES,
        message=support.diagnostic,
    )


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
                _surface_boundary_parameter_points(patches[boundary.patch_index], boundary.boundary_id, sample_count=2)
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


def make_subdivision_surface(
    *,
    request: SubdivisionAuthoringRequest | None = None,
    control_points: Sequence[Sequence[float]] | np.ndarray | None = None,
    faces: Sequence[Sequence[int]] | None = None,
    creases: Sequence[SubdivisionCrease | dict[str, object]] = (),
    subdivision_level: int = 1,
    scheme: str = "catmull_clark",
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    """Create a surface body from an authored subdivision control cage."""

    if request is None:
        if control_points is None or faces is None:
            raise ValueError("make_subdivision_surface requires control_points and faces or a SubdivisionAuthoringRequest.")
        request = SubdivisionAuthoringRequest(
            control_points=control_points,
            faces=tuple(tuple(int(index) for index in face) for face in faces),
            creases=tuple(SubdivisionCrease(**crease) if isinstance(crease, dict) else crease for crease in creases),
            subdivision_level=subdivision_level,
            scheme=scheme,
            metadata=metadata or {},
        )
    patch_metadata = {
        "kernel": {
            "operation": "subdivision-authoring",
            "surface_family": "subdivision",
            "authoring_boundary": "surface-native",
        }
    }
    if request.metadata:
        patch_metadata["kernel"].update(dict(request.metadata.get("kernel", {})))
    patch = SubdivisionSurfacePatch(
        family="subdivision",
        control_points=request.control_points,
        faces=request.faces,
        creases=request.creases,
        subdivision_level=request.subdivision_level,
        scheme=request.scheme,
        metadata=patch_metadata,
    )
    provenance = subdivision_producer_provenance_record(patch).canonical_payload()
    shell = make_surface_shell(
        (patch,),
        connected=False,
        metadata={"kernel": {"surface_family": "subdivision", "producer_provenance": provenance}},
    )
    return make_surface_body(
        (shell,),
        metadata={
            "kernel": {
                "surface_family": "subdivision",
                "authoring_boundary": "surface-native",
                "producer_provenance": provenance,
            }
        },
    )


def make_implicit_surface(
    *,
    request: ImplicitFieldAuthoringRequest | None = None,
    field: ImplicitFieldNode | dict[str, object] | None = None,
    bounds: Sequence[float] | None = None,
    metadata: dict[str, object] | None = None,
    policy: ImplicitFieldSafetyPolicy | None = None,
) -> SurfaceBody:
    """Create a surface body from a safe declarative implicit field."""

    if request is None:
        if field is None or bounds is None:
            raise ValueError("make_implicit_surface requires field and bounds or an ImplicitFieldAuthoringRequest.")
        request = ImplicitFieldAuthoringRequest(
            field=field,
            bounds=tuple(float(value) for value in bounds),
            metadata=metadata or {},
            policy=ImplicitFieldSafetyPolicy() if policy is None else policy,
        )
    patch_metadata = {
        "kernel": {
            "operation": "implicit-authoring",
            "surface_family": "implicit",
            "authoring_boundary": "surface-native",
        }
    }
    if request.metadata:
        patch_metadata["kernel"].update(dict(request.metadata.get("kernel", {})))
    patch = ImplicitSurfacePatch(
        family="implicit",
        field=request.field,
        bounds=request.bounds,
        metadata=patch_metadata,
    )
    provenance = implicit_field_provenance_record(patch, policy=request.policy).canonical_payload()
    shell = make_surface_shell(
        (patch,),
        connected=False,
        metadata={"kernel": {"surface_family": "implicit", "producer_provenance": provenance}},
    )
    return make_surface_body(
        (shell,),
        metadata={
            "kernel": {
                "surface_family": "implicit",
                "authoring_boundary": "surface-native",
                "producer_provenance": provenance,
            }
        },
    )


__all__ = [
    "ADVANCED_PATCH_FAMILIES",
    "AdvancedPatchFamilyPromotionEvidenceRecord",
    "AdvancedPatchFamilyPromotionReport",
    "AvailableFamilyCompletionReport",
    "AvailableFamilyCompletionSnapshot",
    "AvailableFamilyMissingEvidenceDiagnostic",
    "AvailableFamilyOperationCompletenessReport",
    "AvailableFamilyOperationEvidenceRecord",
    "AvailableFamilyReferenceEvidenceReport",
    "AvailableFamilyReferenceEvidenceSummary",
    "NURBSRationalEvaluationMetadata",
    "NURBSConicConstructionDiagnostic",
    "NURBSConicConstructionRequest",
    "NURBSConicProfilePayload",
    "DisplacementDomainMappingRecord",
    "DisplacementEvaluationDiagnostic",
    "DisplacementAuthoringRequest",
    "DisplacementLossinessMetadataRecord",
    "DisplacementPayloadDiagnostic",
    "DisplacementIdentityDiagnostic",
    "DisplacementSourcePatchReferenceRecord",
    "DisplacementSourceProvenanceRecord",
    "DisplacementSourceResolutionResult",
    "NURBSWeightValidationDiagnostic",
    "FrameTransportPolicyRecord",
    "ImplicitExtractionBudgetRecord",
    "ImplicitBoundsDiagnostic",
    "ImplicitBudgetDiagnostic",
    "ImplicitFieldExpressionDiagnostic",
    "ImplicitFieldExpressionGraph",
    "ImplicitFieldExpressionProvenanceSeed",
    "ImplicitFieldSafetyValidationReport",
    "ImplicitOperandFieldAdapterRecord",
    "ImplicitOperandFieldAdapterRefusalDiagnostic",
    "ImplicitOperandFieldAdapterResidualRecord",
    "ImplicitFieldAuthoringRequest",
    "ImplicitFieldProvenanceRecord",
    "ImplicitRejectedNodeLocator",
    "PathFrameDegeneracyDiagnostic",
    "PathFrameSampleRecord",
    "ImplicitResidualClassificationRecord",
    "ImplicitUnsafeAuthoringDiagnostic",
    "SubdivisionApproximationDiagnostic",
    "SubdivisionAuthoringRequest",
    "SubdivisionCageDiagnostic",
    "SubdivisionImportDiagnostic",
    "SubdivisionImportRequest",
    "SubdivisionProducerProvenanceRecord",
    "NormalizedSubdivisionCagePayload",
    "SubdivisionSchemeRecord",
    "PatchFamilyCapabilityRecord",
    "PatchFamilyAvailabilityDiagnostic",
    "PatchFamilyAvailabilityGateRecord",
    "PatchFamilyOperationSupportRecord",
    "PatchFamilyPromotionEvidenceRecord",
    "PatchFamilyPromotionGapRecord",
    "PatchFamilyPromotionReadinessRecord",
    "SurfaceBodyCompletionDiagnostic",
    "SurfaceBodyCompletionEvidenceRecord",
    "SurfaceBodyCompletionReport",
    "SurfaceFamilyBoundarySupportRecord",
    "SurfaceReferenceArtifactClassRecord",
    "SurfaceReferenceEvidenceMatrixReport",
    "SurfaceReferenceFixtureContractRecord",
    "SurfaceReferenceFixtureRequirementRecord",
    "SurfaceContinuityRequest",
    "SurfaceContinuitySupportRecord",
    "SurfaceUnsupportedContinuityDiagnostic",
    "SurfaceContinuityTolerancePolicy",
    "SurfaceSeamBoundaryUseRef",
    "SurfaceContinuityConstraintDiagnostic",
    "SurfaceSeamContinuityConstraint",
    "SurfaceContinuityEnforcementRequest",
    "SurfaceContinuityEnforcementRefusalDiagnostic",
    "SurfaceContinuityEnforcementResult",
    "SurfaceBoundaryDerivativeDiagnostic",
    "SurfaceBoundaryDerivativeSample",
    "SurfaceBoundaryDerivativeSummary",
    "SurfaceContinuityResidualMetrics",
    "SurfaceObservedContinuityClassRecord",
    "SurfaceHigherOrderContinuityValidationReport",
    "SurfaceContinuitySeamParameterLocator",
    "SurfaceContinuityViolationRecord",
    "SurfaceContinuityViolationDiagnostics",
    "PATCH_FAMILY_AVAILABILITY_REQUIRED_OPERATIONS",
    "PATCH_FAMILY_CAPABILITY_MATRIX",
    "PATCH_FAMILY_FEATURE_COVERAGE",
    "PATCH_FAMILY_PROMOTION_CRITERIA",
    "REQUIRED_V1_PATCH_FAMILIES",
    "SURFACE_BODY_COMPLETION_TRACKS",
    "SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS",
    "SURFACE_REFERENCE_ARTIFACT_CLASSES",
    "SURFACE_REFERENCE_FIXTURE_CONTRACTS",
    "SURFACE_BODY_PROMOTED_EVIDENCE_OWNERS",
    "SUPPORTED_SEAM_CONTINUITY_CLASSES",
    "SUPPORTED_SURFACE_PATCH_FAMILIES",
    "SURFACE_SPEC_66_RETIREMENT_NOTE",
    "assert_patch_family_capability_matrix",
    "assert_advanced_patch_family_promotion_gate",
    "assert_valid_nurbs_weights",
    "assert_patch_family_operation_coverage",
    "audit_all_patch_family_promotion_readiness",
    "audit_patch_family_promotion_readiness",
    "assess_patch_family_availability_promotion",
    "build_nurbs_circular_arc_control_net",
    "build_nurbs_exact_conic_profile_payload",
    "build_path_frame_degeneracy_diagnostic",
    "build_subdivision_approximation_diagnostic",
    "build_subdivision_cage_diagnostic",
    "build_subdivision_import_diagnostic",
    "build_implicit_unsafe_authoring_diagnostic",
    "build_implicit_bounds_diagnostic",
    "build_implicit_budget_diagnostic",
    "build_implicit_field_expression_diagnostic",
    "build_implicit_field_safety_validation_report",
    "adapt_surface_patch_to_implicit_field",
    "build_displacement_payload_diagnostic",
    "build_available_family_missing_evidence_diagnostic",
    "build_available_family_no_hidden_mesh_fallback_diagnostic",
    "build_available_family_completion_report",
    "build_dirty_available_family_reference_diagnostic",
    "import_subdivision_cage",
    "normalize_subdivision_cage_import_payload",
    "normalize_subdivision_import_provenance",
    "resolve_displacement_source_identity",
    "displacement_lossiness_metadata_record",
    "classify_implicit_residual",
    "evaluate_advanced_patch_family_promotion_gate",
    "evaluate_implicit_field_gradient",
    "implicit_box_field",
    "implicit_difference_field",
    "implicit_field_provenance_record",
    "implicit_intersection_field",
    "implicit_plane_field",
    "implicit_sphere_field",
    "implicit_union_field",
    "evaluate_path_frame",
    "bind_implicit_expression_domain",
    "build_surface_unsupported_continuity_diagnostic",
    "normalize_surface_seam_continuity_constraint",
    "check_surface_continuity_enforcement_eligibility",
    "validate_surface_continuity_enforcement_result",
    "validate_nurbs_weights",
    "validate_implicit_authoring_safety",
    "validate_implicit_extraction_budget",
    "validate_implicit_field_safety_for_csg",
    "normalize_implicit_field_expression_graph",
    "interpolate_path_twist_scale",
    "make_implicit_extraction_budget",
    "evaluate_surface_body_completion_gate",
    "evaluate_available_family_reference_evidence_gate",
    "evaluate_surface_body_completion_reference_evidence_matrix",
    "assert_surface_reference_requirement_matrix_covers_capabilities",
    "load_surface_reference_requirement_matrix",
    "make_surface_body_completion_evidence_from_capabilities",
    "make_available_family_promoted_reference_evidence",
    "available_family_producer_path_rows",
    "available_family_storage_tessellation_rows",
    "available_family_seam_loft_rows",
    "collect_available_family_reference_evidence",
    "run_advanced_patch_family_promotion_gate",
    "run_patch_family_availability_promotion_pass",
    "surface_family_boundary_support_matrix",
    "surface_continuity_support",
    "validate_surface_seam_continuity_constraint",
    "surface_body_completion_reference_evidence_matrix",
    "surface_reference_artifact_classes",
    "surface_reference_fixture_contracts",
    "snapshot_available_family_completion_report",
    "summarize_available_family_missing_evidence",
    "summarize_available_family_producer_paths",
    "validate_patch_family_availability_gate",
    "verify_available_family_producer_path_rows",
    "verify_available_family_storage_tessellation_rows",
    "verify_available_family_seam_loft_rows",
    "IMPLICIT_FIELD_NODE_KINDS",
    "ParameterDomain",
    "TrimLoop",
    "SurfaceBoundaryRef",
    "SurfaceAdjacencyRecord",
    "SurfaceSeam",
    "SurfaceBoundaryDescriptor",
    "SurfaceContinuityMetadata",
    "SurfaceSeamParticipationRecord",
    "SurfaceSeamValidationResult",
    "SurfacePatch",
    "PlanarSurfacePatch",
    "HeightmapSurfacePatch",
    "DisplacementSurfacePatch",
    "RuledSurfacePatch",
    "RevolutionSurfacePatch",
    "BSplineSurfacePatch",
    "NURBSSurfacePatch",
    "SweepSurfacePatch",
    "SubdivisionCrease",
    "SubdivisionRefinementResult",
    "SubdivisionSurfacePatch",
    "ImplicitFieldNode",
    "ImplicitFieldExpressionDiagnostic",
    "ImplicitFieldExpressionGraph",
    "ImplicitFieldExpressionProvenanceSeed",
    "ImplicitFieldSafetyValidationReport",
    "ImplicitOperandFieldAdapterRecord",
    "ImplicitOperandFieldAdapterRefusalDiagnostic",
    "ImplicitOperandFieldAdapterResidualRecord",
    "ImplicitFieldEvaluationDomain",
    "ImplicitFieldEvaluationResult",
    "ImplicitFieldSafetyPolicy",
    "ImplicitFieldValidationDiagnostic",
    "ImplicitSurfacePatch",
    "assess_implicit_field_security",
    "evaluate_implicit_field",
    "evaluate_implicit_field_domain",
    "make_implicit_field_node",
    "bind_implicit_expression_domain",
    "build_implicit_field_expression_diagnostic",
    "normalize_implicit_field_expression_graph",
    "adapt_surface_patch_to_implicit_field",
    "build_implicit_field_safety_validation_report",
    "validate_implicit_field_safety_for_csg",
    "validate_implicit_field_security",
    "refine_subdivision_control_cage",
    "classify_surface_seam_continuity",
    "extract_surface_boundary_descriptor",
    "evaluate_surface_boundary_derivatives",
    "compute_surface_continuity_residual_metrics",
    "classify_surface_continuity_residuals",
    "validate_higher_order_surface_continuity",
    "build_surface_continuity_violation_locators",
    "format_surface_continuity_violation_diagnostic",
    "surface_adjacency_from_seams",
    "validate_surface_seam_participation",
    "SurfaceShell",
    "SurfaceBody",
    "make_surface_shell",
    "make_surface_body",
    "make_displacement_surface",
    "make_subdivision_surface",
    "subdivision_producer_provenance_record",
    "make_implicit_surface",
]
