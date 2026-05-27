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
    support_phase: Literal["available", "planned"]
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
class PatchFamilyPromotionEvidenceRecord:
    """Evidence for whether a patch family can be promoted to available."""

    family: str
    current_phase: Literal["available", "planned"]
    promoted_phase: Literal["available", "planned"]
    operation_support: tuple[PatchFamilyOperationSupportRecord, ...]
    diagnostics: tuple[PatchFamilyAvailabilityDiagnostic, ...] = ()

    @property
    def promoted(self) -> bool:
        return self.current_phase != self.promoted_phase and self.promoted_phase == "available"


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
    current_phase: Literal["available", "planned"]
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
    "csg": ("caps", "planar-primitives", "revolved-primitives"),
    "loft": ("loft", "linear-bridge-surfaces", "extrude", "rotate-extrude", "planar-primitives"),
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
    "heightmap": PatchFamilyCapabilityRecord(
        family="heightmap",
        support_phase="planned",
        operations=("sample-grid-payload", "evaluation", "tessellation"),
    ),
    "displacement": PatchFamilyCapabilityRecord(
        family="displacement",
        support_phase="planned",
        operations=("source-surface-reference", "sample-grid-payload", "evaluation", "tessellation"),
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
    if capability.support_phase not in ("available", "planned"):
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
    exact = not isinstance(patch, SubdivisionSurfacePatch)
    approximation_metadata = (
        {"method": "finite_subdivision_boundary", "boundary": "parametric-samples"}
        if isinstance(patch, SubdivisionSurfacePatch)
        else {}
    )
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


def _boundary_world_samples(patch: SurfacePatch, descriptor: SurfaceBoundaryDescriptor) -> np.ndarray:
    return np.asarray([patch.point_at(float(u), float(v)) for u, v in descriptor.parameter_points], dtype=float)


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
    control_points: Sequence[Sequence[float]] | np.ndarray,
    faces: Sequence[Sequence[int]],
    creases: Sequence[SubdivisionCrease | dict[str, object]] = (),
    subdivision_level: int = 1,
    scheme: str = "catmull_clark",
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    """Create a surface body from an authored subdivision control cage."""

    patch_metadata = {
        "kernel": {
            "operation": "subdivision-authoring",
            "surface_family": "subdivision",
            "authoring_boundary": "surface-native",
        }
    }
    if metadata:
        patch_metadata["kernel"].update(dict(metadata.get("kernel", {})))
    patch = SubdivisionSurfacePatch(
        family="subdivision",
        control_points=control_points,
        faces=faces,
        creases=tuple(SubdivisionCrease(**crease) if isinstance(crease, dict) else crease for crease in creases),
        subdivision_level=subdivision_level,
        scheme=scheme,
        metadata=patch_metadata,
    )
    shell = make_surface_shell((patch,), connected=False, metadata={"kernel": {"surface_family": "subdivision"}})
    return make_surface_body((shell,), metadata={"kernel": {"surface_family": "subdivision", "authoring_boundary": "surface-native"}})


def make_implicit_surface(
    *,
    field: ImplicitFieldNode | dict[str, object],
    bounds: Sequence[float],
    metadata: dict[str, object] | None = None,
    policy: ImplicitFieldSafetyPolicy | None = None,
) -> SurfaceBody:
    """Create a surface body from a safe declarative implicit field."""

    field_node = _coerce_implicit_field_node(field)
    validate_implicit_field_security(field_node, policy=policy)
    patch_metadata = {
        "kernel": {
            "operation": "implicit-authoring",
            "surface_family": "implicit",
            "authoring_boundary": "surface-native",
        }
    }
    if metadata:
        patch_metadata["kernel"].update(dict(metadata.get("kernel", {})))
    patch = ImplicitSurfacePatch(
        family="implicit",
        field=field_node,
        bounds=tuple(float(value) for value in bounds),
        metadata=patch_metadata,
    )
    shell = make_surface_shell((patch,), connected=False, metadata={"kernel": {"surface_family": "implicit"}})
    return make_surface_body((shell,), metadata={"kernel": {"surface_family": "implicit", "authoring_boundary": "surface-native"}})


__all__ = [
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
    "SurfaceReferenceEvidenceMatrixReport",
    "SurfaceReferenceFixtureRequirementRecord",
    "SurfaceContinuityRequest",
    "SurfaceContinuitySupportRecord",
    "SurfaceUnsupportedContinuityDiagnostic",
    "PATCH_FAMILY_AVAILABILITY_REQUIRED_OPERATIONS",
    "PATCH_FAMILY_CAPABILITY_MATRIX",
    "PATCH_FAMILY_FEATURE_COVERAGE",
    "PATCH_FAMILY_PROMOTION_CRITERIA",
    "REQUIRED_V1_PATCH_FAMILIES",
    "SURFACE_BODY_COMPLETION_TRACKS",
    "SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS",
    "SUPPORTED_SEAM_CONTINUITY_CLASSES",
    "SUPPORTED_SURFACE_PATCH_FAMILIES",
    "SURFACE_SPEC_66_RETIREMENT_NOTE",
    "assert_patch_family_capability_matrix",
    "assert_patch_family_operation_coverage",
    "audit_all_patch_family_promotion_readiness",
    "audit_patch_family_promotion_readiness",
    "assess_patch_family_availability_promotion",
    "build_surface_unsupported_continuity_diagnostic",
    "evaluate_surface_body_completion_gate",
    "evaluate_surface_body_completion_reference_evidence_matrix",
    "make_surface_body_completion_evidence_from_capabilities",
    "run_patch_family_availability_promotion_pass",
    "surface_continuity_support",
    "surface_body_completion_reference_evidence_matrix",
    "validate_patch_family_availability_gate",
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
    "ImplicitFieldEvaluationDomain",
    "ImplicitFieldEvaluationResult",
    "ImplicitFieldSafetyPolicy",
    "ImplicitFieldValidationDiagnostic",
    "ImplicitSurfacePatch",
    "assess_implicit_field_security",
    "evaluate_implicit_field",
    "evaluate_implicit_field_domain",
    "make_implicit_field_node",
    "validate_implicit_field_security",
    "refine_subdivision_control_cage",
    "classify_surface_seam_continuity",
    "extract_surface_boundary_descriptor",
    "surface_adjacency_from_seams",
    "validate_surface_seam_participation",
    "SurfaceShell",
    "SurfaceBody",
    "make_surface_shell",
    "make_surface_body",
    "make_subdivision_surface",
    "make_implicit_surface",
]
