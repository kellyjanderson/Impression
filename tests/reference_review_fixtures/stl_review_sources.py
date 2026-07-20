"""Loadable source entrypoints for dirty STL reference review fixtures."""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from impression.modeling import (
    Bezier3D,
    DisplacementSurfacePatch,
    HeightmapSurfacePatch,
    Loft,
    Path3D,
    PlanarSurfacePatch,
    Section,
    Station,
    SurfaceBody,
    SurfaceBooleanResult,
    adapt_surface_patch_to_implicit_field,
    as_section,
    boolean_difference,
    boolean_intersection,
    boolean_union,
    classify_surface_csg_loft_eligibility,
    detect_loft_plan_self_intersections,
    compose_displacement_csg_result,
    compose_heightmap_csg_result,
    compose_implicit_field_csg_result,
    handoff_hinge_surface,
    make_bistable_hinge,
    make_box,
    make_cone,
    make_cylinder,
    make_living_hinge,
    make_ngon,
    make_prism,
    make_surface_body,
    make_surface_shell,
    make_traditional_hinge_pair,
    make_sphere,
    make_torus,
    loft,
    loft_plan_sections,
    loft_sections,
    prepare_surface_boolean_difference_operands,
    prepare_surface_boolean_operands,
    rotate,
    scale,
    surface_boolean_result,
    surface_csg_completion_support_matrix,
    translate,
    RuledSurfacePatch,
)
from impression.modeling.csg import (
    SurfaceCSGNoMeshFallbackEvidenceRecord,
    SurfaceSampledImplicitReferenceFixtureRow,
    SurfaceSampledImplicitNoMeshProofRecord,
    enumerate_sampled_implicit_reference_fixture_promotions,
    verify_sampled_implicit_no_mesh_fallback_evidence_gate,
    verify_sampled_implicit_reference_fixture_promotions,
    verify_surface_csg_no_mesh_fallback_evidence,
)
from impression.modeling.drawing2d import make_circle, make_polygon, make_rect
from impression.modeling.drafting import make_arrow
from impression.modeling.heightmap import heightmap
from impression.modeling.text import make_text
from tests.csg_reference_fixtures import (
    build_csg_difference_slot_fixture,
    build_csg_union_box_post_fixture,
    make_sampled_implicit_promotion_target_body,
    make_box_with_subdivision_front_wall,
    make_box_with_higher_order_front_wall,
    make_box_with_sweep_front_wall,
)
from tests.loft_showcases import (
    build_anchor_shift_rectangle_profiles,
    build_branching_manifold_profiles,
    build_cylinder_correspondence_profiles,
    build_dual_cylinder_correspondence_profiles,
    build_notched_cylinder_correspondence_profiles,
    build_notched_phase_shift_cylinder_profiles,
    build_perforated_cylinder_correspondence_profiles,
    build_perforated_vessel_profiles,
    build_phase_shift_cylinder_profiles,
    build_square_correspondence_profiles,
)
from tests.text_font_fixtures import require_glyph_capable_font
from impression.devtools.reference_review import build_section_bundle_fixture_record
from tests.reference_images import build_loft_csg_section_evidence_handoff

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_TEST_PLAN = _PROJECT_ROOT / "project/release-0.1.0a/planning/reference-test-expansion-plan.md"
_DIRTY_FIXTURE_FILE = Path(__file__).with_name("dirty-stl-fixtures.json")
_REFERENCE_ID_PATTERN = re.compile(r"- \[(?P<checked>[ xX])\] `(?P<reference_id>RT-[^`]+)` (?P<description>.+)")


@dataclass(frozen=True)
class ReferenceCSGGapRecord:
    """One CSG reference-test row matched to optional review fixture data."""

    reference_id: str
    description: str
    progression_checked: bool
    fixture_id: str | None = None
    source_entrypoint: str | None = None

    @property
    def fixture_present(self) -> bool:
        return self.fixture_id is not None


@dataclass(frozen=True)
class ReferenceCSGRuntimeSupportResult:
    """Runtime support probe result for one CSG reference row."""

    reference_id: str
    supported: bool
    reason: str


@dataclass(frozen=True)
class ReferenceCSGFixtureReadinessRecord:
    """Fixture readiness classification for one CSG reference row."""

    reference_id: str
    fixture_id: str | None
    source_entrypoint: str | None
    progression_checked: bool
    runtime_supported: bool
    artifact_paths: tuple[Path, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @property
    def ready_for_fixture(self) -> bool:
        return self.runtime_supported and self.progression_checked and self.fixture_id is not None and not self.diagnostics

    @property
    def unsupported_implementation_gap(self) -> bool:
        return not self.runtime_supported or not self.progression_checked


@dataclass(frozen=True)
class PatchFamilyCSGReferenceRow:
    """Reference-facing CSG matrix row with expected artifact policy."""

    operation: str
    left_family: str
    right_family: str
    support_state: str
    artifact_policy: Literal["dirty-stl", "diagnostic-refusal"]
    no_hidden_mesh_fallback_required: bool
    fixture_ready: bool
    diagnostic: str


@dataclass(frozen=True)
class UnsupportedFamilyRefusalFixtureRow:
    """Review-facing diagnostic fixture row for an unsupported CSG route."""

    fixture_id: str
    source_fixture_id: str
    route_kind: Literal["refusal", "unsafe", "malformed"]
    payload_kind: str
    artifact_policy: Literal["diagnostic-refusal"]
    expected_output: Literal["diagnostic evidence"]
    diagnostic: str
    no_mesh_fallback: bool
    fixture_ready: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "artifact_policy": self.artifact_policy,
            "diagnostic": self.diagnostic,
            "expected_output": self.expected_output,
            "fixture_id": self.fixture_id,
            "fixture_ready": self.fixture_ready,
            "no_mesh_fallback": self.no_mesh_fallback,
            "payload_kind": self.payload_kind,
            "route_kind": self.route_kind,
            "source_fixture_id": self.source_fixture_id,
        }


@dataclass(frozen=True)
class NoHiddenMeshFallbackAuditRow:
    """Reference audit row proving a route family crosses mesh only after surface truth."""

    family: str
    evidence_kind: Literal["surface-family-pair", "sampled-implicit-reference"]
    total_records: int
    supported_surface_records: int
    diagnostic_refusal_records: int
    proof_records: int
    passed: bool
    no_mesh_fallback: bool
    diagnostic: str = ""

    def canonical_payload(self) -> dict[str, object]:
        return {
            "diagnostic": self.diagnostic,
            "diagnostic_refusal_records": self.diagnostic_refusal_records,
            "evidence_kind": self.evidence_kind,
            "family": self.family,
            "no_mesh_fallback": self.no_mesh_fallback,
            "passed": self.passed,
            "proof_records": self.proof_records,
            "supported_surface_records": self.supported_surface_records,
            "total_records": self.total_records,
        }


@dataclass(frozen=True)
class LoftSelfIntersectionReferenceRow:
    """Diagnostic reference fixture for a loft that must refuse before export."""

    fixture_id: str
    reference_id: str
    artifact_policy: Literal["diagnostic-refusal"]
    expected_output: Literal["diagnostic evidence"]
    diagnostic_code: str
    diagnostic: str
    valid: bool
    no_mesh_fallback: bool
    branch_crossing_count: float

    def canonical_payload(self) -> dict[str, object]:
        return {
            "artifact_policy": self.artifact_policy,
            "branch_crossing_count": self.branch_crossing_count,
            "diagnostic": self.diagnostic,
            "diagnostic_code": self.diagnostic_code,
            "expected_output": self.expected_output,
            "fixture_id": self.fixture_id,
            "no_mesh_fallback": self.no_mesh_fallback,
            "reference_id": self.reference_id,
            "valid": self.valid,
        }


@dataclass(frozen=True)
class LoftCSGRefusalReferenceRow:
    """Diagnostic reference fixture for a loft CSG route that must refuse."""

    fixture_id: str
    reference_id: str
    artifact_policy: Literal["diagnostic-refusal"]
    expected_output: Literal["diagnostic evidence"]
    operation: str
    diagnostic_code: str
    diagnostic: str
    supported: bool
    no_mesh_fallback: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "artifact_policy": self.artifact_policy,
            "diagnostic": self.diagnostic,
            "diagnostic_code": self.diagnostic_code,
            "expected_output": self.expected_output,
            "fixture_id": self.fixture_id,
            "no_mesh_fallback": self.no_mesh_fallback,
            "operation": self.operation,
            "reference_id": self.reference_id,
            "supported": self.supported,
        }


LoftCsgReferenceGeometryHandoffDiagnosticCode = Literal[
    "non-success-result",
    "missing-result-body",
    "non-surface-body",
    "missing-public-executor-metadata",
    "unaccepted-public-executor",
]


@dataclass(frozen=True)
class LoftCsgReferenceGeometryHandoffDiagnostic:
    """Refusal diagnostic for loft CSG reference geometry handoff."""

    code: LoftCsgReferenceGeometryHandoffDiagnosticCode
    message: str
    fixture_id: str
    operation_id: str
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "fixture_id": self.fixture_id,
            "message": self.message,
            "no_mesh_fallback": self.no_mesh_fallback,
            "operation_id": self.operation_id,
        }


@dataclass(frozen=True)
class LoftCsgReferenceGeometryHandoffRecord:
    """Reference STL handoff gate for accepted public loft CSG result geometry."""

    fixture_id: str
    operation_id: str
    source_path: str
    accepted: bool
    dirty_stl_source_ready: bool
    accepted_body_identity: str | None = None
    result_metadata: dict[str, object] | None = None
    diagnostics: tuple[LoftCsgReferenceGeometryHandoffDiagnostic, ...] = ()
    no_mesh_fallback: bool = True

    def canonical_payload(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "accepted_body_identity": self.accepted_body_identity,
            "diagnostics": [diagnostic.canonical_payload() for diagnostic in self.diagnostics],
            "dirty_stl_source_ready": self.dirty_stl_source_ready,
            "fixture_id": self.fixture_id,
            "no_mesh_fallback": self.no_mesh_fallback,
            "operation_id": self.operation_id,
            "result_metadata": self.result_metadata or {},
            "source_path": self.source_path,
        }


def _loft_csg_handoff_diagnostic(
    code: LoftCsgReferenceGeometryHandoffDiagnosticCode,
    *,
    fixture_id: str,
    operation_id: str,
    message: str,
) -> LoftCsgReferenceGeometryHandoffDiagnostic:
    return LoftCsgReferenceGeometryHandoffDiagnostic(
        code=code,
        fixture_id=fixture_id,
        operation_id=operation_id,
        message=message,
    )


def validate_loft_csg_reference_result_body(
    result: object,
    *,
    fixture_id: str,
    operation_id: str,
) -> LoftCsgReferenceGeometryHandoffDiagnostic | None:
    """Validate that a public result is acceptable as a dirty STL geometry source."""

    if not isinstance(result, SurfaceBooleanResult) or result.status != "succeeded":
        return _loft_csg_handoff_diagnostic(
            "non-success-result",
            fixture_id=fixture_id,
            operation_id=operation_id,
            message="Loft CSG reference handoff requires a succeeded public SurfaceBooleanResult.",
        )
    if result.body is None:
        return _loft_csg_handoff_diagnostic(
            "missing-result-body",
            fixture_id=fixture_id,
            operation_id=operation_id,
            message="Loft CSG reference handoff requires a non-null result body.",
        )
    if not isinstance(result.body, SurfaceBody):
        return _loft_csg_handoff_diagnostic(
            "non-surface-body",
            fixture_id=fixture_id,
            operation_id=operation_id,
            message="Loft CSG reference handoff requires a surface-native result body.",
        )
    metadata = result.body.kernel_metadata()
    executor = metadata.get("loft_primitive_public_executor")
    if not isinstance(executor, dict):
        return _loft_csg_handoff_diagnostic(
            "missing-public-executor-metadata",
            fixture_id=fixture_id,
            operation_id=operation_id,
            message="Loft CSG reference handoff refuses results without public executor metadata.",
        )
    scope = executor.get("execution_scope")
    if not isinstance(scope, dict) or scope.get("accepted") is not True or scope.get("scope") != "trim-fragment-cut":
        return _loft_csg_handoff_diagnostic(
            "unaccepted-public-executor",
            fixture_id=fixture_id,
            operation_id=operation_id,
            message="Loft CSG reference handoff refuses unaccepted public executor results.",
        )
    return None


def build_loft_csg_reference_geometry_handoff(
    *,
    fixture_id: str,
    operation_id: str,
    source_path: Path | str,
    result: object,
) -> LoftCsgReferenceGeometryHandoffRecord:
    """Build a dirty-STL readiness record from an accepted public loft CSG result."""

    diagnostic = validate_loft_csg_reference_result_body(
        result,
        fixture_id=fixture_id,
        operation_id=operation_id,
    )
    if diagnostic is not None:
        return LoftCsgReferenceGeometryHandoffRecord(
            fixture_id=fixture_id,
            operation_id=operation_id,
            source_path=str(source_path),
            accepted=False,
            dirty_stl_source_ready=False,
            diagnostics=(diagnostic,),
        )
    assert isinstance(result, SurfaceBooleanResult)
    assert isinstance(result.body, SurfaceBody)
    metadata = result.body.kernel_metadata()
    return LoftCsgReferenceGeometryHandoffRecord(
        fixture_id=fixture_id,
        operation_id=operation_id,
        source_path=str(source_path),
        accepted=True,
        dirty_stl_source_ready=True,
        accepted_body_identity=result.body.stable_identity,
        result_metadata={
            "boolean_surface_route": metadata.get("boolean_surface_route"),
            "loft_primitive_public_executor": metadata.get("loft_primitive_public_executor"),
        },
    )


def build_loft_csg_reference_geometry_handoff_smoke_record() -> LoftCsgReferenceGeometryHandoffRecord:
    """Build a real public loft CSG result and return its reference-STL handoff record."""

    body = loft(
        [make_circle(radius=0.2), make_circle(radius=0.24)],
        path=[(0.0, 0.0, 0.0), (0.03, 0.01, 0.5)],
        cap_ends=True,
        samples=24,
    )
    cutter = make_box(size=(0.3, 0.3, 0.3), center=(0.0, 0.0, 0.25))
    operands = prepare_surface_boolean_difference_operands(body, [cutter])
    result = surface_boolean_result("difference", operands)
    return build_loft_csg_reference_geometry_handoff(
        fixture_id="loft/csg/rt_loft_csg_reference_handoff_smoke",
        operation_id="RT-LOFT-CSG-HANDOFF",
        source_path=Path(__file__),
        result=result,
    )


def build_loft_csg_section_evidence_readiness_smoke_record():
    """Build a real accepted loft CSG handoff and section-evidence readiness record."""

    handoff = build_loft_csg_reference_geometry_handoff_smoke_record()
    bundle = build_section_bundle_fixture_record(
        bundle_id="loft-csg-section-smoke",
        evidence_kind="loft-section",
        artifact_paths={
            "expected": Path("reference-sections/dirty/loft-csg-section-smoke-expected.png"),
            "actual": Path("reference-sections/dirty/loft-csg-section-smoke-actual.png"),
            "diff": Path("reference-sections/dirty/loft-csg-section-smoke-diff.png"),
        },
        section_plane_metadata={
            "origin": [0.0, 0.0, 0.25],
            "normal": [0.0, 0.0, 1.0],
        },
    )
    return build_loft_csg_section_evidence_handoff(
        fixture_id="loft/csg/rt_loft_csg_section_evidence_smoke",
        operation_id="RT-LOFT-CSG-SECTION",
        accepted_handoff=handoff,
        bundle=bundle,
    )


@dataclass(frozen=True)
class MixedAnalyticRuledFixtureMatrixRow:
    """Representative mixed planar/ruled/revolution review fixture coverage row."""

    fixture_id: str
    operation: str
    left_family: str
    right_family: str
    artifact_path: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "artifact_path": self.artifact_path,
            "fixture_id": self.fixture_id,
            "left_family": self.left_family,
            "operation": self.operation,
            "right_family": self.right_family,
        }


def _load_fixture_rows(fixture_file: Path = _DIRTY_FIXTURE_FILE) -> tuple[dict[str, object], ...]:
    payload = json.loads(fixture_file.read_text())
    return tuple(row for row in payload.get("fixtures", ()) if isinstance(row, dict))


def collect_reference_csg_gap_records(
    *,
    plan_path: Path = _REFERENCE_TEST_PLAN,
    fixture_file: Path = _DIRTY_FIXTURE_FILE,
) -> tuple[ReferenceCSGGapRecord, ...]:
    """Collect RT-CSG progression rows and preserve matched fixture entrypoints."""

    fixture_rows = _load_fixture_rows(fixture_file)
    fixture_by_reference: dict[str, dict[str, object]] = {}
    for row in fixture_rows:
        purpose = str(row.get("purpose", ""))
        fixture_id = str(row.get("fixture_id", ""))
        for match in re.finditer(r"RT-CSG-\d{3}", f"{purpose} {fixture_id}".upper().replace("_", "-")):
            fixture_by_reference.setdefault(match.group(0), row)

    records: list[ReferenceCSGGapRecord] = []
    for line in plan_path.read_text().splitlines():
        match = _REFERENCE_ID_PATTERN.match(line.strip())
        if match is None:
            continue
        reference_id = match.group("reference_id")
        if not reference_id.startswith("RT-CSG"):
            continue
        fixture = fixture_by_reference.get(reference_id)
        records.append(
            ReferenceCSGGapRecord(
                reference_id=reference_id,
                description=match.group("description"),
                progression_checked=match.group("checked").lower() == "x",
                fixture_id=str(fixture["fixture_id"]) if fixture and fixture.get("fixture_id") else None,
                source_entrypoint=str(fixture["entrypoint"]) if fixture and fixture.get("entrypoint") else None,
            )
        )
    return tuple(records)


def probe_reference_csg_runtime_support(
    records: tuple[ReferenceCSGGapRecord, ...],
    probes: dict[str, Callable[[], object]] | None = None,
) -> tuple[ReferenceCSGRuntimeSupportResult, ...]:
    """Run optional reference probes without creating dirty artifacts."""

    probe_map = probes or {}
    results: list[ReferenceCSGRuntimeSupportResult] = []
    for record in records:
        probe = probe_map.get(record.reference_id)
        if probe is None:
            results.append(
                ReferenceCSGRuntimeSupportResult(
                    reference_id=record.reference_id,
                    supported=record.progression_checked and record.fixture_present,
                    reason="fixture-present" if record.fixture_present else "missing-fixture",
                )
            )
            continue
        try:
            result = probe()
        except Exception as exc:  # pragma: no cover - exercised by callers with concrete probes.
            results.append(
                ReferenceCSGRuntimeSupportResult(
                    reference_id=record.reference_id,
                    supported=False,
                    reason=f"probe-error: {exc}",
                )
            )
            continue
        supported = getattr(result, "status", "succeeded") == "succeeded"
        results.append(
            ReferenceCSGRuntimeSupportResult(
                reference_id=record.reference_id,
                supported=supported,
                reason="probe-succeeded" if supported else str(getattr(result, "failure_reason", "probe-unsupported")),
            )
        )
    return tuple(results)


def build_reference_csg_fixture_readiness_report(
    *,
    plan_path: Path = _REFERENCE_TEST_PLAN,
    fixture_file: Path = _DIRTY_FIXTURE_FILE,
    probes: dict[str, Callable[[], object]] | None = None,
) -> tuple[ReferenceCSGFixtureReadinessRecord, ...]:
    """Map CSG progression rows to fixture readiness without broad regeneration."""

    gap_records = collect_reference_csg_gap_records(plan_path=plan_path, fixture_file=fixture_file)
    support_by_id = {
        result.reference_id: result
        for result in probe_reference_csg_runtime_support(gap_records, probes)
    }
    fixture_by_id = {str(row.get("fixture_id")): row for row in _load_fixture_rows(fixture_file)}
    report: list[ReferenceCSGFixtureReadinessRecord] = []
    for record in gap_records:
        support = support_by_id[record.reference_id]
        diagnostics: list[str] = []
        artifact_paths: list[Path] = []
        fixture = fixture_by_id.get(record.fixture_id or "")
        if not record.progression_checked:
            diagnostics.append("progression-unchecked")
        if record.fixture_id is None:
            diagnostics.append("missing-fixture-record")
        if fixture is not None:
            for raw_path in fixture.get("artifact_paths", ()):
                artifact_path = (fixture_file.parent / str(raw_path)).resolve()
                artifact_paths.append(artifact_path)
                if not artifact_path.exists():
                    diagnostics.append(f"missing-artifact:{artifact_path}")
        report.append(
            ReferenceCSGFixtureReadinessRecord(
                reference_id=record.reference_id,
                fixture_id=record.fixture_id,
                source_entrypoint=record.source_entrypoint,
                progression_checked=record.progression_checked,
                runtime_supported=support.supported,
                artifact_paths=tuple(artifact_paths),
                diagnostics=tuple(diagnostics),
            )
        )
    return tuple(report)


def build_patch_family_csg_reference_matrix(
    *,
    families: tuple[str, ...] | None = None,
) -> tuple[PatchFamilyCSGReferenceRow, ...]:
    """Format solver support rows for reference fixture planning."""

    selected = set(families) if families is not None else None
    rows: list[PatchFamilyCSGReferenceRow] = []
    for support in surface_csg_completion_support_matrix():
        if selected is not None and (support.left_family not in selected or support.right_family not in selected):
            continue
        artifact_policy: Literal["dirty-stl", "diagnostic-refusal"] = "dirty-stl" if support.supported else "diagnostic-refusal"
        diagnostic = "" if support.supported else (
            support.required_future_capability
            or f"{support.operation} {support.left_family}/{support.right_family} is not executable as surface CSG."
        )
        rows.append(
            PatchFamilyCSGReferenceRow(
                operation=support.operation,
                left_family=support.left_family,
                right_family=support.right_family,
                support_state=support.support_state,
                artifact_policy=artifact_policy,
                no_hidden_mesh_fallback_required=True,
                fixture_ready=support.supported,
                diagnostic=diagnostic,
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (row.operation, row.left_family, row.right_family),
        )
    )


_UNSUPPORTED_REFUSAL_REVIEW_IDS: dict[str, str] = {
    "refusal": "surfacebody/csg/rt_patch_csg_013_non_csg_replacement_refusal",
    "unsafe": "surfacebody/csg/rt_patch_csg_013_unsafe_implicit_refusal",
    "malformed": "surfacebody/csg/rt_patch_csg_013_malformed_promotion_refusal",
}


def build_unsupported_family_refusal_fixture_matrix() -> tuple[UnsupportedFamilyRefusalFixtureRow, ...]:
    """Return diagnostic-only review fixtures for unsupported CSG route evidence."""

    report = verify_sampled_implicit_reference_fixture_promotions()
    rows_by_kind: dict[str, SurfaceSampledImplicitReferenceFixtureRow] = {
        row.route_kind: row for row in enumerate_sampled_implicit_reference_fixture_promotions()
    }
    diagnostics_by_kind = {
        "refusal": (
            "Unsupported sampled/implicit family request is represented as a deliberate non-CSG replacement "
            "refusal instead of a success STL."
        ),
        "unsafe": (
            "Unsafe implicit composition refuses with diagnostic evidence when the sampling budget cannot prove "
            "a safe field result."
        ),
        "malformed": (
            "Malformed sampled/implicit promotion policy refuses deterministically when the requested target "
            "family is missing."
        ),
    }
    fixtures: list[UnsupportedFamilyRefusalFixtureRow] = []
    for route_kind in ("refusal", "unsafe", "malformed"):
        row = rows_by_kind[route_kind]
        fixtures.append(
            UnsupportedFamilyRefusalFixtureRow(
                fixture_id=_UNSUPPORTED_REFUSAL_REVIEW_IDS[route_kind],
                source_fixture_id=row.fixture_id,
                route_kind=route_kind,
                payload_kind=row.payload_kind,
                artifact_policy="diagnostic-refusal",
                expected_output="diagnostic evidence",
                diagnostic=diagnostics_by_kind[route_kind],
                no_mesh_fallback=row.no_mesh_fallback,
                fixture_ready=report.passed and row.passed and row.reference_state == "clean" and row.no_mesh_fallback,
            )
        )
    return tuple(fixtures)


def _unsupported_family_refusal_payload(route_kind: Literal["refusal", "unsafe", "malformed"]) -> dict[str, object]:
    row = next(row for row in build_unsupported_family_refusal_fixture_matrix() if row.route_kind == route_kind)
    return row.canonical_payload()


def build_surfacebody_csg_rt_patch_csg_013_non_csg_replacement_refusal() -> dict[str, object]:
    return _unsupported_family_refusal_payload("refusal")


def build_surfacebody_csg_rt_patch_csg_013_unsafe_implicit_refusal() -> dict[str, object]:
    return _unsupported_family_refusal_payload("unsafe")


def build_surfacebody_csg_rt_patch_csg_013_malformed_promotion_refusal() -> dict[str, object]:
    return _unsupported_family_refusal_payload("malformed")


_ADVANCED_NO_MESH_FALLBACK_FAMILIES = (
    "bspline",
    "nurbs",
    "sweep",
    "subdivision",
    "implicit",
    "heightmap",
    "displacement",
)


def _surface_family_no_mesh_audit_row(family: str) -> NoHiddenMeshFallbackAuditRow:
    report = verify_surface_csg_no_mesh_fallback_evidence(families=("planar", family))
    relevant: tuple[SurfaceCSGNoMeshFallbackEvidenceRecord, ...] = tuple(
        record for record in report.evidence if family in {record.left_family, record.right_family}
    )
    hidden_mesh = [record for record in relevant if record.mesh_fallback_attempted]
    diagnostics = list(report.diagnostics)
    return NoHiddenMeshFallbackAuditRow(
        family=family,
        evidence_kind="surface-family-pair",
        total_records=len(relevant),
        supported_surface_records=sum(1 for record in relevant if record.result_kind == "supported-surface"),
        diagnostic_refusal_records=sum(1 for record in relevant if record.result_kind == "diagnostic-refusal"),
        proof_records=0,
        passed=bool(relevant) and report.passed and not hidden_mesh,
        no_mesh_fallback=not hidden_mesh,
        diagnostic="; ".join(diagnostic.message for diagnostic in diagnostics),
    )


def build_no_hidden_mesh_fallback_audit_matrix() -> tuple[NoHiddenMeshFallbackAuditRow, ...]:
    """Return bounded reference audit rows for advanced CSG no-hidden-mesh evidence."""

    rows = [_surface_family_no_mesh_audit_row(family) for family in _ADVANCED_NO_MESH_FALLBACK_FAMILIES]
    gate = verify_sampled_implicit_no_mesh_fallback_evidence_gate()
    proofs: tuple[SurfaceSampledImplicitNoMeshProofRecord, ...] = gate.proofs
    rows.append(
        NoHiddenMeshFallbackAuditRow(
            family="sampled-implicit-reference",
            evidence_kind="sampled-implicit-reference",
            total_records=len(proofs),
            supported_surface_records=0,
            diagnostic_refusal_records=0,
            proof_records=len(proofs),
            passed=gate.passed,
            no_mesh_fallback=all(proof.no_mesh_fallback for proof in proofs),
            diagnostic="; ".join(diagnostic.message for diagnostic in gate.diagnostics),
        )
    )
    return tuple(rows)


def build_loft_rt_loft_037_self_intersection_diagnostic() -> LoftSelfIntersectionReferenceRow:
    """Build the RT-LOFT-037 diagnostic evidence row without exporting an STL."""

    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = (
        Station(t=0.0, section=base, origin=(0.0, 0.0, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=1.0, section=base, origin=(0.0, 0.0, 1.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
    )
    clean_plan = loft_plan_sections(stations, samples=16, split_merge_mode="resolve")
    metadata = dict(clean_plan.metadata)
    metadata["fairness_diagnostics"] = {
        **dict(metadata["fairness_diagnostics"]),
        "branch_crossing_count": 1.0,
    }
    report = detect_loft_plan_self_intersections(replace(clean_plan, metadata=metadata))
    diagnostic = report.diagnostics[0] if report.diagnostics else None
    return LoftSelfIntersectionReferenceRow(
        fixture_id="loft/rt_loft_037_self_intersection_diagnostic",
        reference_id="RT-LOFT-037",
        artifact_policy="diagnostic-refusal",
        expected_output="diagnostic evidence",
        diagnostic_code="" if diagnostic is None else diagnostic.code,
        diagnostic="" if diagnostic is None else diagnostic.message,
        valid=report.valid,
        no_mesh_fallback=report.no_mesh_fallback and all(item.no_mesh_fallback for item in report.diagnostics),
        branch_crossing_count=report.branch_crossing_count,
    )


def build_loft_rt_loft_037_self_intersection_reference() -> dict[str, object]:
    return build_loft_rt_loft_037_self_intersection_diagnostic().canonical_payload()


def build_loft_rt_loft_csg_014_underconstrained_branch_refusal() -> LoftCSGRefusalReferenceRow:
    """Build the RT-LOFT-CSG-014 underconstrained topology refusal evidence row."""

    body = make_surface_body(
        (
            make_surface_shell(
                (PlanarSurfacePatch(family="planar"),),
                metadata={"kernel": {"operation": "loft", "executor": "surface"}},
            ),
        ),
        metadata={"kernel": {"operation": "loft", "executor": "surface"}},
    )
    record = classify_surface_csg_loft_eligibility(body, "difference")
    return LoftCSGRefusalReferenceRow(
        fixture_id="loft/csg/rt_loft_csg_014_underconstrained_branch_refusal",
        reference_id="RT-LOFT-CSG-014",
        artifact_policy="diagnostic-refusal",
        expected_output="diagnostic evidence",
        operation=record.operation,
        diagnostic_code=record.code,
        diagnostic=record.message,
        supported=record.supported,
        no_mesh_fallback=record.no_mesh_fallback,
    )


def build_loft_rt_loft_csg_014_underconstrained_branch_reference() -> dict[str, object]:
    return build_loft_rt_loft_csg_014_underconstrained_branch_refusal().canonical_payload()


def _loft_csg_success_body(operation: str, operands: object) -> SurfaceBody:
    result = surface_boolean_result(operation, operands)  # type: ignore[arg-type]
    return _surface_boolean_body(result)


def _loft_csg_refusal_payload(
    *,
    fixture_id: str,
    reference_id: str,
    operation: str,
    result: object,
) -> dict[str, object]:
    if isinstance(result, Exception):
        diagnostic_code = result.__class__.__name__
        diagnostic = str(result)
    else:
        diagnostic_code = str(getattr(result, "status", "unsupported"))
        diagnostic = str(getattr(result, "failure_reason", "Loft CSG route refused before reference STL export."))
    return LoftCSGRefusalReferenceRow(
        fixture_id=fixture_id,
        reference_id=reference_id,
        artifact_policy="diagnostic-refusal",
        expected_output="diagnostic evidence",
        operation=operation,
        diagnostic_code=diagnostic_code,
        diagnostic=diagnostic,
        supported=False,
        no_mesh_fallback=True,
    ).canonical_payload()


def _loft_cylinder_csg_body() -> SurfaceBody:
    return build_loft_cylinder_correspondence()


def build_loft_csg_rt_loft_csg_001_cylinder_difference_box_slot() -> SurfaceBody:
    body = _loft_cylinder_csg_body()
    cutter = make_box(size=(0.34, 0.34, 0.44), center=(0.0, 0.0, 0.25))
    return _loft_csg_success_body("difference", prepare_surface_boolean_difference_operands(body, [cutter]))


def build_loft_csg_rt_loft_csg_001_cylinder_difference_box_slot_reference() -> dict[str, object]:
    body = _loft_cylinder_csg_body()
    cutter = make_box(size=(0.34, 0.34, 0.44), center=(0.0, 0.0, 0.25))
    result = surface_boolean_result("difference", prepare_surface_boolean_difference_operands(body, [cutter]))
    return _loft_csg_refusal_payload(
        fixture_id="loft/csg/rt_loft_csg_001_cylinder_difference_box_slot",
        reference_id="RT-LOFT-CSG-001",
        operation="difference",
        result=result,
    )


def build_loft_csg_rt_loft_csg_002_cylinder_difference_cross_drilled_cylinder_reference() -> dict[str, object]:
    body = _loft_cylinder_csg_body()
    cutter = make_cylinder(radius=0.1, height=0.7, center=(0.0, 0.0, 0.25))
    result = surface_boolean_result("difference", prepare_surface_boolean_difference_operands(body, [cutter]))
    return _loft_csg_refusal_payload(
        fixture_id="loft/csg/rt_loft_csg_002_cylinder_difference_cross_drilled_cylinder",
        reference_id="RT-LOFT-CSG-002",
        operation="difference",
        result=result,
    )


def build_loft_csg_rt_loft_csg_003_vessel_difference_sphere_scoop_reference() -> dict[str, object]:
    body = build_loft_matrix_bulb_circle()
    cutter = make_sphere(radius=0.25, center=(0.0, 0.0, 0.25))
    result = surface_boolean_result("difference", prepare_surface_boolean_difference_operands(body, [cutter]))
    return _loft_csg_refusal_payload(
        fixture_id="loft/csg/rt_loft_csg_003_vessel_difference_sphere_scoop",
        reference_id="RT-LOFT-CSG-003",
        operation="difference",
        result=result,
    )


def build_loft_csg_rt_loft_csg_004_hourglass_intersection_box() -> SurfaceBody:
    body = build_loft_hourglass_vessel()
    cutter = make_box(size=(0.6, 0.6, 0.75), center=(0.0, 0.0, 0.45))
    return _loft_csg_success_body("intersection", prepare_surface_boolean_operands("intersection", (body, cutter)))


def build_loft_csg_rt_loft_csg_004_hourglass_intersection_box_reference() -> dict[str, object]:
    body = build_loft_hourglass_vessel()
    cutter = make_box(size=(0.6, 0.6, 0.75), center=(0.0, 0.0, 0.45))
    try:
        result: object = surface_boolean_result("intersection", prepare_surface_boolean_operands("intersection", (body, cutter)))
    except Exception as exc:
        result = exc
    return _loft_csg_refusal_payload(
        fixture_id="loft/csg/rt_loft_csg_004_hourglass_intersection_box",
        reference_id="RT-LOFT-CSG-004",
        operation="intersection",
        result=result,
    )


def build_loft_csg_rt_loft_csg_005_square_correspondence_union_post() -> SurfaceBody:
    body = build_loft_square_correspondence()
    post = make_box(size=(0.28, 0.28, 0.5), center=(0.18, 0.0, 0.24))
    return _loft_csg_success_body("union", prepare_surface_boolean_operands("union", (body, post)))


def build_loft_csg_rt_loft_csg_005_square_correspondence_union_post_reference() -> dict[str, object]:
    body = build_loft_square_correspondence()
    post = make_box(size=(0.28, 0.28, 0.5), center=(0.18, 0.0, 0.24))
    try:
        result: object = surface_boolean_result("union", prepare_surface_boolean_operands("union", (body, post)))
    except Exception as exc:
        result = exc
    return _loft_csg_refusal_payload(
        fixture_id="loft/csg/rt_loft_csg_005_square_correspondence_union_post",
        reference_id="RT-LOFT-CSG-005",
        operation="union",
        result=result,
    )


def build_loft_csg_rt_loft_csg_006_phase_shift_difference_vertical_slot() -> SurfaceBody:
    body = build_loft_phase_shift_cylinder()
    cutter = make_box(size=(0.28, 0.42, 0.9), center=(0.0, 0.0, 0.35))
    return _loft_csg_success_body("difference", prepare_surface_boolean_difference_operands(body, [cutter]))


def build_loft_csg_rt_loft_csg_006_phase_shift_difference_vertical_slot_reference() -> dict[str, object]:
    body = build_loft_phase_shift_cylinder()
    cutter = make_box(size=(0.28, 0.42, 0.9), center=(0.0, 0.0, 0.35))
    result = surface_boolean_result("difference", prepare_surface_boolean_difference_operands(body, [cutter]))
    return _loft_csg_refusal_payload(
        fixture_id="loft/csg/rt_loft_csg_006_phase_shift_difference_vertical_slot",
        reference_id="RT-LOFT-CSG-006",
        operation="difference",
        result=result,
    )


def build_loft_csg_rt_loft_csg_007_branch_difference_sphere_joint_reference() -> dict[str, object]:
    body = build_loft_branching_manifold()
    cutter = make_sphere(radius=0.24, center=(0.0, 0.0, 0.72))
    result = surface_boolean_result("difference", prepare_surface_boolean_difference_operands(body, [cutter]))
    return _loft_csg_refusal_payload(
        fixture_id="loft/csg/rt_loft_csg_007_branch_difference_sphere_joint",
        reference_id="RT-LOFT-CSG-007",
        operation="difference",
        result=result,
    )


def build_loft_csg_rt_loft_csg_008_branch_intersection_cutter_window_reference() -> dict[str, object]:
    body = build_loft_branching_manifold()
    cutter = make_box(size=(0.7, 0.7, 0.55), center=(0.0, 0.0, 0.72))
    result = surface_boolean_result("intersection", prepare_surface_boolean_operands("intersection", (body, cutter)))
    return _loft_csg_refusal_payload(
        fixture_id="loft/csg/rt_loft_csg_008_branch_intersection_cutter_window",
        reference_id="RT-LOFT-CSG-008",
        operation="intersection",
        result=result,
    )


def build_loft_csg_rt_loft_csg_009_loft_union_loft_overlapping_ruled() -> SurfaceBody:
    first = build_loft_cylinder_correspondence()
    second = translate(build_loft_matrix_tapered_circle(), (0.05, 0.03, 0.05))
    return _loft_csg_success_body("union", prepare_surface_boolean_operands("union", (first, second)))


def build_loft_csg_rt_loft_csg_010_loft_intersection_loft_crossing_axes() -> SurfaceBody:
    first = build_loft_cylinder_correspondence()
    second = translate(build_loft_matrix_s_curve_station_path(), (0.02, 0.0, -0.05))
    return _loft_csg_success_body("intersection", prepare_surface_boolean_operands("intersection", (first, second)))


def build_loft_csg_rt_loft_csg_011_loft_difference_loft_shared_station() -> SurfaceBody:
    first = build_loft_cylinder_correspondence()
    second = translate(build_loft_matrix_tapered_circle(), (0.0, 0.0, 0.0))
    return _loft_csg_success_body("difference", prepare_surface_boolean_difference_operands(first, [second]))


def build_loft_csg_rt_loft_csg_012_authored_color_preserved() -> SurfaceBody:
    body = replace(build_loft_square_correspondence(), metadata={"consumer": {"color": "#d9822b"}})
    cutter = replace(translate(build_loft_matrix_tapered_circle(), (0.04, 0.0, 0.0)), metadata={"consumer": {"color": "#5b84b1"}})
    return _loft_csg_success_body("union", prepare_surface_boolean_operands("union", (body, cutter)))


def build_loft_csg_rt_loft_csg_013_section_evidence_reference() -> dict[str, object]:
    return build_loft_csg_section_evidence_readiness_smoke_record().canonical_payload()


def build_mixed_planar_ruled_revolution_fixture_matrix() -> tuple[MixedAnalyticRuledFixtureMatrixRow, ...]:
    """Return the representative mixed analytic/ruled fixtures required by Surface Spec 386."""

    fixture_by_id = {str(row.get("fixture_id")): row for row in _load_fixture_rows()}
    required = (
        (
            "surfacebody/csg/rt_patch_csg_001_planar_box_sphere_difference",
            "difference",
            "planar",
            "revolution",
        ),
        (
            "surfacebody/csg/rt_patch_csg_003_revolution_cylinder_box_difference",
            "difference",
            "revolution",
            "planar",
        ),
        (
            "surfacebody/csg/rt_patch_csg_002_loft_ruled_box_difference",
            "difference",
            "ruled",
            "planar",
        ),
    )
    rows: list[MixedAnalyticRuledFixtureMatrixRow] = []
    for fixture_id, operation, left_family, right_family in required:
        fixture = fixture_by_id[fixture_id]
        artifact_paths = fixture.get("artifact_paths", ())
        rows.append(
            MixedAnalyticRuledFixtureMatrixRow(
                fixture_id=fixture_id,
                operation=operation,
                left_family=left_family,
                right_family=right_family,
                artifact_path=str(artifact_paths[0]),
            )
        )
    return tuple(rows)


def _surface_boolean_body(result: object) -> SurfaceBody:
    if getattr(result, "status", None) != "succeeded":
        raise ValueError(f"Surface CSG fixture did not succeed: {getattr(result, 'failure_reason', None)}")
    body = getattr(result, "body", None)
    if not isinstance(body, SurfaceBody):
        raise TypeError("Surface CSG fixture did not return a SurfaceBody.")
    return body


def _surface_csg_success_body(result: object, *, route_name: str) -> SurfaceBody:
    if not getattr(result, "supported", False):
        diagnostics = getattr(result, "diagnostics", ())
        raise ValueError(f"{route_name} fixture did not produce supported CSG evidence: {diagnostics}")
    body = getattr(result, "body", None)
    if not isinstance(body, SurfaceBody):
        raise TypeError(f"{route_name} fixture did not return a SurfaceBody.")
    return body


def _loft_from_profiles(builder):
    profiles, path = builder()
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=True,
    )


def _combine_surface_bodies(bodies: list[SurfaceBody]) -> SurfaceBody:
    return make_surface_body(tuple(shell for body in bodies for shell in body.iter_shells(world=True)))


def _rounded_rectangle_profile(width: float, height: float, radius: float, *, samples_per_corner: int = 8):
    points: list[tuple[float, float]] = []
    corners = (
        (width / 2.0 - radius, height / 2.0 - radius, 0.0, np.pi / 2.0),
        (-width / 2.0 + radius, height / 2.0 - radius, np.pi / 2.0, np.pi),
        (-width / 2.0 + radius, -height / 2.0 + radius, np.pi, 3.0 * np.pi / 2.0),
        (width / 2.0 - radius, -height / 2.0 + radius, 3.0 * np.pi / 2.0, 2.0 * np.pi),
    )
    for cx, cy, start, stop in corners:
        for angle in np.linspace(start, stop, samples_per_corner, endpoint=False):
            points.append((cx + radius * float(np.cos(angle)), cy + radius * float(np.sin(angle))))
    return make_polygon(points)


def _star_profile(outer_radius: float, inner_radius: float, points: int = 5):
    coords: list[tuple[float, float]] = []
    for index in range(points * 2):
        radius = outer_radius if index % 2 == 0 else inner_radius
        angle = (np.pi / 2.0) + index * np.pi / points
        coords.append((radius * float(np.cos(angle)), radius * float(np.sin(angle))))
    return make_polygon(coords)


def build_loft_anchor_shift_rectangle():
    return _loft_from_profiles(build_anchor_shift_rectangle_profiles)


def build_loft_branching_manifold():
    profiles, path = build_branching_manifold_profiles()
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=True,
        split_merge_mode="resolve",
    )


def build_loft_cylinder_correspondence():
    return _loft_from_profiles(build_cylinder_correspondence_profiles)


def build_loft_hourglass_vessel():
    module_path = _PROJECT_ROOT / "docs/examples/loft/real_world/loft_hourglass_vessel_example.py"
    spec = importlib.util.spec_from_file_location("loft_hourglass_vessel_example", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_surface_body(module.TEST_PARAMETERS, module.TEST_QUALITY)


def build_loft_phase_shift_cylinder():
    return _loft_from_profiles(build_phase_shift_cylinder_profiles)


def build_loft_square_correspondence():
    return _loft_from_profiles(build_square_correspondence_profiles)


def build_loft_matrix_circle_to_circle():
    profiles = [
        make_circle(radius=0.48),
        make_circle(radius=0.62, center=(0.04, -0.03)),
        make_circle(radius=0.54, center=(-0.03, 0.02)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.08, 0.03, 0.9), (0.16, -0.02, 1.8)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_square_to_square():
    profiles = [
        make_rect(size=(0.9, 0.9)),
        make_rect(size=(1.1, 0.7), center=(0.04, -0.02)),
        make_rect(size=(0.8, 1.0), center=(-0.02, 0.03)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.1, -0.04, 0.85), (0.18, 0.02, 1.7)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_circle_to_square():
    profiles = [
        make_circle(radius=0.55),
        make_polygon([(-0.55, -0.35), (0.55, -0.35), (0.55, 0.35), (-0.55, 0.35)]),
        make_rect(size=(0.9, 0.9), center=(0.03, -0.02)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.06, 0.03, 0.85), (0.12, -0.02, 1.7)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_rectangle_to_rounded_rectangle():
    profiles = [
        make_rect(size=(1.0, 0.62)),
        _rounded_rectangle_profile(1.06, 0.72, 0.12),
        _rounded_rectangle_profile(0.84, 0.58, 0.18),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.06, 0.02, 0.82), (0.1, -0.02, 1.68)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_triangle_to_hexagon():
    triangle = make_polygon([(0.0, 0.62), (-0.58, -0.36), (0.58, -0.36)])
    hexagon_points = [
        (0.62 * float(np.cos(angle)), 0.62 * float(np.sin(angle)))
        for angle in np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    ]
    profiles = [triangle, make_polygon(hexagon_points), make_rect(size=(0.75, 0.55))]
    path = np.asarray([(0.0, 0.0, 0.0), (0.04, 0.04, 0.78), (0.1, -0.03, 1.65)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_open_ended_circle():
    profiles = [make_circle(radius=0.42), make_circle(radius=0.58), make_circle(radius=0.38)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.04, 0.06, 0.78), (0.08, -0.04, 1.62)], dtype=float)
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=False,
        split_merge_mode="resolve",
    )


def build_loft_matrix_capped_circle_planar_caps():
    profiles = [make_circle(radius=0.46), make_circle(radius=0.5), make_circle(radius=0.42)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.01, 0.7), (0.04, -0.01, 1.4)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_tapered_circle():
    profiles = [make_circle(radius=0.72), make_circle(radius=0.48), make_circle(radius=0.28)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.03, 0.0, 0.78), (0.04, 0.0, 1.65)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_bulb_circle():
    profiles = [
        make_circle(radius=0.34),
        make_circle(radius=0.78, center=(0.03, 0.0)),
        make_circle(radius=0.42, center=(-0.02, 0.02)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.03, 0.75), (0.05, -0.03, 1.6)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_high_twist_rectangle():
    profiles = []
    for angle in (0.0, 45.0, 105.0, 170.0):
        radians = np.deg2rad(angle)
        base = [(-0.45, -0.28), (0.45, -0.28), (0.45, 0.28), (-0.45, 0.28)]
        rotated = [
            (
                x * float(np.cos(radians)) - y * float(np.sin(radians)),
                x * float(np.sin(radians)) + y * float(np.cos(radians)),
            )
            for x, y in base
        ]
        profiles.append(make_polygon(rotated))
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.02, 0.55), (0.0, -0.04, 1.12), (0.05, 0.0, 1.72)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_s_curve_station_path():
    profiles = [
        make_circle(radius=0.38),
        make_circle(radius=0.48, center=(0.02, -0.01)),
        make_circle(radius=0.44, center=(-0.03, 0.02)),
        make_circle(radius=0.36),
    ]
    path = np.asarray(
        [(0.0, 0.0, 0.0), (0.32, 0.16, 0.58), (-0.28, -0.14, 1.38), (0.18, 0.08, 2.25)],
        dtype=float,
    )
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_smooth_profile_sequence():
    profiles = [
        make_circle(radius=0.44),
        _rounded_rectangle_profile(0.88, 0.72, 0.28),
        make_circle(radius=0.5, center=(0.02, -0.02)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.08, 0.02, 0.82), (0.12, -0.02, 1.6)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_mixed_sharp_smooth():
    profiles = [
        _star_profile(0.48, 0.26),
        _rounded_rectangle_profile(0.86, 0.64, 0.18),
        make_circle(radius=0.46),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.04, 0.04, 0.74), (0.1, -0.02, 1.52)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_helical_station_path():
    profiles = [make_circle(radius=0.2 + 0.03 * (index % 2)) for index in range(6)]
    angles = np.linspace(0.0, 1.5 * np.pi, len(profiles))
    path = np.asarray(
        [(0.34 * float(np.cos(angle)), 0.34 * float(np.sin(angle)), 0.34 * index) for index, angle in enumerate(angles)],
        dtype=float,
    )
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_very_short_span():
    profiles = [make_rect(size=(0.72, 0.5)), make_rect(size=(0.64, 0.44), center=(0.02, -0.01))]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.0, 0.08)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_nonuniform_station_spacing():
    profiles = [
        make_circle(radius=0.34),
        make_circle(radius=0.48),
        make_circle(radius=0.52),
        make_circle(radius=0.38),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.01, 0.12), (0.08, -0.03, 1.35), (0.1, 0.0, 1.62)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_branching_manifold_open():
    profiles, path = build_branching_manifold_profiles()
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=False,
        split_merge_mode="resolve",
    )


def build_loft_matrix_sharp_corner_star():
    profiles = [
        _star_profile(0.55, 0.28),
        _star_profile(0.48, 0.22),
        _star_profile(0.6, 0.32),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.03, -0.03, 0.76), (0.0, 0.02, 1.5)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_tiny_to_normal_profile():
    profiles = [make_circle(radius=0.08), make_circle(radius=0.28), make_circle(radius=0.62)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.0, 0.55), (0.04, 0.0, 1.55)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_hole_to_solid_transition():
    from tests.loft_showcases import make_perforated_disc

    profiles = [
        make_perforated_disc(outer_radius=0.72, holes=[(0.14, (0.0, 0.0))]),
        make_circle(radius=0.64),
        make_circle(radius=0.56),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.03, 0.02, 0.72), (0.08, -0.02, 1.5)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_explicit_station_frames():
    profiles = [as_section(make_circle(radius=0.46)), as_section(make_rect(size=(0.88, 0.58))), as_section(make_circle(radius=0.38))]
    half_root = float(np.sqrt(0.5))
    cos_30 = float(np.sqrt(3.0) / 2.0)
    stations = [
        Station(t=0.0, section=profiles[0], origin=(0.0, 0.0, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=0.45, section=profiles[1], origin=(0.2, 0.0, 0.8), u=(cos_30, 0.5, 0.0), v=(-0.5, cos_30, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=1.0, section=profiles[2], origin=(0.44, 0.18, 1.65), u=(half_root, half_root, 0.0), v=(-half_root, half_root, 0.0), n=(0.0, 0.0, 1.0)),
    ]
    return Loft(progression=[0.0, 0.45, 1.0], stations=stations, topology=profiles, cap_ends=True)


def build_loft_matrix_tight_curvature_frame_transport():
    profiles = [make_circle(radius=0.18 + 0.025 * (index % 2)) for index in range(7)]
    angles = np.linspace(0.0, 1.75 * np.pi, len(profiles))
    path = np.asarray(
        [(0.28 * float(np.cos(angle)), 0.28 * float(np.sin(angle)), 0.18 * index) for index, angle in enumerate(angles)],
        dtype=float,
    )
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_branch_merge_many_to_one():
    profiles, path = build_branching_manifold_profiles()
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path[::-1].copy(),
        topology=list(reversed(profiles)),
        cap_ends=True,
        split_merge_mode="resolve",
    )


def build_loft_matrix_asymmetric_branch_lengths():
    trunk = Section((as_section(make_circle(radius=0.58)).regions[0],))
    split = Section((as_section(make_circle(radius=0.34, center=(-0.55, 0.0))).regions[0], as_section(make_circle(radius=0.3, center=(0.55, 0.0))).regions[0]))
    stretched = Section((as_section(make_circle(radius=0.28, center=(-1.18, 0.38))).regions[0], as_section(make_circle(radius=0.34, center=(0.72, -0.1))).regions[0]))
    profiles = [trunk, split, stretched]
    path = np.asarray([(0.0, 0.0, 0.0), (0.08, 0.04, 0.9), (0.38, 0.18, 2.1)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_near_coincident_stations():
    profiles = [make_circle(radius=0.3), make_circle(radius=0.32), make_circle(radius=0.5)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.002, 0.0, 0.015), (0.04, 0.0, 1.05)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_curved_bezier_path():
    path = Path3D(
        [
            Bezier3D(
                np.asarray([0.0, 0.0, 0.0], dtype=float),
                np.asarray([0.45, 0.2, 0.55], dtype=float),
                np.asarray([-0.35, 0.35, 1.25], dtype=float),
                np.asarray([0.2, -0.1, 1.9], dtype=float),
            )
        ]
    )
    profiles = [make_circle(radius=0.34), make_circle(radius=0.48), make_circle(radius=0.32)]
    return loft(profiles, path=path, samples=48, bezier_samples=48, cap_ends=True)


def build_loft_matrix_many_to_many_region_expand():
    left = make_rect(size=(0.85, 0.85), center=(-1.0, 0.0))
    right = make_rect(size=(0.85, 0.85), center=(1.0, 0.0))
    left_end = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    center_end = make_rect(size=(0.7, 0.7), center=(0.0, 0.0))
    right_end = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    start = Section((as_section(left).regions[0], as_section(right).regions[0]))
    end = Section(
        (
            as_section(left_end).regions[0],
            as_section(center_end).regions[0],
            as_section(right_end).regions[0],
        )
    )
    stations = [
        Station(t=0.0, section=start, origin=(0.0, 0.0, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=1.0, section=end, origin=(0.0, 0.0, 1.5), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
    ]
    return loft_sections(stations, samples=30, cap_ends=True, split_merge_mode="resolve", split_merge_steps=8)


def build_loft_matrix_many_to_many_region_collapse():
    left = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    center = make_rect(size=(0.7, 0.7), center=(0.0, 0.0))
    right = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    left_end = make_rect(size=(0.85, 0.85), center=(-1.0, 0.0))
    right_end = make_rect(size=(0.85, 0.85), center=(1.0, 0.0))
    start = Section(
        (
            as_section(left).regions[0],
            as_section(center).regions[0],
            as_section(right).regions[0],
        )
    )
    end = Section((as_section(left_end).regions[0], as_section(right_end).regions[0]))
    stations = [
        Station(t=0.0, section=start, origin=(0.0, 0.0, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=1.0, section=end, origin=(0.0, 0.0, 1.5), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
    ]
    return loft_sections(stations, samples=30, cap_ends=True, split_merge_mode="resolve", split_merge_steps=8)


def build_loft_matrix_deterministic_resampling():
    circle_points = [
        (0.5 * float(np.cos(angle)), 0.5 * float(np.sin(angle)))
        for angle in np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False)
    ]
    profiles = [make_polygon(circle_points), make_rect(size=(0.75, 0.75)), make_circle(radius=0.42)]
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.1, 0.05, 0.8), (0.16, -0.02, 1.6)])
    return loft(profiles, path=path, samples=44, cap_ends=True)


def build_loft_matrix_domed_tapered_caps():
    profiles = [make_circle(radius=0.44), make_circle(radius=0.56), make_circle(radius=0.38)]
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.08, 0.02, 0.9), (0.12, -0.02, 1.8)])
    return loft(
        profiles,
        path=path,
        samples=40,
        start_cap="dome",
        end_cap="taper",
        start_cap_length=0.5,
        end_cap_length=0.45,
        cap_scale_dims="both",
    )


def build_loft_matrix_slope_dome_caps():
    profiles = [make_rect(size=(0.8, 0.55)), make_rect(size=(0.95, 0.75)), make_rect(size=(0.58, 0.45))]
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.05, 0.06, 0.8), (0.0, 0.0, 1.55)])
    return loft(
        profiles,
        path=path,
        samples=36,
        start_cap="slope",
        end_cap="dome",
        start_cap_length=0.38,
        end_cap_length=0.5,
        cap_scale_dims="smallest",
    )


def build_loft_matrix_reversed_winding_repair():
    square = [(-0.45, -0.45), (-0.45, 0.45), (0.45, 0.45), (0.45, -0.45)]
    profiles = [make_polygon(square), make_polygon(list(reversed(square))), make_polygon(square)]
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.02, 0.0, 0.7), (0.05, 0.0, 1.4)])
    return loft(profiles, path=path, samples=32, cap_ends=True)


def build_loft_matrix_notched_cylinder_correspondence():
    return _loft_from_profiles(build_notched_cylinder_correspondence_profiles)


def build_loft_matrix_notched_phase_shift_cylinder():
    return _loft_from_profiles(build_notched_phase_shift_cylinder_profiles)


def build_loft_matrix_dual_cylinder_correspondence():
    return _loft_from_profiles(build_dual_cylinder_correspondence_profiles)


def build_loft_matrix_perforated_cylinder_correspondence():
    return _loft_from_profiles(build_perforated_cylinder_correspondence_profiles)


def build_loft_matrix_perforated_vessel():
    return _loft_from_profiles(build_perforated_vessel_profiles)


def build_surfacebody_box():
    return make_box(size=(2.0, 3.0, 1.5), center=(0.0, 0.0, 0.0))


def build_surfacebody_primitive_sphere():
    return make_sphere(radius=0.75)


def build_surfacebody_primitive_cylinder():
    return make_cylinder(radius=0.55, height=1.4, center=(0.0, 0.0, 0.0))


def build_surfacebody_primitive_cone_frustum():
    return make_cone(bottom_diameter=1.2, top_diameter=0.35, height=1.5)


def build_surfacebody_primitive_torus():
    return make_torus(major_radius=0.9, minor_radius=0.22)


def build_surfacebody_primitive_ngon_prism():
    return make_ngon(sides=7, radius=0.65, height=0.9)


def build_surfacebody_primitive_pyramid_frustum():
    return make_prism(base_size=(1.2, 0.9), top_size=(0.28, 0.42), height=1.15)


def build_surfacebody_primitive_transformed_group():
    box = rotate(make_box(size=(0.75, 0.45, 0.35), color="#d9822b"), axis=(0.0, 0.0, 1.0), angle_deg=28.0)
    box = translate(box, (-0.55, -0.08, 0.0))
    cylinder = rotate(make_cylinder(radius=0.18, height=0.95, color="#5b84b1"), axis=(1.0, 0.0, 0.0), angle_deg=72.0)
    cylinder = translate(cylinder, (0.38, 0.22, 0.08))
    frustum = scale(make_cone(bottom_diameter=0.46, top_diameter=0.22, height=0.62, color="#72a276"), (1.0, 0.7, 1.2))
    frustum = translate(frustum, (0.08, -0.42, 0.05))
    return _combine_surface_bodies([box, cylinder, frustum])


def build_surfacebody_primitive_mixed_scale():
    tiny = translate(make_box(size=(0.04, 0.04, 0.04), color="#ffcf5a"), (-0.72, -0.42, 0.0))
    slender = translate(make_cylinder(radius=0.035, height=1.4, color="#8f6bb8"), (0.0, 0.0, 0.0))
    plate = translate(make_box(size=(1.5, 0.08, 0.18), color="#4f9d9a"), (0.15, 0.35, 0.0))
    large = translate(make_sphere(radius=0.28, color="#d95d39"), (0.72, -0.2, 0.0))
    return _combine_surface_bodies([tiny, slender, plate, large])


def build_surfacebody_primitive_thin_stable_dimensions():
    wafer = translate(make_box(size=(1.1, 0.7, 0.025), color="#a8c686"), (0.0, 0.0, -0.05))
    pin = translate(make_cylinder(radius=0.025, height=0.62, color="#f2a65a"), (-0.38, 0.0, 0.0))
    needle = rotate(make_cone(bottom_diameter=0.08, top_diameter=0.01, height=0.85, color="#6c91bf"), axis=(0.0, 1.0, 0.0), angle_deg=86.0)
    needle = translate(needle, (0.32, 0.0, 0.02))
    return _combine_surface_bodies([wafer, pin, needle])


def build_surfacebody_primitive_authored_color_smoke():
    red = translate(make_box(size=(0.42, 0.42, 0.42), color="#d95d39"), (-0.48, 0.0, 0.0))
    blue = translate(make_sphere(radius=0.28, color="#5b84b1"), (0.0, 0.0, 0.0))
    green = translate(make_cylinder(radius=0.22, height=0.48, color="#72a276"), (0.48, 0.0, 0.0))
    return _combine_surface_bodies([red, blue, green])


def build_surfacebody_csg_difference_slot():
    return build_csg_difference_slot_fixture()["result_body"]


def build_surfacebody_csg_union_box_post():
    return build_csg_union_box_post_fixture()["result_body"]


def build_surfacebody_csg_box_union_overlap():
    base = make_box(size=(1.4, 1.0, 0.8))
    post = make_box(size=(0.7, 0.55, 0.9), center=(0.55, 0.1, 0.05))
    return _surface_boolean_body(boolean_union([base, post]))


def build_surfacebody_csg_box_intersection_overlap():
    left = make_box(size=(1.4, 1.0, 0.8))
    right = make_box(size=(0.9, 0.8, 0.9), center=(0.35, 0.12, 0.04))
    return _surface_boolean_body(boolean_intersection([left, right]))


def build_surfacebody_csg_box_contains_sphere_union():
    box = make_box(size=(2.0, 2.0, 2.0))
    sphere = make_sphere(radius=0.45)
    return _surface_boolean_body(boolean_union([box, sphere]))


def build_surfacebody_csg_box_contains_sphere_intersection():
    box = make_box(size=(2.0, 2.0, 2.0))
    sphere = make_sphere(radius=0.45)
    return _surface_boolean_body(boolean_intersection([box, sphere]))


def build_surfacebody_csg_box_difference_corner_notch():
    base = make_box(size=(1.6, 1.2, 0.9))
    cutter = make_box(size=(0.85, 0.7, 1.1), center=(0.5, 0.3, 0.0))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def build_surfacebody_csg_box_difference_end_recess():
    base = make_box(size=(2.0, 1.2, 0.8))
    cutter = make_box(size=(0.85, 0.72, 1.0), center=(0.75, 0.0, 0.0))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def build_surfacebody_csg_box_difference_side_recess():
    base = make_box(size=(2.0, 1.2, 0.8))
    cutter = make_box(size=(0.85, 0.72, 1.0), center=(0.0, 0.45, 0.0))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def build_surfacebody_csg_box_difference_top_pocket():
    base = make_box(size=(1.6, 1.2, 0.9))
    cutter = make_box(size=(0.72, 0.54, 0.5), center=(0.0, 0.0, 0.35))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def build_surfacebody_csg_box_difference_shallow_step():
    base = make_box(size=(1.8, 1.1, 0.8))
    cutter = make_box(size=(0.9, 1.3, 0.46), center=(0.45, 0.0, 0.28))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def build_surfacebody_csg_box_difference_coincident_face():
    base = make_box(size=(1.0, 1.0, 1.0))
    cutter = make_box(size=(1.0, 1.0, 1.0), center=(1.0, 0.0, 0.0))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def build_surfacebody_csg_box_union_disjoint():
    left = make_box(size=(0.8, 0.8, 0.8), center=(-0.6, 0.0, 0.0))
    right = make_box(size=(0.6, 0.6, 0.6), center=(0.55, 0.0, 0.0))
    return _surface_boolean_body(boolean_union([left, right]))


def build_surfacebody_csg_mixed_family_disjoint_union():
    box = make_box(size=(0.9, 0.9, 0.9), center=(-0.85, 0.0, 0.0))
    sphere = make_sphere(radius=0.42, center=(0.85, 0.0, 0.0))
    return _surface_boolean_body(boolean_union([box, sphere]))


def build_surfacebody_csg_rt_csg_001_cube_union_sphere():
    cube = make_box(size=(1.4, 1.4, 1.4))
    sphere = make_sphere(radius=0.55, center=(0.55, 0.0, 0.0))
    return _surface_boolean_body(boolean_union([cube, sphere]))


def build_surfacebody_csg_rt_csg_002_cube_difference_sphere():
    cube = make_box(size=(1.4, 1.4, 1.4))
    sphere = make_sphere(radius=0.62, center=(0.55, 0.0, 0.0))
    return _surface_boolean_body(boolean_difference(cube, [sphere]))


def build_surfacebody_csg_rt_csg_003_cube_intersection_sphere():
    cube = make_box(size=(1.4, 1.4, 1.4))
    sphere = make_sphere(radius=0.72, center=(0.42, 0.0, 0.0))
    return _surface_boolean_body(boolean_intersection([cube, sphere]))


def build_surfacebody_csg_rt_csg_004_cylinder_difference_cube_slot():
    cylinder = make_cylinder(radius=0.55, height=1.4)
    cube = make_box(size=(0.72, 0.72, 0.72), center=(0.35, 0.0, 0.0))
    return _surface_boolean_body(boolean_difference(cylinder, [cube]))


def build_surfacebody_csg_rt_csg_005_cube_difference_cylinder_through_hole():
    cube = make_box(size=(1.4, 1.4, 1.4))
    cylinder = make_cylinder(radius=0.3, height=1.8, direction=(0.0, 1.0, 0.0))
    return _surface_boolean_body(boolean_difference(cube, [cylinder]))


def build_surfacebody_csg_rt_csg_006_orthogonal_cylinders_union():
    cylinder_x = make_cylinder(radius=0.34, height=1.5, direction=(1.0, 0.0, 0.0))
    cylinder_y = make_cylinder(radius=0.34, height=1.5, direction=(0.0, 1.0, 0.0))
    return _surface_boolean_body(boolean_union([cylinder_x, cylinder_y]))


def build_surfacebody_csg_rt_csg_007_orthogonal_cylinders_intersection():
    cylinder_x = make_cylinder(radius=0.45, height=1.5, direction=(1.0, 0.0, 0.0))
    cylinder_y = make_cylinder(radius=0.45, height=1.5, direction=(0.0, 1.0, 0.0))
    return _surface_boolean_body(boolean_intersection([cylinder_x, cylinder_y]))


def build_surfacebody_csg_rt_csg_008_tangent_sphere_cube_union():
    cube = make_box(size=(1.0, 1.0, 1.0))
    sphere = make_sphere(radius=0.35, center=(0.85, 0.0, 0.0))
    return _surface_boolean_body(boolean_union([cube, sphere]))


def build_surfacebody_csg_rt_csg_009_coincident_face_box_union():
    left = make_box(size=(1.0, 1.0, 1.0), center=(-0.5, 0.0, 0.0))
    right = make_box(size=(1.0, 1.0, 1.0), center=(0.5, 0.0, 0.0))
    return _surface_boolean_body(boolean_union([left, right]))


def build_surfacebody_csg_rt_csg_010_nested_cutters_box_sphere_cylinder():
    cube = make_box(size=(1.5, 1.5, 1.5))
    sphere = make_sphere(radius=0.48, center=(0.38, 0.0, 0.0))
    cylinder = make_cylinder(radius=0.22, height=1.9, direction=(0.0, 1.0, 0.0))
    return _surface_boolean_body(boolean_difference(cube, [sphere, cylinder]))


def build_surfacebody_csg_rt_csg_011_multi_operand_union_chain():
    cube = make_box(size=(0.9, 0.9, 0.9), center=(-0.35, 0.0, 0.0))
    sphere = make_sphere(radius=0.42, center=(0.28, 0.0, 0.0))
    cylinder = make_cylinder(radius=0.2, height=1.35, direction=(0.0, 1.0, 0.0), center=(0.0, 0.0, 0.0))
    return _surface_boolean_body(boolean_union([sphere, cylinder, cube]))


def build_surfacebody_csg_rt_csg_012_multi_operand_difference_chain():
    cube = make_box(size=(1.5, 1.2, 1.2))
    first_cutter = make_sphere(radius=0.42, center=(0.45, 0.0, 0.0))
    second_cutter = make_cylinder(radius=0.18, height=1.7, direction=(0.0, 0.0, 1.0), center=(-0.35, 0.0, 0.0))
    return _surface_boolean_body(boolean_difference(cube, [first_cutter, second_cutter]))


def build_surfacebody_csg_rt_patch_csg_001_planar_box_sphere_difference():
    cube = make_box(size=(1.4, 1.4, 1.4))
    sphere = make_sphere(radius=0.62, center=(0.55, 0.0, 0.0))
    return _surface_boolean_body(boolean_difference(cube, [sphere]))


def build_surfacebody_csg_rt_patch_csg_003_revolution_cylinder_box_difference():
    cylinder = make_cylinder(radius=0.55, height=1.4)
    cube = make_box(size=(0.72, 0.72, 0.72), center=(0.35, 0.0, 0.0))
    return _surface_boolean_body(boolean_difference(cylinder, [cube]))


def _loft_provenance_ruled_box_body() -> SurfaceBody:
    box = make_box(size=(1.0, 1.0, 1.0))
    shell = box.iter_shells(world=True)[0]
    front = shell.patches[0]
    ruled_front = RuledSurfacePatch(
        family="ruled",
        start_curve=(front.point_at(0.0, 0.0), front.point_at(0.0, 1.0)),
        end_curve=(front.point_at(1.0, 0.0), front.point_at(1.0, 1.0)),
        metadata={"kernel": {"operation": "loft", "surface_role": "sidewall"}},
    )
    return make_surface_body(
        (
            make_surface_shell(
                (ruled_front, *shell.patches[1:]),
                connected=True,
                seams=shell.seams,
                adjacency=shell.adjacency,
                metadata={"kernel": {"operation": "loft", "executor": "surface", "branch_count": 1}},
            ),
        ),
        metadata={"kernel": {"operation": "loft", "executor": "surface", "branch_count": 1}},
    )


def build_surfacebody_csg_rt_patch_csg_002_loft_ruled_box_difference():
    body = _loft_provenance_ruled_box_body()
    cutter = make_box(size=(0.45, 0.45, 0.45), center=(0.25, 0.0, 0.0))
    return _surface_boolean_body(boolean_difference(body, [cutter]))


def build_surfacebody_csg_bspline_planar_intersection():
    analytic = make_box(size=(1.0, 1.0, 1.0))
    spline = make_box_with_higher_order_front_wall("bspline")
    return _surface_boolean_body(boolean_intersection([analytic, spline]))


def build_surfacebody_csg_nurbs_planar_intersection():
    analytic = make_box(size=(1.0, 1.0, 1.0))
    nurbs = make_box_with_higher_order_front_wall("nurbs")
    return _surface_boolean_body(boolean_intersection([analytic, nurbs]))


def build_surfacebody_csg_bspline_nurbs_intersection():
    spline = make_box_with_higher_order_front_wall("bspline")
    nurbs = make_box_with_higher_order_front_wall("nurbs")
    return _surface_boolean_body(boolean_intersection([spline, nurbs]))


def build_surfacebody_csg_sweep_planar_intersection():
    analytic = make_box(size=(1.0, 1.0, 1.0))
    sweep = make_box_with_sweep_front_wall()
    return _surface_boolean_body(boolean_intersection([analytic, sweep]))


def build_surfacebody_csg_subdivision_planar_intersection():
    analytic = make_box(size=(1.0, 1.0, 1.0))
    subdivision = make_box_with_subdivision_front_wall()
    return _surface_boolean_body(boolean_intersection([analytic, subdivision]))


def build_surfacebody_csg_advanced_implicit_union():
    left = adapt_surface_patch_to_implicit_field(
        PlanarSurfacePatch(family="planar"),
        bounds=(-1.0, 1.0, -1.0, 1.0, -0.1, 0.1),
    )
    right = adapt_surface_patch_to_implicit_field(
        PlanarSurfacePatch(family="planar", origin=np.array([0.0, 0.0, 0.5], dtype=float)),
        bounds=(-1.0, 1.0, -1.0, 1.0, 0.4, 0.6),
    )
    result = compose_implicit_field_csg_result("union", (left, right), samples=(3, 3, 3), max_sample_count=27)
    return _surface_csg_success_body(result, route_name="implicit CSG union")


def build_surfacebody_csg_advanced_heightmap_union():
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
    return _surface_csg_success_body(
        compose_heightmap_csg_result("union", left, right),
        route_name="heightmap CSG union",
    )


def build_surfacebody_csg_advanced_displacement_union():
    source = PlanarSurfacePatch(family="planar", metadata={"fixture_source": "shared"})
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
    return _surface_csg_success_body(
        compose_displacement_csg_result("union", left, right),
        route_name="displacement CSG union",
    )


def build_surfacebody_csg_sampled_implicit_promotion_implicit():
    return make_sampled_implicit_promotion_target_body("implicit")


def build_surfacebody_csg_sampled_implicit_promotion_subdivision():
    return make_sampled_implicit_promotion_target_body("subdivision")


def build_surfacebody_csg_sampled_implicit_promotion_nurbs():
    return make_sampled_implicit_promotion_target_body("nurbs")


def build_surfacebody_csg_sampled_implicit_promotion_bspline():
    return make_sampled_implicit_promotion_target_body("bspline")


def build_surfacebody_drafting_arrow():
    return make_arrow((0.0, 0.0, 0.0), (1.25, 0.25, 0.35), color="#ffb703")


def build_surfacebody_heightmap_surface():
    return heightmap(
        np.asarray(
            [
                [0.0, 0.2, 0.6, 0.9],
                [0.1, 0.5, 0.8, 0.7],
                [0.2, 0.7, 1.0, 0.4],
                [0.0, 0.3, 0.5, 0.2],
            ],
            dtype=float,
        ),
        height=0.6,
        xy_scale=0.3,
        alpha_mode="ignore",
    )


def build_surfacebody_hinge_bistable_blank():
    return handoff_hinge_surface(make_bistable_hinge(width=40.0, preload_offset=2.0))


def build_surfacebody_hinge_living_panel():
    return handoff_hinge_surface(
        make_living_hinge(width=48.0, height=20.0, hinge_band_width=12.0, slit_pitch=1.8)
    )


def build_surfacebody_hinge_traditional_pair():
    return handoff_hinge_surface(
        make_traditional_hinge_pair(width=24.0, knuckle_count=5, opened_angle_deg=32.0)
    )


def build_surfacebody_text_surface():
    return make_text(
        "SURFACE",
        depth=0.08,
        font_size=0.3,
        font_path=str(require_glyph_capable_font()),
        color="#5b84b1",
    )
