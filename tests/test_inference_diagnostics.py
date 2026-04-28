from __future__ import annotations

from impression.modeling import (
    ControlStationInferenceAssessment,
    HiddenControlStationProvenanceRecord,
    HiddenControlStationRecord,
    FitAssessmentReport,
    FitResidualReport,
    InferenceFitDriftSummary,
    InferenceStructuralPreservationSummary,
    Path3D,
    PathBackedProgression,
    ProgressionStationAttachment,
    ReducedProgressionBundle,
    RetainedStationRecord,
    SharedInferenceDiagnosticBundle,
    SharedWholeLoftTrajectoryAssessment,
    StructuralPreservationReport,
    Station,
    as_section,
    assess_control_station_inference,
)
from impression.modeling.drawing2d import make_rect


def _fit_assessment() -> FitAssessmentReport:
    return FitAssessmentReport.from_residual(
        FitResidualReport(
            metric_name="max_distance_to_station_polyline",
            residual_value=0.1,
            acceptance_threshold=0.25,
            approximation_posture="approximate",
            exact_threshold=0.0,
        )
    )


def _structural_report() -> StructuralPreservationReport:
    return StructuralPreservationReport(
        required_topology_station_ids=("topo-0", "topo-1"),
        retained_topology_station_ids=("topo-0",),
        dropped_topology_station_ids=("topo-1",),
        supporting_diagnostic_references=("diag-0",),
    )


def _progression() -> PathBackedProgression:
    return PathBackedProgression(path=Path3D.from_points([(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)]))


def _station(progress: float = 0.5) -> Station:
    return Station(
        t=progress,
        section=as_section(make_rect(size=(1.0, 1.0))),
        origin=(0.0, 0.0, progress),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
    )


def test_shared_diagnostic_bundles_expose_retained_dropped_drift_structural_and_evidence_fields() -> None:
    bundle = SharedInferenceDiagnosticBundle(
        retained_station_entries=(("topo-0", "topology"), ("control-0", "hidden_control")),
        dropped_station_entries=(("topo-1", "topology"),),
        fit_drift=InferenceFitDriftSummary.from_fit_assessment(_fit_assessment()),
        structural_preservation=InferenceStructuralPreservationSummary.from_report(_structural_report()),
        evidence_references=("shared_trajectory_polyline",),
        provenance_references=("inferred:dense_station_inference",),
    )

    assert bundle.retained_station_entries[0] == ("topo-0", "topology")
    assert bundle.dropped_station_entries == (("topo-1", "topology"),)
    assert bundle.fit_drift is not None
    assert bundle.structural_preservation is not None
    assert bundle.evidence_references == ("shared_trajectory_polyline",)


def test_schema_shape_remains_inspectable_and_reusable() -> None:
    bundle = SharedInferenceDiagnosticBundle(
        retained_station_entries=(("topo-0", "topology"),),
        dropped_station_entries=(),
        evidence_references=("fit_candidate",),
        provenance_references=("explicit:authored_path",),
    )

    assert bundle.identity[0] == "shared_inference_diagnostic_bundle"
    assert bundle.replay_payload[0] == "shared_inference_diagnostic_bundle"


def test_bundle_schema_remains_stable_across_inference_branches() -> None:
    station_bundle = SharedInferenceDiagnosticBundle(
        retained_station_entries=(("topo-0", "topology"),),
        dropped_station_entries=(("topo-1", "topology"),),
        fit_drift=InferenceFitDriftSummary.from_fit_assessment(_fit_assessment()),
        evidence_references=("station_polyline_exact",),
        provenance_references=("inferred:dense_station_inference",),
    )
    trajectory_bundle = SharedInferenceDiagnosticBundle(
        retained_station_entries=(("topo-0", "topology"),),
        dropped_station_entries=(("topo-1", "topology"),),
        fit_drift=InferenceFitDriftSummary.from_fit_assessment(_fit_assessment()),
        evidence_references=("whole_loft_shared_trajectory",),
        provenance_references=("inferred:shared_trajectory_fit",),
    )

    assert len(station_bundle.replay_payload) == len(trajectory_bundle.replay_payload)


def test_schema_is_durable_enough_for_replay_and_testing() -> None:
    first = SharedInferenceDiagnosticBundle(
        retained_station_entries=(("topo-0", "topology"),),
        dropped_station_entries=(("topo-1", "topology"),),
        evidence_references=("fit-a",),
        provenance_references=("inferred:dense_station_inference",),
    )
    second = SharedInferenceDiagnosticBundle(
        retained_station_entries=(("topo-0", "topology"),),
        dropped_station_entries=(("topo-1", "topology"),),
        evidence_references=("fit-a",),
        provenance_references=("inferred:dense_station_inference",),
    )

    assert first.replay_payload == second.replay_payload


def test_schema_supports_later_reporting_consumers_without_branch_specific_mutation() -> None:
    bundle = SharedInferenceDiagnosticBundle(
        retained_station_entries=(("topo-0", "topology"),),
        dropped_station_entries=(("topo-1", "topology"),),
        fit_drift=InferenceFitDriftSummary.from_fit_assessment(_fit_assessment()),
        structural_preservation=InferenceStructuralPreservationSummary.from_report(_structural_report()),
        evidence_references=("fit-a",),
        provenance_references=("inferred:dense_station_inference",),
    )

    assert bundle.replay_payload[4] is not None
    assert bundle.replay_payload[5] is not None


def test_representative_inference_branches_populate_the_shared_bundle_through_the_same_contract() -> None:
    station_bundle = SharedInferenceDiagnosticBundle.from_station_fit(
        assessment=_fit_assessment(),
        evidence_reference="station_polyline_exact",
        provenance_reference="inferred:dense_station_inference",
    )
    trajectory_bundle = SharedInferenceDiagnosticBundle.from_shared_trajectory_assessment(
        assessment=SharedWholeLoftTrajectoryAssessment(
            candidate=None,
            confidence=0.0,
            posture="refused",
            reason="missing_shared_whole_loft_trajectory_candidates",
        ),
        provenance_reference="inferred:shared_trajectory_fit",
    )

    assert isinstance(station_bundle, SharedInferenceDiagnosticBundle)
    assert isinstance(trajectory_bundle, SharedInferenceDiagnosticBundle)


def test_reporting_consumers_can_read_populated_bundles_without_branch_specific_assumptions() -> None:
    bundle = SharedInferenceDiagnosticBundle.from_station_fit(
        assessment=_fit_assessment(),
        evidence_reference="station_polyline_exact",
        provenance_reference="inferred:dense_station_inference",
    )

    assert bundle.replay_payload[0] == "shared_inference_diagnostic_bundle"
    assert bundle.evidence_references == ("station_polyline_exact",)


def test_cross_feature_bundle_population_remains_consistent() -> None:
    bundle_from_fit = SharedInferenceDiagnosticBundle.from_station_fit(
        assessment=_fit_assessment(),
        evidence_reference="station_polyline_exact",
        provenance_reference="inferred:dense_station_inference",
    )
    bundle_from_trajectory = SharedInferenceDiagnosticBundle.from_shared_trajectory_assessment(
        assessment=SharedWholeLoftTrajectoryAssessment(
            candidate=None,
            confidence=0.0,
            posture="refused",
            reason="missing_shared_whole_loft_trajectory_candidates",
        ),
        provenance_reference="inferred:shared_trajectory_fit",
    )

    assert len(bundle_from_fit.replay_payload) == len(bundle_from_trajectory.replay_payload)


def test_reuse_posture_does_not_collapse_into_ad_hoc_per_branch_behavior() -> None:
    attachment = ProgressionStationAttachment.from_station(
        progression=_progression(),
        station=_station(0.25),
        station_index=0,
    )
    hidden = HiddenControlStationRecord.from_station(
        station_id="control-0",
        station=_station(0.75),
        provenance=HiddenControlStationProvenanceRecord(),
    )
    retained = (
        RetainedStationRecord.from_attachment(
            station_id="topo-0",
            attachment=attachment,
            diagnostic_references=("diag-topo",),
        ),
        RetainedStationRecord.from_hidden_control(
            record=hidden,
            progression_value=0.75,
            diagnostic_references=("diag-control",),
        ),
    )
    assessment = assess_control_station_inference(
        bundle=ReducedProgressionBundle.from_progression(
            bundle_id="reduction-0",
            progression=_progression(),
            retained_progression_values=(0.0, 0.75, 1.0),
            hidden_control_station_ids=("control-0",),
        ),
        retained_station_records=retained,
        required_topology_station_ids=("topo-0",),
    )

    bundle = SharedInferenceDiagnosticBundle.from_control_station_inference(
        assessment=assessment,
        retained_station_records=retained,
        provenance_reference="inferred:control_station_inference",
    )

    assert bundle.retained_station_entries[0] == ("topo-0", "topology")
    assert bundle.retained_station_entries[1] == ("control-0", "hidden_control")


def test_reporting_remains_aligned_with_the_populated_bundle_contract() -> None:
    assessment = ControlStationInferenceAssessment(
        bundle=None,
        structural_preservation=_structural_report(),
        posture="refused",
        reason="missing_topology_critical_structure",
    )
    bundle = SharedInferenceDiagnosticBundle.from_control_station_inference(
        assessment=assessment,
        retained_station_records=(),
        provenance_reference="inferred:control_station_inference",
    )

    assert bundle.structural_preservation is not None
    assert bundle.dropped_station_entries == (("topo-1", "topology"),)
