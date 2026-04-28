from __future__ import annotations

from impression.modeling import (
    FitAssessmentReport,
    FitResidualReport,
    InferenceFitDriftSummary,
    InferenceStructuralPreservationSummary,
    SharedInferenceDiagnosticBundle,
    StructuralPreservationReport,
)


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
