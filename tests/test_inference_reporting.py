from __future__ import annotations

from impression.modeling import (
    DeveloperInferenceInspection,
    DownstreamInferenceReport,
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
        required_topology_station_ids=("topo-0",),
        retained_topology_station_ids=("topo-0",),
        dropped_topology_station_ids=(),
        supporting_diagnostic_references=("diag-0",),
    )


def test_representative_inference_branches_emit_retained_dropped_and_drift_explanations_for_developer_inspection() -> None:
    bundle = SharedInferenceDiagnosticBundle(
        retained_station_entries=(("topo-0", "topology"),),
        dropped_station_entries=(("topo-1", "topology"),),
        fit_drift=InferenceFitDriftSummary.from_fit_assessment(_fit_assessment()),
        structural_preservation=InferenceStructuralPreservationSummary.from_report(_structural_report()),
        evidence_references=("fit-a",),
        provenance_references=("inferred:dense_station_inference",),
    )
    inspection = DeveloperInferenceInspection.from_bundle(bundle)

    assert inspection.retained_station_entries == (("topo-0", "topology"),)
    assert inspection.dropped_station_entries == (("topo-1", "topology"),)
    assert inspection.fit_drift is not None


def test_exact_vs_inferred_provenance_remains_visible_in_developer_facing_inspection() -> None:
    bundle = SharedInferenceDiagnosticBundle(
        retained_station_entries=(("topo-0", "topology"),),
        dropped_station_entries=(),
        evidence_references=("fit-a",),
        provenance_references=("inferred:dense_station_inference",),
    )
    inspection = DeveloperInferenceInspection.from_bundle(bundle)

    assert inspection.provenance_references == ("inferred:dense_station_inference",)


def test_developer_facing_explainability_stays_aligned_with_shared_diagnostic_bundles() -> None:
    bundle = SharedInferenceDiagnosticBundle(
        retained_station_entries=(("topo-0", "topology"), ("control-0", "hidden_control")),
        dropped_station_entries=(),
        evidence_references=("fit-a",),
        provenance_references=("explicit:authored_path",),
    )
    inspection = DeveloperInferenceInspection.from_bundle(bundle)

    assert inspection.retained_station_entries == bundle.retained_station_entries


def test_retained_dropped_and_drift_summaries_remain_inspectable() -> None:
    bundle = SharedInferenceDiagnosticBundle(
        retained_station_entries=(("topo-0", "topology"),),
        dropped_station_entries=(),
        fit_drift=InferenceFitDriftSummary.from_fit_assessment(_fit_assessment()),
        evidence_references=("fit-a",),
        provenance_references=("explicit:authored_path",),
    )
    inspection = DeveloperInferenceInspection.from_bundle(bundle)

    assert inspection.fit_drift is not None
    assert inspection.summary_reason


def test_developer_facing_inspection_does_not_overstate_weak_inference_as_certain_truth() -> None:
    bundle = SharedInferenceDiagnosticBundle(
        retained_station_entries=(("topo-0", "topology"),),
        dropped_station_entries=(),
        evidence_references=("fit-a",),
        provenance_references=("inferred:shared_trajectory_fit",),
    )
    inspection = DeveloperInferenceInspection.from_bundle(bundle)

    assert inspection.certainty_posture == "uncertain"


def test_representative_inference_branches_emit_downstream_refusal_and_uncertainty_summaries() -> None:
    refused = DownstreamInferenceReport.from_bundle(
        SharedInferenceDiagnosticBundle(
            retained_station_entries=(("topo-0", "topology"),),
            dropped_station_entries=(("topo-1", "topology"),),
            evidence_references=("fit-a",),
            provenance_references=("inferred:dense_station_inference",),
        )
    )
    uncertain = DownstreamInferenceReport.from_bundle(
        SharedInferenceDiagnosticBundle(
            retained_station_entries=(("topo-0", "topology"),),
            dropped_station_entries=(),
            evidence_references=("fit-a",),
            provenance_references=("inferred:dense_station_inference",),
        )
    )

    assert refused.refusal_summary is not None
    assert uncertain.uncertainty_summary is not None


def test_downstream_reporting_preserves_exact_vs_inferred_provenance_where_required() -> None:
    report = DownstreamInferenceReport.from_bundle(
        SharedInferenceDiagnosticBundle(
            retained_station_entries=(("topo-0", "topology"),),
            dropped_station_entries=(),
            evidence_references=("fit-a",),
            provenance_references=("inferred:shared_trajectory_fit",),
        )
    )

    assert report.provenance_references == ("inferred:shared_trajectory_fit",)


def test_uncertainty_and_refusal_remain_first_class_downstream_outputs() -> None:
    uncertain = DownstreamInferenceReport.from_bundle(
        SharedInferenceDiagnosticBundle(
            retained_station_entries=(("topo-0", "topology"),),
            dropped_station_entries=(),
            evidence_references=("fit-a",),
            provenance_references=("inferred:shared_trajectory_fit",),
        )
    )
    refused = DownstreamInferenceReport.from_bundle(
        SharedInferenceDiagnosticBundle(
            retained_station_entries=(("topo-0", "topology"),),
            dropped_station_entries=(("topo-1", "topology"),),
            evidence_references=("fit-a",),
            provenance_references=("explicit:authored_path",),
        )
    )

    assert uncertain.outcome == "uncertain"
    assert refused.outcome == "refused"


def test_downstream_reporting_does_not_overstate_weak_inference_as_certain_truth() -> None:
    report = DownstreamInferenceReport.from_bundle(
        SharedInferenceDiagnosticBundle(
            retained_station_entries=(("topo-0", "topology"),),
            dropped_station_entries=(),
            evidence_references=("fit-a",),
            provenance_references=("inferred:dense_station_inference",),
        )
    )

    assert report.outcome == "uncertain"


def test_downstream_reporting_remains_aligned_with_shared_diagnostic_bundle_contract() -> None:
    bundle = SharedInferenceDiagnosticBundle(
        retained_station_entries=(("topo-0", "topology"), ("control-0", "hidden_control")),
        dropped_station_entries=(),
        evidence_references=("fit-a",),
        provenance_references=("explicit:authored_path",),
    )
    report = DownstreamInferenceReport.from_bundle(bundle)

    assert report.retained_station_count == 2
    assert report.dropped_station_count == 0
