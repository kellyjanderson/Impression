from __future__ import annotations

from impression.modeling import (
    DeveloperInferenceInspection,
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
