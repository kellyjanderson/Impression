from __future__ import annotations

from impression.modeling import (
    FitAssessmentReport,
    FitConfigurationRecord,
    KnotCountPolicyRecord,
    KnotPlacementPolicyRecord,
    ParameterizationPolicyRecord,
    compare_shared_trajectory_curve_fit_candidates,
    compare_station_derived_curve_fit_candidates,
    generate_shared_whole_loft_trajectory_candidates,
    generate_shared_trajectory_curve_fit_candidates,
    generate_station_derived_curve_fit_candidates,
    prepare_dense_loft_fit_descriptors,
)
from impression.modeling.drawing2d import make_circle, make_rect
from impression.modeling import Station, as_section


def _dense_fixture() -> list[Station]:
    return [
        Station(
            t=0.0,
            section=as_section(make_rect(size=(1.0, 1.0))),
            origin=(0.0, 0.0, 0.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
        Station(
            t=0.5,
            section=as_section(make_circle(radius=0.6)),
            origin=(0.3, 0.1, 0.5),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
        Station(
            t=1.0,
            section=as_section(make_rect(size=(0.8, 1.2))),
            origin=(1.0, 0.0, 1.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        ),
    ]


def _fit_config() -> FitConfigurationRecord:
    return FitConfigurationRecord(
        parameterization_policy=ParameterizationPolicyRecord(method="uniform"),
        knot_count_policy=KnotCountPolicyRecord(strategy="fixed", control_point_count=3),
        knot_placement_policy=KnotPlacementPolicyRecord(placement_method="uniform_internal"),
    )


def test_representative_dense_fixtures_produce_station_derived_candidate_fits() -> None:
    candidates = generate_station_derived_curve_fit_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )

    assert len(candidates) == 2
    assert candidates[0].candidate_id == "station_polyline_exact"


def test_comparison_returns_either_an_accepted_candidate_or_an_explicit_refusal() -> None:
    candidates = generate_station_derived_curve_fit_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )

    selected, assessment = compare_station_derived_curve_fit_candidates(candidates)

    assert isinstance(assessment, FitAssessmentReport)
    assert assessment.decision_outcome in {"accepted", "refused"}
    if assessment.decision_outcome == "accepted":
        assert selected is not None


def test_candidate_comparison_references_explicit_residual_diagnostics() -> None:
    candidates = generate_station_derived_curve_fit_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )
    _, assessment = compare_station_derived_curve_fit_candidates(candidates)

    assert assessment.residual_report.metric_name.startswith("max_distance")


def test_refusal_remains_explicit_when_no_station_derived_candidate_is_trustworthy() -> None:
    candidates = generate_station_derived_curve_fit_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )
    weak_candidates = tuple(
        type(candidate)(
            candidate_id=candidate.candidate_id,
            curve=candidate.curve,
            residual_report=type(candidate.residual_report)(
                metric_name=candidate.residual_report.metric_name,
                residual_value=1.0,
                acceptance_threshold=0.1,
                approximation_posture="approximate",
                exact_threshold=0.0,
            ),
            fit_configuration_identity=candidate.fit_configuration_identity,
        )
        for candidate in candidates
    )

    selected, assessment = compare_station_derived_curve_fit_candidates(weak_candidates)

    assert selected is None
    assert assessment.decision_outcome == "refused"


def test_weak_candidates_are_not_silently_selected() -> None:
    candidates = generate_station_derived_curve_fit_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )
    weak_candidates = tuple(
        type(candidate)(
            candidate_id=candidate.candidate_id,
            curve=candidate.curve,
            residual_report=type(candidate.residual_report)(
                metric_name=candidate.residual_report.metric_name,
                residual_value=0.5,
                acceptance_threshold=0.1,
                approximation_posture="approximate",
                exact_threshold=0.0,
            ),
            fit_configuration_identity=candidate.fit_configuration_identity,
        )
        for candidate in candidates
    )

    selected, assessment = compare_station_derived_curve_fit_candidates(weak_candidates)

    assert selected is None
    assert assessment.decision_outcome == "refused"


def test_representative_dense_fixtures_produce_shared_trajectory_candidate_fits() -> None:
    candidates = generate_shared_trajectory_curve_fit_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )

    assert len(candidates) == 2
    assert candidates[0].candidate_id == "shared_trajectory_polyline"


def test_shared_trajectory_comparison_returns_either_an_accepted_candidate_or_an_explicit_refusal() -> None:
    candidates = generate_shared_trajectory_curve_fit_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )

    selected, assessment = compare_shared_trajectory_curve_fit_candidates(candidates)

    assert isinstance(assessment, FitAssessmentReport)
    assert assessment.decision_outcome in {"accepted", "refused"}
    if assessment.decision_outcome == "accepted":
        assert selected is not None


def test_shared_trajectory_comparison_references_explicit_residual_diagnostics() -> None:
    candidates = generate_shared_trajectory_curve_fit_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )
    _, assessment = compare_shared_trajectory_curve_fit_candidates(candidates)

    assert assessment.residual_report.metric_name.startswith("max_distance")


def test_refusal_remains_explicit_when_no_shared_trajectory_candidate_is_trustworthy() -> None:
    candidates = generate_shared_trajectory_curve_fit_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )
    weak_candidates = tuple(
        type(candidate)(
            candidate_id=candidate.candidate_id,
            curve=candidate.curve,
            residual_report=type(candidate.residual_report)(
                metric_name=candidate.residual_report.metric_name,
                residual_value=0.5,
                acceptance_threshold=0.1,
                approximation_posture="approximate",
                exact_threshold=0.0,
            ),
            fit_configuration_identity=candidate.fit_configuration_identity,
        )
        for candidate in candidates
    )

    selected, assessment = compare_shared_trajectory_curve_fit_candidates(weak_candidates)

    assert selected is None
    assert assessment.decision_outcome == "refused"


def test_weak_trajectory_fits_are_not_silently_promoted() -> None:
    candidates = generate_shared_trajectory_curve_fit_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )
    weak_candidates = tuple(
        type(candidate)(
            candidate_id=candidate.candidate_id,
            curve=candidate.curve,
            residual_report=type(candidate.residual_report)(
                metric_name=candidate.residual_report.metric_name,
                residual_value=1.0,
                acceptance_threshold=0.1,
                approximation_posture="approximate",
                exact_threshold=0.0,
            ),
            fit_configuration_identity=candidate.fit_configuration_identity,
        )
        for candidate in candidates
    )

    selected, assessment = compare_shared_trajectory_curve_fit_candidates(weak_candidates)

    assert selected is None
    assert assessment.decision_outcome == "refused"


def test_representative_loft_evidence_produces_shared_whole_loft_trajectory_candidates() -> None:
    candidates = generate_shared_whole_loft_trajectory_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )

    assert len(candidates) == 2
    assert candidates[0].candidate_id.startswith("whole_loft_")


def test_candidate_outputs_remain_inspectable_and_durable() -> None:
    candidates = generate_shared_whole_loft_trajectory_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )

    assert candidates[0].source_fit_candidate_id == "shared_trajectory_polyline"
    assert candidates[0].source_residual_report.metric_name.startswith("max_distance")
    assert candidates[0].fit_configuration_identity == _fit_config().identity


def test_generation_remains_limited_to_whole_loft_shared_trajectories() -> None:
    candidates = generate_shared_whole_loft_trajectory_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )

    assert all(candidate.evidence_lane == "shared_trajectory_curve_fit" for candidate in candidates)


def test_candidate_output_shape_remains_stable_for_identical_inputs() -> None:
    first = generate_shared_whole_loft_trajectory_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )
    second = generate_shared_whole_loft_trajectory_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )

    assert tuple(candidate.candidate_id for candidate in first) == tuple(
        candidate.candidate_id for candidate in second
    )
    assert tuple(candidate.source_fit_candidate_id for candidate in first) == tuple(
        candidate.source_fit_candidate_id for candidate in second
    )
    assert tuple(candidate.fit_configuration_identity for candidate in first) == tuple(
        candidate.fit_configuration_identity for candidate in second
    )


def test_fitted_curve_evidence_contribution_remains_explicit() -> None:
    candidates = generate_shared_whole_loft_trajectory_candidates(
        prepare_dense_loft_fit_descriptors(_dense_fixture()),
        fit_configuration=_fit_config(),
    )

    assert all(candidate.source_fit_candidate_id for candidate in candidates)
    assert all(candidate.source_residual_report.metric_name for candidate in candidates)
