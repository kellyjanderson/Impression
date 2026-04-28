from __future__ import annotations

import numpy as np

from impression.modeling import (
    FitAssessmentReport,
    FitConfigurationRecord,
    FitResidualReport,
    KnotCountPolicyRecord,
    KnotPlacementPolicyRecord,
    ParameterizationPolicyRecord,
)


def test_parameterization_policy_record_accepts_explicit_initial_scope_choices():
    policy = ParameterizationPolicyRecord(
        method="centripetal",
        domain_start=2.0,
        domain_end=5.0,
    )

    assert policy.method == "centripetal"
    assert policy.domain_start == 2.0
    assert policy.domain_end == 5.0


def test_identical_evidence_and_policy_reproduce_identical_parameter_assignments():
    policy = ParameterizationPolicyRecord(
        method="chord_length",
        domain_start=0.0,
        domain_end=1.0,
    )
    points = [(0.0, 0.0), (0.3, 0.4), (0.9, 0.9), (1.2, 1.0)]

    first = policy.assign_parameters(points)
    second = policy.assign_parameters(points)

    assert np.allclose(first, second)


def test_parameterization_policy_is_durable_for_replay():
    policy = ParameterizationPolicyRecord(
        method="uniform",
        domain_start=-1.0,
        domain_end=1.0,
    )
    values = policy.assign_parameters([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])

    assert np.allclose(values, [-1.0, 0.0, 1.0])


def test_parameterization_policy_choices_are_visible_to_fit_consumers():
    policy = ParameterizationPolicyRecord(
        method="chord_length",
        domain_start=10.0,
        domain_end=20.0,
    )

    assert policy.method == "chord_length"
    assert policy.domain_start == 10.0
    assert policy.domain_end == 20.0


def test_knot_count_policy_record_accepts_explicit_initial_scope_choices():
    policy = KnotCountPolicyRecord(strategy="fixed", control_point_count=6)

    assert policy.strategy == "fixed"
    assert policy.control_point_count == 6
    assert policy.resolve_control_point_count(sample_count=10, degree=3) == 6


def test_knot_placement_policy_record_is_inspectable_for_later_fit_inputs():
    policy = KnotPlacementPolicyRecord(placement_method="average_parameter")

    assert policy.placement_method == "average_parameter"


def test_knot_count_policy_is_durable_and_replayable():
    policy = KnotCountPolicyRecord(strategy="fixed", control_point_count=5)

    first = policy.resolve_control_point_count(sample_count=9, degree=3)
    second = policy.resolve_control_point_count(sample_count=9, degree=3)

    assert first == second == 5


def test_knot_placement_policy_is_durable_and_replayable():
    policy = KnotPlacementPolicyRecord(placement_method="uniform_internal")
    parameters = (0.0, 0.2, 0.6, 1.0)

    first = policy.build_knot_vector(parameters, control_point_count=4, degree=2)
    second = policy.build_knot_vector(parameters, control_point_count=4, degree=2)

    assert first == second
    assert first == (0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0)


def test_fit_configuration_record_references_parameterization_and_knot_policies():
    parameterization = ParameterizationPolicyRecord(method="centripetal")
    knot_count = KnotCountPolicyRecord(strategy="fixed", control_point_count=6)
    knot_placement = KnotPlacementPolicyRecord(placement_method="average_parameter")

    config = FitConfigurationRecord(
        parameterization_policy=parameterization,
        knot_count_policy=knot_count,
        knot_placement_policy=knot_placement,
    )

    assert config.parameterization_policy is parameterization
    assert config.knot_count_policy is knot_count
    assert config.knot_placement_policy is knot_placement


def test_fit_configuration_identity_is_inspectable_from_later_fit_results():
    config = FitConfigurationRecord(
        parameterization_policy=ParameterizationPolicyRecord(method="uniform"),
        knot_count_policy=KnotCountPolicyRecord(strategy="fixed", control_point_count=5),
        knot_placement_policy=KnotPlacementPolicyRecord(
            placement_method="uniform_internal"
        ),
    )
    later_fit_result = {"fit_configuration_identity": config.identity}

    assert later_fit_result["fit_configuration_identity"] == config.identity


def test_fit_configuration_is_durable_and_replayable():
    config = FitConfigurationRecord(
        parameterization_policy=ParameterizationPolicyRecord(
            method="chord_length",
            domain_start=2.0,
            domain_end=3.0,
        ),
        knot_count_policy=KnotCountPolicyRecord(strategy="fixed", control_point_count=7),
        knot_placement_policy=KnotPlacementPolicyRecord(
            placement_method="average_parameter"
        ),
    )

    first = config.identity
    second = config.identity

    assert first == second


def test_later_inference_branches_can_link_to_the_exact_fit_configuration_used():
    config = FitConfigurationRecord(
        parameterization_policy=ParameterizationPolicyRecord(method="centripetal"),
        knot_count_policy=KnotCountPolicyRecord(strategy="fixed", control_point_count=4),
        knot_placement_policy=KnotPlacementPolicyRecord(
            placement_method="uniform_internal"
        ),
    )
    inference_branch_record = {
        "fit_configuration": config,
        "fit_configuration_identity": config.identity,
    }

    assert inference_branch_record["fit_configuration"] == config
    assert inference_branch_record["fit_configuration_identity"] == config.identity


def test_fit_configuration_comparison_remains_stable_across_identical_inputs():
    kwargs = dict(
        parameterization_policy=ParameterizationPolicyRecord(method="uniform"),
        knot_count_policy=KnotCountPolicyRecord(strategy="fixed", control_point_count=5),
        knot_placement_policy=KnotPlacementPolicyRecord(
            placement_method="average_parameter"
        ),
    )

    first = FitConfigurationRecord(**kwargs)
    second = FitConfigurationRecord(**kwargs)

    assert first == second
    assert first.identity == second.identity


def test_representative_fits_emit_residual_reports_and_a_decision_outcome():
    residual = FitResidualReport(
        metric_name="max_distance",
        residual_value=0.02,
        acceptance_threshold=0.05,
        approximation_posture="approximate",
        exact_threshold=0.0,
    )

    assessment = FitAssessmentReport.from_residual(residual)

    assert assessment.residual_report == residual
    assert assessment.decision_outcome == "accepted"
    assert assessment.decision_reason == "approximate_within_threshold"


def test_refusal_remains_inspectable_as_a_first_class_outcome():
    residual = FitResidualReport(
        metric_name="max_distance",
        residual_value=0.2,
        acceptance_threshold=0.05,
        approximation_posture="approximate",
    )

    assessment = FitAssessmentReport.from_residual(residual)

    assert assessment.decision_outcome == "refused"
    assert assessment.decision_reason == "residual_above_threshold"


def test_fit_drift_is_reported_using_durable_metrics():
    residual = FitResidualReport(
        metric_name="rms_distance",
        residual_value=0.03,
        acceptance_threshold=0.04,
        approximation_posture="approximate",
        exact_threshold=0.01,
    )

    assert residual.metric_name == "rms_distance"
    assert residual.residual_value == 0.03
    assert residual.acceptance_threshold == 0.04
    assert residual.exact_threshold == 0.01
    assert residual.approximation_posture == "approximate"


def test_acceptance_and_refusal_remain_distinguishable_and_replayable():
    accepted = FitAssessmentReport.from_residual(
        FitResidualReport(
            metric_name="max_distance",
            residual_value=0.01,
            acceptance_threshold=0.05,
            approximation_posture="exact",
            exact_threshold=0.01,
        )
    )
    refused = FitAssessmentReport.from_residual(
        FitResidualReport(
            metric_name="max_distance",
            residual_value=0.08,
            acceptance_threshold=0.05,
            approximation_posture="approximate",
        )
    )

    assert accepted.decision_outcome == "accepted"
    assert refused.decision_outcome == "refused"
    assert accepted != refused


def test_weak_fits_are_not_silently_promoted_to_accepted_outputs():
    residual = FitResidualReport(
        metric_name="max_distance",
        residual_value=0.11,
        acceptance_threshold=0.05,
        approximation_posture="approximate",
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "accepted outcomes require a residual within the acceptance threshold.",
    ):
        FitAssessmentReport(
            residual_report=residual,
            decision_outcome="accepted",
            decision_reason="should_fail",
        )
