from __future__ import annotations

import numpy as np

from impression.modeling import (
    FitConfigurationRecord,
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
