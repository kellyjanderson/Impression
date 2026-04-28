from __future__ import annotations

import numpy as np

from impression.modeling import (
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
