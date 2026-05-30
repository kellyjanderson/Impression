from __future__ import annotations

import numpy as np
import pytest

from impression.modeling import (
    LoftControlNetConstructionRecord,
    SplineBasisEvaluationRecord,
    SplineBasisDerivativeEvaluationRecord,
    SplineControlNetRecord,
    SplineKnotPolicyRecord,
    build_loft_control_net,
    canonicalize_spline_control_net,
    evaluate_bspline_basis,
    evaluate_bspline_basis_derivative,
    make_clamped_knot_vector,
    validate_clamped_knot_vector,
)


def test_clamped_knot_policy_and_basis_partition_are_deterministic() -> None:
    knots = make_clamped_knot_vector(control_point_count=4, degree=2)

    knot_record = validate_clamped_knot_vector(knots, control_point_count=4, degree=2)
    basis = evaluate_bspline_basis(degree=2, knots=knots, control_point_count=4, parameter=0.5)

    assert isinstance(knot_record, SplineKnotPolicyRecord)
    assert knot_record.clamped is True
    assert knot_record.parameter_range == (0.0, 1.0)
    assert isinstance(basis, SplineBasisEvaluationRecord)
    assert len(basis.basis) == 3
    assert basis.partition_error < 1e-12
    assert basis.canonical_payload()["span"] == basis.span


def test_bspline_basis_derivative_record_is_deterministic_and_conservative() -> None:
    knots = make_clamped_knot_vector(control_point_count=4, degree=2)

    derivative = evaluate_bspline_basis_derivative(
        degree=2,
        knots=knots,
        control_point_count=4,
        parameter=0.5,
    )

    assert isinstance(derivative, SplineBasisDerivativeEvaluationRecord)
    assert len(derivative.derivatives) == 3
    assert abs(derivative.derivative_sum) < 1e-12
    assert derivative.canonical_payload()["derivatives"] == derivative.derivatives


def test_spline_control_net_canonicalizer_records_shape_and_refuses_meshy_payloads() -> None:
    record = canonicalize_spline_control_net(
        [
            [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
        ]
    )

    assert isinstance(record, SplineControlNetRecord)
    assert record.shape == (2, 2, 3)
    assert record.canonical_payload()["point_count_u"] == 2

    with pytest.raises(ValueError, match="control_net"):
        canonicalize_spline_control_net([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])


def test_spline_knot_validation_refuses_unclamped_and_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="clamped"):
        validate_clamped_knot_vector((0.0, 0.0, 0.5, 1.0, 1.0, 1.0), control_point_count=3, degree=2)

    with pytest.raises(ValueError, match="control_point_count"):
        make_clamped_knot_vector(control_point_count=2, degree=2)

    with pytest.raises(ValueError, match="parameter must be finite"):
        evaluate_bspline_basis(
            degree=1,
            knots=(0.0, 0.0, 1.0, 1.0),
            control_point_count=2,
            parameter=float("nan"),
        )


def test_loft_control_net_builder_records_shape_and_clamped_knots() -> None:
    record = build_loft_control_net(
        [
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
            [(0.0, 0.0, 1.0), (1.0, 0.0, 1.2), (1.0, 1.0, 1.0)],
            [(0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (1.0, 1.0, 2.0)],
        ],
        degree_u=2,
        degree_v=2,
    )

    assert isinstance(record, LoftControlNetConstructionRecord)
    assert record.control_net.shape == (3, 3, 3)
    assert record.degree_u == 2
    assert record.degree_v == 2
    assert record.knots_u == (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert record.knots_v == (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    np.testing.assert_allclose(record.control_net[1, 1], (1.0, 0.0, 1.2))
    assert record.canonical_payload()["control_net_shape"] == (3, 3, 3)


def test_loft_control_net_builder_refuses_mismatched_or_nonfinite_profiles() -> None:
    with pytest.raises(ValueError, match="matching point counts"):
        build_loft_control_net(
            [
                [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                [(0.0, 0.0, 1.0)],
            ]
        )

    with pytest.raises(ValueError, match="finite"):
        build_loft_control_net(
            [
                [(0.0, 0.0, 0.0), (float("inf"), 0.0, 0.0)],
                [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0)],
            ]
        )
