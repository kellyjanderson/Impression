from __future__ import annotations

import numpy as np
import pytest

from impression.modeling import BSpline2D, BSpline3D


def test_bspline2d_owns_explicit_curve_fields():
    curve = BSpline2D(
        control_points=[(0.0, 0.0), (0.4, 0.8), (0.8, 0.2), (1.0, 0.0)],
        degree=3,
        knots=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
        closure="closed",
    )

    assert curve.degree == 3
    assert curve.closure == "closed"
    assert curve.knots == (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    assert len(curve.control_points) == 4
    assert np.allclose(curve.control_points[1], [0.4, 0.8])


def test_bspline3d_owns_explicit_curve_fields():
    curve = BSpline3D(
        control_points=[
            (0.0, 0.0, 0.0),
            (0.2, 0.3, 0.5),
            (0.8, -0.1, 1.0),
            (1.0, 0.0, 1.2),
        ],
        degree=3,
        knots=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
        closure="periodic",
    )

    assert curve.degree == 3
    assert curve.closure == "periodic"
    assert len(curve.control_points) == 4
    assert np.allclose(curve.control_points[-1], [1.0, 0.0, 1.2])


def test_bspline_rejects_missing_owned_knot_vector():
    with pytest.raises(ValueError):
        BSpline2D(
            control_points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            degree=2,
            knots=(0.0, 0.0, 0.0, 1.0),
        )


def test_bspline_requires_explicit_closure_policy_value():
    with pytest.raises(ValueError):
        BSpline3D(
            control_points=[
                (0.0, 0.0, 0.0),
                (0.5, 0.5, 0.5),
                (1.0, 0.0, 1.0),
            ],
            degree=2,
            knots=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            closure="auto",
        )


def test_bspline_evaluation_is_stable_for_identical_inputs():
    curve = BSpline2D(
        control_points=[(0.0, 0.0), (0.2, 0.8), (0.8, 0.8), (1.0, 0.0)],
        degree=3,
        knots=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
    )

    point_a = curve.evaluate(0.25)
    point_b = curve.evaluate(0.25)
    assert np.allclose(point_a, point_b)


def test_bspline_tangent_access_is_available():
    curve = BSpline3D(
        control_points=[
            (0.0, 0.0, 0.0),
            (0.3, 0.2, 0.4),
            (0.7, 0.4, 0.8),
            (1.0, 0.0, 1.2),
        ],
        degree=3,
        knots=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
    )

    tangent = curve.tangent(0.5)
    assert tangent.shape == (3,)
    assert np.isclose(np.linalg.norm(tangent), 1.0)


def test_bspline_sampling_is_deterministic():
    curve = BSpline2D(
        control_points=[(0.0, 0.0), (0.3, 0.7), (0.7, 0.7), (1.0, 0.0)],
        degree=3,
        knots=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
    )

    first = curve.sample(12)
    second = curve.sample(12)
    assert np.allclose(first, second)


def test_closed_or_periodic_sampling_closes_the_output():
    curve = BSpline2D(
        control_points=[(0.0, 0.0), (0.2, 0.8), (0.8, 0.2), (1.0, 0.0)],
        degree=3,
        knots=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
        closure="periodic",
    )

    sampled = curve.sample(16)
    assert np.allclose(sampled[0], sampled[-1])
