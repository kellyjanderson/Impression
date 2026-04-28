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
