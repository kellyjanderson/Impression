from __future__ import annotations

import numpy as np
import pytest

from impression.modeling.drawing2d import (
    Arc2D,
    Bezier2D,
    Line2D,
    Path2D,
    Profile2D,
    make_circle,
    make_ngon,
    make_polygon,
    make_polyline,
    make_rect,
)


def test_line2d_sample_positive():
    line = Line2D(start=(0, 0), end=(1, 1))
    pts = line.sample()
    assert pts.shape == (2, 2)
    assert np.allclose(pts[0], [0, 0])
    assert np.allclose(pts[1], [1, 1])


def test_line2d_invalid_coordinate():
    with pytest.raises(ValueError):
        Line2D(start=(0, 0, 0), end=(1, 1))


def test_arc2d_sample_positive():
    arc = Arc2D(center=(0, 0), radius=1.0, start_angle_deg=0, end_angle_deg=90)
    pts = arc.sample(segments_per_circle=32)
    assert pts.shape[1] == 2
    assert np.allclose(pts[0], [1.0, 0.0], atol=1e-6)
    assert np.allclose(pts[-1], [0.0, 1.0], atol=1e-6)


def test_arc2d_invalid_radius():
    with pytest.raises(ValueError):
        Arc2D(center=(0, 0), radius=0.0, start_angle_deg=0, end_angle_deg=90)


def test_bezier2d_sample_positive():
    bezier = Bezier2D(p0=(0, 0), p1=(0.5, 1.0), p2=(1.0, 1.0), p3=(1.5, 0))
    pts = bezier.sample(samples=8)
    assert np.allclose(pts[0], [0.0, 0.0])
    assert np.allclose(pts[-1], [1.5, 0.0])


def test_bezier2d_invalid_control_point():
    with pytest.raises(ValueError):
        Bezier2D(p0=(0, 0, 0), p1=(0, 0), p2=(0, 0), p3=(0, 0))


def test_path2d_open_closed():
    open_path = Path2D.from_points([(0, 0), (1, 0.5), (2, 0)], closed=False)
    closed_path = Path2D.from_points([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True)
    assert not open_path.closed
    assert closed_path.closed
    open_pts = open_path.sample()
    closed_pts = closed_path.sample()
    assert not np.allclose(open_pts[0], open_pts[-1])
    assert np.allclose(closed_pts[0], closed_pts[-1])


def test_path2d_requires_two_points():
    with pytest.raises(ValueError):
        Path2D.from_points([(0, 0)], closed=False)


def test_profile2d_hole_positive():
    outer = Path2D.from_points([(-1, -1), (1, -1), (1, 1), (-1, 1)], closed=True)
    inner = Path2D.from_points([(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)], closed=True)
    profile = Profile2D(outer=outer, holes=[inner])
    polylines = profile.to_polylines()
    assert len(polylines) == 2


def test_profile2d_requires_closed_outer():
    outer = Path2D.from_points([(-1, -1), (1, -1), (1, 1)], closed=False)
    with pytest.raises(ValueError):
        Profile2D(outer=outer, holes=[])


def test_make_rect_positive():
    profile = make_rect(size=(2.0, 1.0))
    assert profile.outer.closed


def test_make_rect_invalid_size():
    with pytest.raises(ValueError):
        make_rect(size=(0.0, 1.0))


def test_make_circle_negative_radius():
    with pytest.raises(ValueError):
        make_circle(radius=0.0)


def test_make_ngon_negative_sides():
    with pytest.raises(ValueError):
        make_ngon(sides=2, radius=1.0)


def test_make_polygon_positive():
    profile = make_polygon([(0, 0), (1, 0), (0, 1)])
    assert profile.outer.closed


def test_make_polygon_requires_three_points():
    with pytest.raises(ValueError):
        make_polygon([(0, 0), (1, 0)])


def test_make_polyline_requires_two_points():
    with pytest.raises(ValueError):
        make_polyline([(0, 0)])
