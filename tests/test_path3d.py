from __future__ import annotations

import numpy as np
import pytest

from impression.modeling import Arc3D, Line3D, Path3D


def test_path3d_from_points():
    path = Path3D.from_points([(0, 0, 0), (1, 0, 0), (1, 1, 0)], closed=False)
    pts = path.sample()
    assert pts.shape[1] == 3
    assert np.allclose(pts[0], [0, 0, 0])
    assert np.allclose(pts[-1], [1, 1, 0])


def test_path3d_requires_two_points():
    with pytest.raises(ValueError):
        Path3D.from_points([(0, 0, 0)], closed=False)


def test_arc3d_invalid_normal():
    with pytest.raises(ValueError):
        Arc3D(center=(0, 0, 0), radius=1.0, start_angle_deg=0, end_angle_deg=90, normal=(0, 0, 0))


def test_line3d_invalid_coordinate():
    with pytest.raises(ValueError):
        Line3D(start=(0, 0), end=(1, 0, 0))
