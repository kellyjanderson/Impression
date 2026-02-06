from __future__ import annotations

import pytest

from impression.modeling import MeshQuality, linear_extrude, rotate_extrude
from impression.modeling.drawing2d import make_polygon, make_rect


def test_linear_extrude_positive():
    profile = make_rect(size=(1.0, 0.6))
    mesh = linear_extrude(profile, height=1.0)
    assert mesh.n_faces > 0
    assert mesh.n_vertices > 0


def test_linear_extrude_invalid_height():
    profile = make_rect(size=(1.0, 0.6))
    with pytest.raises(ValueError):
        linear_extrude(profile, height=0.0)


def test_rotate_extrude_positive():
    profile = make_polygon([(0.4, -0.6), (0.8, -0.2), (0.6, 0.4), (0.2, 0.2)])
    mesh = rotate_extrude(profile, angle_deg=180, segments=24)
    assert mesh.n_faces > 0
    assert mesh.n_vertices > 0


def test_linear_extrude_quality_preview():
    profile = make_rect(size=(1.0, 0.6))
    mesh = linear_extrude(profile, height=1.0, quality=MeshQuality(lod="preview"))
    assert mesh.n_faces > 0


def test_rotate_extrude_invalid_plane():
    profile = make_polygon([(0.4, -0.6), (0.8, -0.2), (0.6, 0.4)])
    with pytest.raises(ValueError):
        rotate_extrude(profile, axis_direction=(0, 0, 1), plane_normal=(0, 0, 1))
