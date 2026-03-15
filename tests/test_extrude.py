from __future__ import annotations

import pytest

from impression.modeling import MeshQuality, as_section, linear_extrude, rotate_extrude
from impression.modeling.drawing2d import make_polygon, make_rect


def test_linear_extrude_positive():
    section = as_section(make_rect(size=(1.0, 0.6)))
    mesh = linear_extrude(section, height=1.0)
    assert mesh.n_faces > 0
    assert mesh.n_vertices > 0


def test_linear_extrude_invalid_height():
    section = as_section(make_rect(size=(1.0, 0.6)))
    with pytest.raises(ValueError):
        linear_extrude(section, height=0.0)


def test_rotate_extrude_positive():
    section = as_section(make_polygon([(0.4, -0.6), (0.8, -0.2), (0.6, 0.4), (0.2, 0.2)]))
    mesh = rotate_extrude(section, angle_deg=180, segments=24)
    assert mesh.n_faces > 0
    assert mesh.n_vertices > 0


def test_linear_extrude_quality_preview():
    section = as_section(make_rect(size=(1.0, 0.6)))
    mesh = linear_extrude(section, height=1.0, quality=MeshQuality(lod="preview"))
    assert mesh.n_faces > 0


def test_rotate_extrude_invalid_plane():
    section = as_section(make_polygon([(0.4, -0.6), (0.8, -0.2), (0.6, 0.4)]))
    with pytest.raises(ValueError):
        rotate_extrude(section, axis_direction=(0, 0, 1), plane_normal=(0, 0, 1))
