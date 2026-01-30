from __future__ import annotations

import pytest

from impression.modeling import Path3D, loft
from impression.modeling.drawing2d import make_rect


def test_loft_positive():
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(0.6, 1.4)),
        make_rect(size=(0.8, 0.8)),
    ]
    path = Path3D.from_points([(0, 0, 0), (0, 0, 1), (0, 0, 2)])
    mesh = loft(profiles, path=path, cap_ends=True, samples=40)
    assert mesh.n_faces > 0
    assert mesh.n_vertices > 0


def test_loft_requires_two_profiles():
    with pytest.raises(ValueError):
        loft([make_rect(size=(1.0, 1.0))])


def test_loft_hole_mismatch():
    a = make_rect(size=(1.0, 1.0))
    b = make_rect(size=(1.0, 1.0))
    b.holes.append(a.outer)
    with pytest.raises(ValueError):
        loft([a, b])


def test_loft_rotates_profiles_along_path():
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(1.0, 1.0)),
    ]
    path = Path3D.from_points([(0, 0, 0), (2, 0, 0)])
    mesh = loft(profiles, path=path, samples=40)
    _, _, _, _, zmin, zmax = mesh.bounds
    assert (zmax - zmin) > 0.1
