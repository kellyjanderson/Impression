from __future__ import annotations

import pytest

from impression.modeling import extrude_sdf, loft_sdf
from impression.modeling.drawing2d import make_rect

try:
    import skimage  # noqa: F401
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


@pytest.mark.skipif(not HAS_SKIMAGE, reason="scikit-image not installed")
def test_extrude_sdf_basic():
    profile = make_rect(size=(1.0, 0.6))
    mesh = extrude_sdf(profile, height=1.0, cap_radius=0.2, grid_spacing=0.3)
    assert mesh.n_faces > 0
    assert mesh.n_vertices > 0


@pytest.mark.skipif(not HAS_SKIMAGE, reason="scikit-image not installed")
def test_loft_sdf_basic():
    profiles = [make_rect(size=(1.0, 0.6)), make_rect(size=(0.7, 0.9))]
    mesh = loft_sdf(profiles, positions=[0.0, 1.0], cap_radius=0.2, grid_spacing=0.4)
    assert mesh.n_faces > 0
    assert mesh.n_vertices > 0
