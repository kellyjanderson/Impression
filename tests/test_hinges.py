from __future__ import annotations

import pytest

from impression.mesh import analyze_mesh
from impression.modeling import (
    make_bistable_hinge,
    make_living_hinge,
    make_traditional_hinge_leaf,
    make_traditional_hinge_pair,
)


def test_traditional_hinge_leaf_generates_mesh() -> None:
    mesh = make_traditional_hinge_leaf(width=24.0, knuckle_count=5)
    analysis = analyze_mesh(mesh)
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    assert analysis.nonmanifold_edges == 0


def test_traditional_hinge_pair_parts() -> None:
    assembly = make_traditional_hinge_pair(width=24.0, include_pin=True, opened_angle_deg=35.0)
    parts = assembly.to_meshes()
    assert len(parts) == 3
    assert all(mesh.n_faces > 0 for mesh in parts)


def test_living_hinge_generates_cut_pattern() -> None:
    living = make_living_hinge(width=48.0, height=20.0, hinge_band_width=12.0, slit_pitch=1.8)
    assert living.n_faces > 12
    xmin, xmax, _, _, _, _ = living.bounds
    assert (xmax - xmin) == pytest.approx(48.0)


def test_bistable_hinge_generates_mesh() -> None:
    mesh = make_bistable_hinge(width=40.0, preload_offset=2.0)
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0


def test_bistable_hinge_invalid_span() -> None:
    with pytest.raises(ValueError):
        make_bistable_hinge(width=24.0, anchor_width=9.0, shuttle_width=7.0)


def test_living_hinge_invalid_slit_geometry() -> None:
    with pytest.raises(ValueError):
        make_living_hinge(height=5.0, edge_margin=2.0, bridge=1.5)
