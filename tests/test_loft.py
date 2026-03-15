from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from impression.mesh import analyze_mesh
from impression.modeling import (
    MeshQuality,
    Path3D,
    Section,
    Station,
    loft,
    loft_endcaps,
    loft_sections,
    as_section,
)
from impression.modeling.drawing2d import Path2D, PlanarShape2D, make_circle, make_rect
from impression.modeling.loft import _build_stations, _minimum_cost_hole_assignment


def _assert_mesh_quality(mesh) -> None:
    analysis = analyze_mesh(mesh)
    assert analysis.boundary_edges == 0, analysis.issues()
    assert analysis.nonmanifold_edges == 0, analysis.issues()
    assert analysis.degenerate_faces == 0, analysis.issues()


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


def test_loft_quality_preview():
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(0.6, 1.4)),
    ]
    mesh = loft(profiles, samples=40, quality=MeshQuality(lod="preview"))
    assert mesh.n_faces > 0


def test_loft_requires_two_profiles():
    with pytest.raises(ValueError):
        loft([make_rect(size=(1.0, 1.0))])


def test_loft_hole_mismatch():
    a = make_rect(size=(1.0, 1.0))
    b = make_rect(size=(1.0, 1.0))
    b.holes.append(a.outer)
    with pytest.raises(ValueError, match="Unsupported topology transition"):
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


def test_loft_cap_types_add_geometry():
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(1.0, 1.0)),
    ]
    base = loft(profiles, samples=40)
    capped = loft(
        profiles,
        samples=40,
        start_cap="taper",
        end_cap="dome",
        cap_steps=4,
        start_cap_length=0.5,
        end_cap_length=0.75,
        cap_scale_dims="both",
    )
    assert capped.n_vertices > base.n_vertices


def test_loft_invalid_cap_type():
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(1.0, 1.0)),
    ]
    with pytest.raises(ValueError):
        loft(profiles, start_cap="banana")


def test_loft_invalid_scale_dims():
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(1.0, 1.0)),
    ]
    with pytest.raises(ValueError):
        loft(profiles, cap_scale_dims="z")


def test_loft_endcaps_hole_mismatch_reports_topology_transition():
    a = make_rect(size=(1.0, 1.0))
    b = make_rect(size=(1.0, 1.0))
    b.holes.append(a.outer)
    with pytest.raises(ValueError, match="Unsupported topology transition"):
        loft_endcaps([a, b], endcap_mode="FLAT")


def test_loft_invalid_hole_containment_reports_topology_transition():
    a = make_rect(size=(1.0, 1.0))
    b = make_rect(size=(1.0, 1.0))
    b.holes.append(make_rect(size=(0.2, 0.2), center=(3.0, 0.0)).outer)
    with pytest.raises(ValueError, match="Unsupported topology transition"):
        loft([a, b], samples=40)


def test_loft_endcaps_legacy_amount_compatibility():
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(1.0, 1.0)),
    ]
    mesh = loft_endcaps(
        profiles,
        endcap_mode="ROUND",
        endcap_amount=0.25,
        endcap_steps=6,
        endcap_placement="BOTH",
    )
    assert mesh.n_faces > 0


def test_loft_endcaps_independent_depth_radius():
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(1.0, 1.0)),
    ]
    mesh = loft_endcaps(
        profiles,
        endcap_mode="CHAMFER",
        endcap_depth=0.4,
        endcap_radius=0.15,
        endcap_parameter_mode="independent",
        endcap_placement="END",
    )
    assert mesh.n_vertices > 0


def test_loft_endcaps_linked_mode_rejects_mismatch():
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(1.0, 1.0)),
    ]
    with pytest.raises(ValueError):
        loft_endcaps(
            profiles,
            endcap_mode="ROUND",
            endcap_depth=0.4,
            endcap_radius=0.2,
            endcap_parameter_mode="linked",
            endcap_placement="BOTH",
        )


def test_loft_station_frames_are_right_handed():
    stations = _build_stations(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.1, 0.7],
            [1.0, 0.2, 1.5],
        ]
    )
    assert stations[0].t == pytest.approx(0.0)
    assert stations[-1].t == pytest.approx(1.0)
    for station in stations:
        handedness = float(np.dot(np.cross(station.u, station.v), station.n))
        assert np.linalg.norm(station.u) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(station.v) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(station.n) == pytest.approx(1.0, abs=1e-6)
        assert float(np.dot(station.u, station.v)) == pytest.approx(0.0, abs=1e-6)
        assert float(np.dot(station.u, station.n)) == pytest.approx(0.0, abs=1e-6)
        assert float(np.dot(station.v, station.n)) == pytest.approx(0.0, abs=1e-6)
        assert handedness > 0.0


def test_minimum_cost_hole_assignment_matches_by_centroid():
    source_holes = [
        make_rect(size=(0.3, 0.3), center=(-0.5, 0.0)).outer.sample(),
        make_rect(size=(0.3, 0.3), center=(0.5, 0.0)).outer.sample(),
    ]
    target_holes = [
        make_rect(size=(0.3, 0.3), center=(0.5, 0.0)).outer.sample(),
        make_rect(size=(0.3, 0.3), center=(-0.5, 0.0)).outer.sample(),
    ]
    assignment = _minimum_cost_hole_assignment(source_holes, target_holes)
    assert assignment == (1, 0)


def test_minimum_cost_hole_assignment_rejects_ambiguous_non_overlapping_transition():
    source_holes = [
        make_rect(size=(0.3, 0.3), center=(-0.5, 0.0)).outer.sample(),
        make_rect(size=(0.3, 0.3), center=(0.5, 0.0)).outer.sample(),
    ]
    target_holes = [
        make_rect(size=(0.3, 0.3), center=(-0.5, 5.0)).outer.sample(),
        make_rect(size=(0.3, 0.3), center=(0.5, 5.0)).outer.sample(),
    ]
    with pytest.raises(ValueError):
        _minimum_cost_hole_assignment(source_holes, target_holes)


def test_minimum_cost_hole_assignment_tiebreaks_by_target_index():
    source_holes = [
        make_rect(size=(0.3, 0.3), center=(0.0, 0.0)).outer.sample(),
        make_rect(size=(0.3, 0.3), center=(0.0, 0.0)).outer.sample(),
    ]
    target_holes = [
        make_rect(size=(0.3, 0.3), center=(0.0, 0.0)).outer.sample(),
        make_rect(size=(0.3, 0.3), center=(0.0, 0.0)).outer.sample(),
    ]
    assignment = _minimum_cost_hole_assignment(source_holes, target_holes)
    assert assignment == (0, 1)


def test_loft_is_deterministic_for_identical_inputs():
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(0.8, 1.2)),
        make_rect(size=(1.1, 0.9)),
    ]
    path = Path3D.from_points([(0, 0, 0), (0.2, 0.1, 0.8), (0.3, 0.2, 1.6)])
    mesh_a = loft(profiles, path=path, samples=48, cap_ends=True)
    mesh_b = loft(profiles, path=path, samples=48, cap_ends=True)
    assert mesh_a.vertices.shape == mesh_b.vertices.shape
    assert mesh_a.faces.shape == mesh_b.faces.shape
    assert (mesh_a.vertices == mesh_b.vertices).all()
    assert (mesh_a.faces == mesh_b.faces).all()


def test_loft_endcaps_reports_region_collapse_for_oversized_endcap():
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(1.0, 1.0)),
    ]
    with pytest.raises(ValueError, match="region collapsed during endcap generation"):
        loft_endcaps(
            profiles,
            endcap_mode="ROUND",
            endcap_depth=10.0,
            endcap_radius=10.0,
            endcap_steps=8,
            endcap_placement="END",
        )


def test_loft_hole_order_is_deterministically_matched():
    outer = make_rect(size=(2.0, 1.4)).outer
    left_hole = Path2D.from_points(
        [(-0.6, -0.2), (-0.2, -0.2), (-0.2, 0.2), (-0.6, 0.2)],
        closed=True,
    )
    right_hole = Path2D.from_points(
        [(0.2, -0.2), (0.6, -0.2), (0.6, 0.2), (0.2, 0.2)],
        closed=True,
    )
    start = PlanarShape2D(outer=outer, holes=[left_hole, right_hole])
    mid_a = PlanarShape2D(outer=make_rect(size=(1.8, 1.2)).outer, holes=[left_hole, right_hole])
    mid_b = PlanarShape2D(outer=make_rect(size=(1.8, 1.2)).outer, holes=[right_hole, left_hole])
    end = PlanarShape2D(outer=make_rect(size=(1.6, 1.0)).outer, holes=[left_hole, right_hole])

    mesh_a = loft([start, mid_a, end], samples=40)
    mesh_b = loft([start, mid_b, end], samples=40)

    assert mesh_a.vertices.shape == mesh_b.vertices.shape
    assert mesh_a.faces.shape == mesh_b.faces.shape
    assert (mesh_a.vertices == mesh_b.vertices).all()
    assert (mesh_a.faces == mesh_b.faces).all()


def test_loft_sections_basic_with_station_frames():
    a = make_rect(size=(1.0, 1.0))
    b = make_rect(size=(0.8, 1.2))
    stations = [
        Station(
            t=0.0,
            section=as_section(a),
            origin=[0.0, 0.0, 0.0],
            u=[1.0, 0.0, 0.0],
            v=[0.0, 1.0, 0.0],
            n=[0.0, 0.0, 1.0],
        ),
        Station(
            t=1.0,
            section=as_section(b),
            origin=[0.1, 0.0, 1.0],
            u=[1.0, 0.0, 0.0],
            v=[0.0, 1.0, 0.0],
            n=[0.0, 0.0, 1.0],
        ),
    ]
    mesh = loft_sections(stations, samples=40, cap_ends=True)
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_loft_sections_requires_strictly_ordered_t():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=0.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    with pytest.raises(ValueError, match="strictly ordered by t"):
        loft_sections(stations, samples=24)


def test_loft_sections_rejects_non_orthonormal_frame():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[1, 0, 0], n=[0, 0, 1]),
    ]
    with pytest.raises(ValueError, match="orthogonal"):
        loft_sections(stations, samples=24)


def test_loft_sections_rejects_invalid_split_merge_controls():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    with pytest.raises(ValueError, match="split_merge_mode"):
        loft_sections(stations, samples=24, split_merge_mode="banana")
    with pytest.raises(ValueError, match="split_merge_steps"):
        loft_sections(stations, samples=24, split_merge_mode="resolve", split_merge_steps=0)
    with pytest.raises(ValueError, match="split_merge_bias"):
        loft_sections(stations, samples=24, split_merge_mode="resolve", split_merge_bias=2.0)


def test_loft_sections_multi_region_order_is_deterministically_matched():
    left0 = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    right0 = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    left1 = make_rect(size=(0.7, 0.9), center=(-1.1, 0.0))
    right1 = make_rect(size=(0.9, 0.7), center=(1.1, 0.0))

    s0 = Section((as_section(left0).regions[0], as_section(right0).regions[0]))
    s1_ordered = Section((as_section(left1).regions[0], as_section(right1).regions[0]))
    s1_swapped = Section((as_section(right1).regions[0], as_section(left1).regions[0]))

    stations_a = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1_ordered, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    stations_b = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1_swapped, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]

    mesh_a = loft_sections(stations_a, samples=32, cap_ends=True)
    mesh_b = loft_sections(stations_b, samples=32, cap_ends=True)
    assert mesh_a.vertices.shape == mesh_b.vertices.shape
    assert mesh_a.faces.shape == mesh_b.faces.shape
    assert (mesh_a.vertices == mesh_b.vertices).all()
    assert (mesh_a.faces == mesh_b.faces).all()
    _assert_mesh_quality(mesh_a)
    _assert_mesh_quality(mesh_b)


def test_loft_sections_multi_hole_order_is_deterministically_matched():
    outer = make_rect(size=(2.0, 1.4)).outer
    left_hole = make_circle(radius=0.16, center=(-0.45, 0.0)).outer
    right_hole = make_circle(radius=0.16, center=(0.45, 0.0)).outer
    start = PlanarShape2D(outer=outer, holes=[left_hole, right_hole])
    end_a = PlanarShape2D(outer=outer, holes=[left_hole, right_hole])
    end_b = PlanarShape2D(outer=outer, holes=[right_hole, left_hole])
    stations_a = [
        Station(t=0.0, section=as_section(start), origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=as_section(end_a), origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    stations_b = [
        Station(t=0.0, section=as_section(start), origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=as_section(end_b), origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    mesh_a = loft_sections(stations_a, samples=32, cap_ends=True)
    mesh_b = loft_sections(stations_b, samples=32, cap_ends=True)
    assert mesh_a.vertices.shape == mesh_b.vertices.shape
    assert mesh_a.faces.shape == mesh_b.faces.shape
    assert (mesh_a.vertices == mesh_b.vertices).all()
    assert (mesh_a.faces == mesh_b.faces).all()
    _assert_mesh_quality(mesh_a)
    _assert_mesh_quality(mesh_b)


def test_loft_sections_supports_region_birth_transition():
    base = as_section(make_rect(size=(1.0, 1.0)))
    extra = as_section(make_rect(size=(0.5, 0.5), center=(2.0, 0.0)))
    s0 = Section((base.regions[0],))
    s1 = Section((base.regions[0], extra.regions[0]))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    mesh = loft_sections(stations, samples=24, cap_ends=True)
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_loft_sections_supports_region_death_transition():
    base = as_section(make_rect(size=(1.0, 1.0)))
    extra = as_section(make_rect(size=(0.5, 0.5), center=(2.0, 0.0)))
    s0 = Section((base.regions[0], extra.regions[0]))
    s1 = Section((base.regions[0],))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    mesh = loft_sections(stations, samples=24, cap_ends=True)
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_loft_sections_supports_hole_birth_transition():
    outer = make_rect(size=(1.2, 1.2)).outer
    hole = make_rect(size=(0.4, 0.4)).outer
    start = PlanarShape2D(outer=outer, holes=[])
    end = PlanarShape2D(outer=outer, holes=[hole])
    stations = [
        Station(
            t=0.0,
            section=as_section(start),
            origin=[0, 0, 0],
            u=[1, 0, 0],
            v=[0, 1, 0],
            n=[0, 0, 1],
        ),
        Station(
            t=1.0,
            section=as_section(end),
            origin=[0, 0, 1],
            u=[1, 0, 0],
            v=[0, 1, 0],
            n=[0, 0, 1],
        ),
    ]
    mesh = loft_sections(stations, samples=28, cap_ends=True)
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_loft_sections_supports_hole_death_transition():
    outer = make_rect(size=(1.2, 1.2)).outer
    hole = make_rect(size=(0.4, 0.4)).outer
    start = PlanarShape2D(outer=outer, holes=[hole])
    end = PlanarShape2D(outer=outer, holes=[])
    stations = [
        Station(
            t=0.0,
            section=as_section(start),
            origin=[0, 0, 0],
            u=[1, 0, 0],
            v=[0, 1, 0],
            n=[0, 0, 1],
        ),
        Station(
            t=1.0,
            section=as_section(end),
            origin=[0, 0, 1],
            u=[1, 0, 0],
            v=[0, 1, 0],
            n=[0, 0, 1],
        ),
    ]
    mesh = loft_sections(stations, samples=28, cap_ends=True)
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_loft_sections_rejects_ambiguous_region_merge_transition():
    left = make_rect(size=(0.9, 0.9), center=(-0.2, 0.0))
    right = make_rect(size=(0.9, 0.9), center=(0.2, 0.0))
    merged = make_rect(size=(1.2, 1.0), center=(0.0, 0.0))
    s0 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1 = Section((as_section(merged).regions[0],))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    with pytest.raises(ValueError, match="region split/merge ambiguity"):
        loft_sections(stations, samples=32, cap_ends=True)


def test_loft_sections_rejects_ambiguous_hole_merge_transition():
    outer = make_rect(size=(2.0, 1.4)).outer
    hole_a = make_rect(size=(0.35, 0.35), center=(-0.12, 0.0)).outer
    hole_b = make_rect(size=(0.35, 0.35), center=(0.12, 0.0)).outer
    merged_hole = make_rect(size=(0.8, 0.4), center=(0.0, 0.0)).outer
    start = PlanarShape2D(outer=outer, holes=[hole_a, hole_b])
    end = PlanarShape2D(outer=outer, holes=[merged_hole])
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=as_section(end), origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    with pytest.raises(ValueError, match="hole split/merge ambiguity"):
        loft_sections(stations, samples=32, cap_ends=True)


def test_loft_sections_resolves_region_merge_transition_when_enabled():
    left = make_rect(size=(0.9, 0.9), center=(-0.2, 0.0))
    right = make_rect(size=(0.9, 0.9), center=(0.2, 0.0))
    merged = make_rect(size=(1.2, 1.0), center=(0.0, 0.0))
    s0 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1 = Section((as_section(merged).regions[0],))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    mesh = loft_sections(
        stations,
        samples=32,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=10,
        split_merge_bias=0.5,
    )
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_loft_sections_resolve_mode_stages_transitions_with_more_steps():
    left = make_rect(size=(0.9, 0.9), center=(-0.2, 0.0))
    right = make_rect(size=(0.9, 0.9), center=(0.2, 0.0))
    merged = make_rect(size=(1.2, 1.0), center=(0.0, 0.0))
    s0 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1 = Section((as_section(merged).regions[0],))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    mesh_low = loft_sections(
        stations,
        samples=32,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=2,
        split_merge_bias=0.5,
    )
    mesh_high = loft_sections(
        stations,
        samples=32,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=10,
        split_merge_bias=0.5,
    )
    assert mesh_high.n_vertices > mesh_low.n_vertices
    assert mesh_high.n_faces > mesh_low.n_faces
    _assert_mesh_quality(mesh_low)
    _assert_mesh_quality(mesh_high)


def test_loft_sections_resolves_region_split_transition_when_enabled():
    merged = make_rect(size=(1.2, 1.0), center=(0.0, 0.0))
    left = make_rect(size=(0.9, 0.9), center=(-0.2, 0.0))
    right = make_rect(size=(0.9, 0.9), center=(0.2, 0.0))
    s0 = Section((as_section(merged).regions[0],))
    s1 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    mesh = loft_sections(
        stations,
        samples=32,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=10,
        split_merge_bias=0.5,
    )
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_loft_sections_resolves_hole_merge_transition_when_enabled():
    outer = make_rect(size=(2.0, 1.4)).outer
    hole_a = make_circle(radius=0.16, center=(-0.30, 0.0)).outer
    hole_b = make_circle(radius=0.16, center=(0.30, 0.0)).outer
    merged_hole = make_circle(radius=0.42, center=(0.0, 0.0)).outer
    start = PlanarShape2D(outer=outer, holes=[hole_a, hole_b])
    end = PlanarShape2D(outer=outer, holes=[merged_hole])
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=as_section(end), origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    mesh = loft_sections(
        stations,
        samples=32,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=10,
        split_merge_bias=0.5,
    )
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_loft_sections_resolves_hole_split_transition_when_enabled():
    outer = make_rect(size=(2.0, 1.4)).outer
    merged_hole = make_circle(radius=0.42, center=(0.0, 0.0)).outer
    hole_a = make_circle(radius=0.16, center=(-0.30, 0.0)).outer
    hole_b = make_circle(radius=0.16, center=(0.30, 0.0)).outer
    start = PlanarShape2D(outer=outer, holes=[merged_hole])
    end = PlanarShape2D(outer=outer, holes=[hole_a, hole_b])
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=as_section(end), origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    mesh = loft_sections(
        stations,
        samples=32,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=10,
        split_merge_bias=0.5,
    )
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_loft_sections_resolve_mode_rejects_many_to_many_region_ambiguity():
    left = make_rect(size=(0.8, 0.8), center=(-0.8, 0.0))
    center = make_rect(size=(0.8, 0.8), center=(0.0, 0.0))
    right = make_rect(size=(0.8, 0.8), center=(0.8, 0.0))
    wide_left = make_rect(size=(1.3, 0.9), center=(-0.35, 0.0))
    wide_right = make_rect(size=(1.3, 0.9), center=(0.35, 0.0))
    s0 = Section((as_section(left).regions[0], as_section(center).regions[0], as_section(right).regions[0]))
    s1 = Section((as_section(wide_left).regions[0], as_section(wide_right).regions[0]))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    with pytest.raises(ValueError, match="region split/merge ambiguity"):
        loft_sections(
            stations,
            samples=28,
            cap_ends=True,
            split_merge_mode="resolve",
            split_merge_steps=8,
            split_merge_bias=0.5,
        )


def test_loft_sections_resolve_mode_rejects_many_to_many_hole_ambiguity():
    outer = make_rect(size=(2.0, 1.5)).outer
    h0 = make_rect(size=(0.25, 0.25), center=(-0.6, 0.0)).outer
    h1 = make_rect(size=(0.25, 0.25), center=(0.0, 0.0)).outer
    h2 = make_rect(size=(0.25, 0.25), center=(0.6, 0.0)).outer
    k0 = make_rect(size=(0.45, 0.3), center=(-0.3, 0.0)).outer
    k1 = make_rect(size=(0.45, 0.3), center=(0.3, 0.0)).outer
    start = PlanarShape2D(outer=outer, holes=[h0, h1, h2])
    end = PlanarShape2D(outer=outer, holes=[k0, k1])
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=as_section(end), origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    with pytest.raises(ValueError, match="hole split/merge ambiguity"):
        loft_sections(
            stations,
            samples=28,
            cap_ends=True,
            split_merge_mode="resolve",
            split_merge_steps=8,
            split_merge_bias=0.5,
        )


def test_loft_sections_resolve_mode_is_deterministic_for_identical_inputs():
    left = make_rect(size=(0.9, 0.9), center=(-0.2, 0.0))
    right = make_rect(size=(0.9, 0.9), center=(0.2, 0.0))
    merged = make_rect(size=(1.2, 1.0), center=(0.0, 0.0))
    s0 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1 = Section((as_section(merged).regions[0],))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    mesh_a = loft_sections(
        stations,
        samples=32,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=10,
        split_merge_bias=0.5,
    )
    mesh_b = loft_sections(
        stations,
        samples=32,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=10,
        split_merge_bias=0.5,
    )
    assert mesh_a.vertices.shape == mesh_b.vertices.shape
    assert mesh_a.faces.shape == mesh_b.faces.shape
    assert (mesh_a.vertices == mesh_b.vertices).all()
    assert (mesh_a.faces == mesh_b.faces).all()
    _assert_mesh_quality(mesh_a)


def test_loft_split_merge_resolve_example_meshes_are_watertight():
    module_path = Path("docs/examples/loft/loft_split_merge_resolve_example.py")
    spec = importlib.util.spec_from_file_location("loft_split_merge_resolve_example", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    scene = module.build()
    meshes = scene if isinstance(scene, list) else [scene]
    assert meshes, "Expected at least one mesh from resolve demo build()."
    for mesh in meshes:
        _assert_mesh_quality(mesh)
