from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from impression.mesh import analyze_mesh
from impression.modeling import (
    MeshQuality,
    Path3D,
    PlannedLoopRef,
    PlannedRegionRef,
    Section,
    Station,
    loft,
    loft_execute_plan,
    loft_endcaps,
    loft_plan_sections,
    loft_sections,
    as_section,
)
from impression.modeling.drawing2d import Path2D, PlanarShape2D, make_circle, make_rect
from impression.modeling.loft import (
    _build_stations,
    _minimum_cost_hole_assignment,
    _local_assignment_fairness,
)


def _assert_mesh_quality(mesh) -> None:
    analysis = analyze_mesh(mesh)
    assert analysis.boundary_edges == 0, analysis.issues()
    assert analysis.nonmanifold_edges == 0, analysis.issues()
    assert analysis.degenerate_faces == 0, analysis.issues()


def _plan_transition_signature(plan) -> tuple:
    signatures: list[tuple] = []
    for transition in plan.transitions:
        region_sigs: list[tuple] = []
        for pair in transition.region_pairs:
            loop_sigs = tuple(
                (
                    loop_pair.prev_loop_ref.kind,
                    loop_pair.prev_loop_ref.index,
                    loop_pair.curr_loop_ref.kind,
                    loop_pair.curr_loop_ref.index,
                    loop_pair.role,
                )
                for loop_pair in pair.loop_pairs
            )
            closure_sigs = tuple(
                (closure.side, closure.scope, closure.loop_index) for closure in pair.closures
            )
            region_sigs.append(
                (
                    pair.branch_id,
                    pair.prev_region_ref.kind,
                    pair.prev_region_ref.index,
                    pair.curr_region_ref.kind,
                    pair.curr_region_ref.index,
                    pair.action,
                    loop_sigs,
                    closure_sigs,
                )
            )
        signatures.append(
            (
                transition.interval,
                transition.topology_case,
                transition.ambiguity_class,
                transition.branch_order,
                tuple(region_sigs),
            )
        )
    return tuple(signatures)


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


def test_loft_supports_hole_birth():
    outer = make_rect(size=(1.2, 1.2)).outer
    hole = make_rect(size=(0.4, 0.4)).outer
    start = PlanarShape2D(outer=outer, holes=[])
    end = PlanarShape2D(outer=outer, holes=[hole])
    mesh = loft([start, end], samples=28, cap_ends=True)
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


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


def test_loft_supports_region_split():
    start = Section((as_section(make_rect(size=(1.4, 1.0))).regions[0],))
    left = as_section(make_rect(size=(0.5, 0.8), center=(-0.55, 0.0))).regions[0]
    right = as_section(make_rect(size=(0.5, 0.8), center=(0.55, 0.0))).regions[0]
    end = Section((left, right))
    mesh = loft([start, end], samples=28, cap_ends=True, split_merge_mode="resolve")
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


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


def test_local_assignment_fairness_prefers_balanced_branch_lengths():
    src = [np.array([-3.0, -3.0]), np.array([-3.0, -2.0])]
    dst = [np.array([-3.0, -2.0]), np.array([-3.0, -1.0])]
    balanced = _local_assignment_fairness((0, 1), src, dst)
    imbalanced = _local_assignment_fairness((1, 0), src, dst)
    assert balanced < imbalanced


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


def test_loft_sections_matches_planner_executor_pipeline():
    outer = make_rect(size=(1.4, 1.2)).outer
    hole = make_circle(radius=0.2, center=(0.1, 0.0)).outer
    start = PlanarShape2D(outer=outer, holes=[hole])
    end = PlanarShape2D(outer=make_rect(size=(1.1, 1.4)).outer, holes=[hole])
    stations = [
        Station(
            t=0.0,
            section=as_section(start),
            origin=[0.0, 0.0, 0.0],
            u=[1.0, 0.0, 0.0],
            v=[0.0, 1.0, 0.0],
            n=[0.0, 0.0, 1.0],
        ),
        Station(
            t=1.0,
            section=as_section(end),
            origin=[0.0, 0.0, 1.0],
            u=[1.0, 0.0, 0.0],
            v=[0.0, 1.0, 0.0],
            n=[0.0, 0.0, 1.0],
        ),
    ]
    direct = loft_sections(stations, samples=36, cap_ends=True)
    plan = loft_plan_sections(stations, samples=36)
    from_plan = loft_execute_plan(plan, cap_ends=True)
    assert np.array_equal(direct.vertices, from_plan.vertices)
    assert np.array_equal(direct.faces, from_plan.faces)
    _assert_mesh_quality(from_plan)


def test_loft_sections_resolve_mode_matches_planner_executor_pipeline():
    left = make_rect(size=(0.9, 0.9), center=(-0.2, 0.0))
    right = make_rect(size=(0.9, 0.9), center=(0.2, 0.0))
    merged = make_rect(size=(1.2, 1.0), center=(0.0, 0.0))
    s0 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1 = Section((as_section(merged).regions[0],))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    direct = loft_sections(
        stations,
        samples=32,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=10,
        split_merge_bias=0.5,
    )
    plan = loft_plan_sections(
        stations,
        samples=32,
        split_merge_mode="resolve",
        split_merge_steps=10,
        split_merge_bias=0.5,
    )
    from_plan = loft_execute_plan(plan, cap_ends=True)
    assert np.array_equal(direct.vertices, from_plan.vertices)
    assert np.array_equal(direct.faces, from_plan.faces)
    _assert_mesh_quality(from_plan)


def test_loft_plan_sections_marks_synthetic_loop_birth_and_closure():
    outer = make_rect(size=(1.2, 1.2)).outer
    hole = make_rect(size=(0.4, 0.4)).outer
    start = PlanarShape2D(outer=outer, holes=[])
    end = PlanarShape2D(outer=outer, holes=[hole])
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=as_section(end), origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=28)
    pair = plan.transitions[0].region_pairs[0]
    assert isinstance(pair.prev_region_ref, PlannedRegionRef)
    assert isinstance(pair.curr_region_ref, PlannedRegionRef)
    assert all(isinstance(loop_pair.prev_loop_ref, PlannedLoopRef) for loop_pair in pair.loop_pairs)
    assert all(isinstance(loop_pair.curr_loop_ref, PlannedLoopRef) for loop_pair in pair.loop_pairs)
    assert any(loop_pair.role == "synthetic_birth" for loop_pair in pair.loop_pairs)
    assert any(closure.scope == "loop" and closure.side == "prev" for closure in pair.closures)


def test_loft_plan_sections_marks_synthetic_region_death_and_closure():
    base = as_section(make_rect(size=(1.0, 1.0)))
    extra = as_section(make_rect(size=(0.5, 0.5), center=(2.0, 0.0)))
    s0 = Section((base.regions[0], extra.regions[0]))
    s1 = Section((base.regions[0],))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=24)
    transition = plan.transitions[0]
    assert any(pair.curr_region_ref.kind == "synthetic" for pair in transition.region_pairs)
    assert any(
        closure.scope == "region" and closure.side == "curr"
        for pair in transition.region_pairs
        for closure in pair.closures
    )


def test_loft_plan_sections_sets_plan_metadata_contract():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=24)
    assert plan.metadata["plan_schema_version"] == 1
    assert plan.metadata["planner"] == "loft_plan_sections"
    assert plan.metadata["ambiguity_mode"] == "fail"
    assert plan.metadata["ambiguity_cost_profile"] == "balanced"
    assert plan.metadata["ambiguity_max_branches"] == 64
    assert plan.metadata["ambiguity_resolved_intervals_count"] == 0
    assert plan.metadata["ambiguity_failed_intervals_count"] == 0
    assert plan.metadata["ambiguity_class_counts"] == {
        "permutation": 0,
        "containment": 0,
        "symmetry": 0,
        "closure": 0,
    }
    assert plan.metadata["fairness_mode"] == "local"
    assert plan.metadata["fairness_weight"] == 0.2
    assert plan.metadata["fairness_iterations"] == 12
    assert plan.metadata["skeleton_mode"] == "auto"
    assert plan.metadata["fairness_optimization_convergence_status"] == "not_run"
    assert plan.metadata["fairness_objective_pre"] == {
        "curvature_continuity": 0.0,
        "branch_crossing": 0.0,
        "branch_acceleration": 0.0,
        "synthetic_harshness": 0.0,
        "closure_stress": 0.0,
    }
    assert plan.metadata["fairness_objective_post"] == {
        "curvature_continuity": 0.0,
        "branch_crossing": 0.0,
        "branch_acceleration": 0.0,
        "synthetic_harshness": 0.0,
        "closure_stress": 0.0,
    }
    assert plan.metadata["fairness_diagnostics"] == {
        "branch_crossing_count": 0.0,
        "continuity_score": 0.0,
        "closure_distortion_score": 0.0,
    }


def test_loft_plan_sections_sets_ambiguity_mode_default_auto_in_resolve_mode():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=24, split_merge_mode="resolve")
    assert plan.metadata["ambiguity_mode"] == "auto"


def test_loft_plan_sections_records_explicit_ambiguity_controls_in_metadata():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(
        stations,
        samples=24,
        split_merge_mode="resolve",
        ambiguity_mode="fail",
        ambiguity_cost_profile="distance_first",
        ambiguity_max_branches=17,
    )
    assert plan.metadata["ambiguity_mode"] == "fail"
    assert plan.metadata["ambiguity_cost_profile"] == "distance_first"
    assert plan.metadata["ambiguity_max_branches"] == 17
    assert "ambiguity_class_counts" in plan.metadata


def test_loft_plan_sections_records_explicit_fairness_controls_in_metadata():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(
        stations,
        samples=24,
        split_merge_mode="resolve",
        fairness_mode="global",
        fairness_weight=0.45,
        skeleton_mode="auto",
        fairness_iterations=20,
    )
    assert plan.metadata["fairness_mode"] == "global"
    assert plan.metadata["fairness_weight"] == 0.45
    assert plan.metadata["skeleton_mode"] == "auto"
    assert plan.metadata["fairness_iterations"] == 20
    assert plan.metadata["fairness_optimization_convergence_status"] == "not_run"
    assert plan.metadata["fairness_objective_pre"] == plan.metadata["fairness_objective_post"]


def test_loft_execute_plan_rejects_invalid_loop_closure_index_with_interval_context():
    outer = make_rect(size=(1.2, 1.2)).outer
    hole = make_rect(size=(0.4, 0.4)).outer
    start = PlanarShape2D(outer=outer, holes=[])
    end = PlanarShape2D(outer=outer, holes=[hole])
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=as_section(end), origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=24)
    transition = plan.transitions[0]
    pair = transition.region_pairs[0]
    assert pair.closures and pair.closures[0].scope == "loop"
    broken_closure = replace(pair.closures[0], loop_index=999)
    broken_pair = replace(pair, closures=(broken_closure, *pair.closures[1:]))
    broken_transition = replace(transition, region_pairs=(broken_pair, *transition.region_pairs[1:]))
    broken_plan = replace(plan, transitions=(broken_transition, *plan.transitions[1:]))

    with pytest.raises(ValueError, match=r"interval 0->1.*loop_index 999 out of range"):
        loft_execute_plan(broken_plan, cap_ends=True)


def test_loft_execute_plan_rejects_duplicate_closure_ownership():
    outer = make_rect(size=(1.2, 1.2)).outer
    hole = make_rect(size=(0.4, 0.4)).outer
    start = PlanarShape2D(outer=outer, holes=[])
    end = PlanarShape2D(outer=outer, holes=[hole])
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=as_section(end), origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=24)
    transition = plan.transitions[0]
    pair = transition.region_pairs[0]
    assert pair.closures
    duplicate = pair.closures[0]
    broken_pair = replace(pair, closures=(*pair.closures, duplicate))
    broken_transition = replace(transition, region_pairs=(broken_pair, *transition.region_pairs[1:]))
    broken_plan = replace(plan, transitions=(broken_transition, *plan.transitions[1:]))

    with pytest.raises(ValueError, match=r"interval 0->1.*duplicate closure ownership"):
        loft_execute_plan(broken_plan, cap_ends=True)


def test_loft_execute_plan_rejects_invalid_branch_order_length():
    left0 = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    right0 = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    left1 = make_rect(size=(0.7, 0.9), center=(-1.1, 0.0))
    right1 = make_rect(size=(0.9, 0.7), center=(1.1, 0.0))
    s0 = Section((as_section(left0).regions[0], as_section(right0).regions[0]))
    s1 = Section((as_section(left1).regions[0], as_section(right1).regions[0]))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=24)
    transition = plan.transitions[0]
    broken_transition = replace(transition, branch_order=())
    broken_plan = replace(plan, transitions=(broken_transition, *plan.transitions[1:]))

    with pytest.raises(ValueError, match=r"interval \(0, 1\): branch_order length must match region_pairs"):
        loft_execute_plan(broken_plan, cap_ends=True)


def test_loft_execute_plan_rejects_duplicate_branch_order_ids():
    left0 = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    right0 = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    left1 = make_rect(size=(0.7, 0.9), center=(-1.1, 0.0))
    right1 = make_rect(size=(0.9, 0.7), center=(1.1, 0.0))
    s0 = Section((as_section(left0).regions[0], as_section(right0).regions[0]))
    s1 = Section((as_section(left1).regions[0], as_section(right1).regions[0]))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=24)
    transition = plan.transitions[0]
    assert len(transition.branch_order) == 2
    broken_transition = replace(
        transition,
        branch_order=(transition.branch_order[0], transition.branch_order[0]),
    )
    broken_plan = replace(plan, transitions=(broken_transition, *plan.transitions[1:]))

    with pytest.raises(ValueError, match=r"interval \(0, 1\): branch_order contains duplicate branch IDs"):
        loft_execute_plan(broken_plan, cap_ends=True)


def test_loft_execute_plan_rejects_branch_order_mismatch_with_region_pairs():
    left0 = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    right0 = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    left1 = make_rect(size=(0.7, 0.9), center=(-1.1, 0.0))
    right1 = make_rect(size=(0.9, 0.7), center=(1.1, 0.0))
    s0 = Section((as_section(left0).regions[0], as_section(right0).regions[0]))
    s1 = Section((as_section(left1).regions[0], as_section(right1).regions[0]))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=24)
    transition = plan.transitions[0]
    broken_transition = replace(transition, branch_order=tuple(reversed(transition.branch_order)))
    broken_plan = replace(plan, transitions=(broken_transition, *plan.transitions[1:]))

    with pytest.raises(ValueError, match=r"interval \(0, 1\): branch_order must match region_pairs emission order"):
        loft_execute_plan(broken_plan, cap_ends=True)


def test_loft_execute_plan_rejects_split_birth_without_prev_region_closure():
    source = make_rect(size=(1.2, 1.0), center=(0.0, 0.0))
    left = make_rect(size=(0.8, 0.9), center=(-0.8, 0.0))
    right = make_rect(size=(0.8, 0.9), center=(0.8, 0.0))
    stations = [
        Station(
            t=0.0,
            section=Section((as_section(source).regions[0],)),
            origin=[0, 0, 0],
            u=[1, 0, 0],
            v=[0, 1, 0],
            n=[0, 0, 1],
        ),
        Station(
            t=1.0,
            section=Section((as_section(left).regions[0], as_section(right).regions[0])),
            origin=[0, 0, 1],
            u=[1, 0, 0],
            v=[0, 1, 0],
            n=[0, 0, 1],
        ),
    ]
    plan = loft_plan_sections(stations, samples=24, split_merge_mode="resolve")
    for transition_idx, transition in enumerate(plan.transitions):
        for region_pair_idx, pair in enumerate(transition.region_pairs):
            if pair.action != "split_birth":
                continue
            broken_closures = tuple(
                closure
                for closure in pair.closures
                if not (closure.scope == "region" and closure.side == "prev")
            )
            broken_pair = replace(pair, closures=broken_closures)
            broken_region_pairs = list(transition.region_pairs)
            broken_region_pairs[region_pair_idx] = broken_pair
            broken_transition = replace(transition, region_pairs=tuple(broken_region_pairs))
            broken_transitions = list(plan.transitions)
            broken_transitions[transition_idx] = broken_transition
            broken_plan = replace(plan, transitions=tuple(broken_transitions))
            with pytest.raises(
                ValueError,
                match=r"split_birth requires exactly one prev region closure and zero curr region closures",
            ):
                loft_execute_plan(broken_plan, cap_ends=True)
            return
    raise AssertionError("Expected at least one split_birth region pair in resolved plan.")


def test_loft_execute_plan_rejects_synthetic_birth_closure_on_wrong_side():
    outer = make_rect(size=(1.2, 1.2)).outer
    hole = make_rect(size=(0.4, 0.4)).outer
    start = PlanarShape2D(outer=outer, holes=[])
    end = PlanarShape2D(outer=outer, holes=[hole])
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=as_section(end), origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=24)
    transition = plan.transitions[0]
    pair = transition.region_pairs[0]
    synthetic_birth_idx = next(idx for idx, lp in enumerate(pair.loop_pairs) if lp.role == "synthetic_birth")
    closure_idx = next(
        idx
        for idx, closure in enumerate(pair.closures)
        if closure.scope == "loop" and closure.loop_index == synthetic_birth_idx
    )
    broken_closure = replace(pair.closures[closure_idx], side="curr")
    broken_closures = list(pair.closures)
    broken_closures[closure_idx] = broken_closure
    broken_pair = replace(pair, closures=tuple(broken_closures))
    broken_transition = replace(transition, region_pairs=(broken_pair, *transition.region_pairs[1:]))
    broken_plan = replace(plan, transitions=(broken_transition, *plan.transitions[1:]))

    with pytest.raises(
        ValueError,
        match=r"synthetic_birth loop pair \d+ must be closed on prev side only",
    ):
        loft_execute_plan(broken_plan, cap_ends=True)


def test_loft_execute_plan_rejects_out_of_range_actual_region_ref():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=24)
    transition = plan.transitions[0]
    pair = transition.region_pairs[0]
    broken_pair = replace(
        pair,
        prev_region_ref=replace(pair.prev_region_ref, index=999),
    )
    broken_transition = replace(transition, region_pairs=(broken_pair, *transition.region_pairs[1:]))
    broken_plan = replace(plan, transitions=(broken_transition, *plan.transitions[1:]))
    with pytest.raises(ValueError, match=r"interval 0->1.*prev region ref out of range"):
        loft_execute_plan(broken_plan, cap_ends=True)


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
    with pytest.raises(ValueError, match="ambiguity_mode"):
        loft_sections(stations, samples=24, split_merge_mode="resolve", ambiguity_mode="banana")
    with pytest.raises(ValueError, match="ambiguity_cost_profile"):
        loft_sections(stations, samples=24, split_merge_mode="resolve", ambiguity_cost_profile="banana")
    with pytest.raises(ValueError, match="ambiguity_max_branches"):
        loft_sections(stations, samples=24, split_merge_mode="resolve", ambiguity_max_branches=0)
    with pytest.raises(ValueError, match="fairness_mode"):
        loft_sections(stations, samples=24, split_merge_mode="resolve", fairness_mode="banana")
    with pytest.raises(ValueError, match="fairness_weight"):
        loft_sections(stations, samples=24, split_merge_mode="resolve", fairness_weight=-0.1)
    with pytest.raises(ValueError, match="skeleton_mode"):
        loft_sections(stations, samples=24, split_merge_mode="resolve", skeleton_mode="banana")
    with pytest.raises(ValueError, match="fairness_iterations"):
        loft_sections(stations, samples=24, split_merge_mode="resolve", fairness_iterations=0)


def test_loft_sections_accepts_fairness_and_skeleton_controls_surface():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    mesh = loft_sections(
        stations,
        samples=24,
        cap_ends=True,
        split_merge_mode="resolve",
        fairness_mode="global",
        fairness_weight=0.35,
        skeleton_mode="auto",
        fairness_iterations=20,
    )
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_loft_sections_skeleton_required_fails_when_unavailable():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    with pytest.raises(ValueError, match="skeleton_required_unavailable"):
        loft_sections(
            stations,
            samples=24,
            split_merge_mode="resolve",
            fairness_mode="global",
            skeleton_mode="required",
        )


def test_loft_sections_skeleton_auto_falls_back_and_preserves_mesh_quality():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    mesh = loft_sections(
        stations,
        samples=24,
        split_merge_mode="resolve",
        fairness_mode="global",
        skeleton_mode="auto",
        cap_ends=True,
    )
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_loft_plan_sections_reports_nonzero_fairness_terms_for_split_birth_transition():
    base = as_section(make_rect(size=(1.0, 1.0)))
    extra = as_section(make_rect(size=(0.5, 0.5), center=(2.0, 0.0)))
    s0 = Section((base.regions[0],))
    s1 = Section((base.regions[0], extra.regions[0]))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(
        stations,
        samples=28,
        split_merge_mode="resolve",
        fairness_mode="local",
        fairness_weight=0.2,
        skeleton_mode="auto",
        fairness_iterations=12,
    )
    pre = plan.metadata["fairness_objective_pre"]
    post = plan.metadata["fairness_objective_post"]
    assert pre == post
    assert plan.metadata["fairness_diagnostics"]["branch_crossing_count"] == pytest.approx(
        pre["branch_crossing"]
    )
    assert plan.metadata["fairness_diagnostics"]["continuity_score"] == pytest.approx(
        pre["curvature_continuity"]
    )
    assert plan.metadata["fairness_diagnostics"]["closure_distortion_score"] == pytest.approx(
        pre["closure_stress"]
    )
    assert pre["synthetic_harshness"] > 0.0
    assert pre["closure_stress"] > 0.0
    assert pre["branch_crossing"] >= 0.0
    assert pre["curvature_continuity"] >= 0.0
    assert pre["branch_acceleration"] >= 0.0


def test_loft_plan_sections_global_fairness_converges_on_multistation_ambiguity():
    s0_left = make_circle(radius=0.18, center=(-0.4, -0.4))
    s0_right = make_circle(radius=0.18, center=(0.4, -0.4))
    s1_left = make_circle(radius=0.18, center=(-0.4, 0.0))
    s1_right = make_circle(radius=0.18, center=(0.4, 0.0))
    s2_bottom = make_circle(radius=0.18, center=(0.0, -0.4))
    s2_top = make_circle(radius=0.18, center=(0.0, 0.4))

    sec0 = Section((as_section(s0_left).regions[0], as_section(s0_right).regions[0]))
    sec1 = Section((as_section(s1_left).regions[0], as_section(s1_right).regions[0]))
    sec2 = Section((as_section(s2_bottom).regions[0], as_section(s2_top).regions[0]))
    stations = [
        Station(t=0.0, section=sec0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=0.5, section=sec1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=sec2, origin=[0, 0, 2], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan_a = loft_plan_sections(
        stations,
        samples=28,
        split_merge_mode="resolve",
        fairness_mode="global",
        fairness_weight=0.4,
        fairness_iterations=8,
    )
    plan_b = loft_plan_sections(
        stations,
        samples=28,
        split_merge_mode="resolve",
        fairness_mode="global",
        fairness_weight=0.4,
        fairness_iterations=8,
    )

    assert plan_a.metadata["fairness_optimization_convergence_status"] == "converged"
    assert _plan_transition_signature(plan_a) == _plan_transition_signature(plan_b)


def test_loft_plan_sections_global_fairness_reports_max_iterations_when_budget_limited():
    s0_left = make_circle(radius=0.18, center=(-0.4, -0.4))
    s0_right = make_circle(radius=0.18, center=(0.4, -0.4))
    s1_left = make_circle(radius=0.18, center=(-0.4, 0.0))
    s1_right = make_circle(radius=0.18, center=(0.4, 0.0))
    s2_bottom = make_circle(radius=0.18, center=(0.0, -0.4))
    s2_top = make_circle(radius=0.18, center=(0.0, 0.4))

    sec0 = Section((as_section(s0_left).regions[0], as_section(s0_right).regions[0]))
    sec1 = Section((as_section(s1_left).regions[0], as_section(s1_right).regions[0]))
    sec2 = Section((as_section(s2_bottom).regions[0], as_section(s2_top).regions[0]))
    stations = [
        Station(t=0.0, section=sec0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=0.5, section=sec1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=sec2, origin=[0, 0, 2], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(
        stations,
        samples=28,
        split_merge_mode="resolve",
        fairness_mode="global",
        fairness_weight=0.4,
        fairness_iterations=1,
    )

    assert plan.metadata["fairness_optimization_convergence_status"] == "max_iterations"


def test_loft_plan_sections_global_fairness_not_worse_than_off_on_fixed_fixture():
    a0 = make_circle(radius=0.22, center=(-1.2, 0.0))
    b0 = make_circle(radius=0.22, center=(1.2, 0.0))
    a1 = make_circle(radius=0.22, center=(-1.5, 0.0))
    b1 = make_circle(radius=0.22, center=(1.5, 0.0))
    a2 = make_circle(radius=0.22, center=(-1.5, 0.0))
    b2 = make_circle(radius=0.22, center=(1.5, 0.0))

    s0 = Section((as_section(a0).regions[0], as_section(b0).regions[0]))
    s1 = Section((as_section(a1).regions[0], as_section(b1).regions[0]))
    s2 = Section((as_section(a2).regions[0], as_section(b2).regions[0]))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=0.5, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s2, origin=[0, 0, 2], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    off_plan = loft_plan_sections(
        stations,
        samples=24,
        split_merge_mode="resolve",
        fairness_mode="off",
    )
    global_plan = loft_plan_sections(
        stations,
        samples=24,
        split_merge_mode="resolve",
        fairness_mode="global",
        fairness_weight=0.8,
        fairness_iterations=12,
    )

    off_diag = off_plan.metadata["fairness_diagnostics"]
    global_diag = global_plan.metadata["fairness_diagnostics"]
    off_score = (
        float(off_diag["branch_crossing_count"])
        + float(off_diag["continuity_score"])
        + float(off_diag["closure_distortion_score"])
    )
    global_score = (
        float(global_diag["branch_crossing_count"])
        + float(global_diag["continuity_score"])
        + float(global_diag["closure_distortion_score"])
    )
    assert global_score <= off_score + 1e-9
    assert global_plan.metadata["fairness_optimization_convergence_status"] == "converged"


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


def test_loft_plan_sections_region_2_to_2_symmetric_ambiguity_resolves_deterministically():
    left0 = make_circle(radius=0.36, center=(-0.75, 0.0))
    right0 = make_circle(radius=0.36, center=(0.75, 0.0))
    left1 = make_circle(radius=0.36, center=(-0.75, 0.0))
    right1 = make_circle(radius=0.36, center=(0.75, 0.0))

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

    plan_a = loft_plan_sections(
        stations_a,
        samples=32,
        split_merge_mode="resolve",
        ambiguity_mode="auto",
    )
    plan_b = loft_plan_sections(
        stations_b,
        samples=32,
        split_merge_mode="resolve",
        ambiguity_mode="auto",
    )

    assert plan_a.transitions[0].topology_case == "one_to_one"
    assert plan_b.transitions[0].topology_case == "one_to_one"
    assert _plan_transition_signature(plan_a) == _plan_transition_signature(plan_b)

    mesh_a = loft_execute_plan(plan_a, cap_ends=True)
    mesh_b = loft_execute_plan(plan_b, cap_ends=True)
    assert np.array_equal(mesh_a.vertices, mesh_b.vertices)
    assert np.array_equal(mesh_a.faces, mesh_b.faces)
    _assert_mesh_quality(mesh_a)


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


def test_loft_plan_sections_hole_2_to_2_symmetric_ambiguity_resolves_deterministically():
    outer = make_rect(size=(2.2, 1.5)).outer
    left_hole = make_circle(radius=0.17, center=(-0.52, 0.0)).outer
    right_hole = make_circle(radius=0.17, center=(0.52, 0.0)).outer
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

    plan_a = loft_plan_sections(
        stations_a,
        samples=32,
        split_merge_mode="resolve",
        ambiguity_mode="auto",
    )
    plan_b = loft_plan_sections(
        stations_b,
        samples=32,
        split_merge_mode="resolve",
        ambiguity_mode="auto",
    )
    assert _plan_transition_signature(plan_a) == _plan_transition_signature(plan_b)

    mesh_a = loft_execute_plan(plan_a, cap_ends=True)
    mesh_b = loft_execute_plan(plan_b, cap_ends=True)
    assert np.array_equal(mesh_a.vertices, mesh_b.vertices)
    assert np.array_equal(mesh_a.faces, mesh_b.faces)
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


def test_loft_sections_fail_mode_supports_region_birth_transition():
    base = as_section(make_rect(size=(1.0, 1.0)))
    extra = as_section(make_rect(size=(0.5, 0.5), center=(2.0, 0.0)))
    s0 = Section((base.regions[0],))
    s1 = Section((base.regions[0], extra.regions[0]))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    mesh = loft_sections(stations, samples=24, cap_ends=True, split_merge_mode="fail")
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


def test_loft_sections_fail_mode_supports_region_death_transition():
    base = as_section(make_rect(size=(1.0, 1.0)))
    extra = as_section(make_rect(size=(0.5, 0.5), center=(2.0, 0.0)))
    s0 = Section((base.regions[0], extra.regions[0]))
    s1 = Section((base.regions[0],))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    mesh = loft_sections(stations, samples=24, cap_ends=True, split_merge_mode="fail")
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


def test_loft_sections_resolve_mode_supports_many_to_many_region_2_to_3():
    left = make_rect(size=(0.85, 0.85), center=(-1.0, 0.0))
    right = make_rect(size=(0.85, 0.85), center=(1.0, 0.0))
    left_end = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    center_end = make_rect(size=(0.7, 0.7), center=(0.0, 0.0))
    right_end = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    s0 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1 = Section(
        (
            as_section(left_end).regions[0],
            as_section(center_end).regions[0],
            as_section(right_end).regions[0],
        )
    )
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(
        stations,
        samples=30,
        split_merge_mode="resolve",
        split_merge_steps=8,
        split_merge_bias=0.5,
    )
    mm_transitions = [t for t in plan.transitions if t.topology_case == "many_to_many_expand"]
    assert len(mm_transitions) == 1
    actions = [pair.action for pair in mm_transitions[0].region_pairs]
    assert actions.count("split_match") == 2
    assert actions.count("split_birth") == 1
    assert plan.metadata["region_topology_case_counts"]["many_to_many_expand"] == 1
    assert plan.metadata["region_action_counts"]["split_birth"] >= 1

    mesh = loft_sections(
        stations,
        samples=30,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=8,
        split_merge_bias=0.5,
    )
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_loft_sections_resolve_mode_supports_many_to_many_region_3_to_2():
    left = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    center = make_rect(size=(0.7, 0.7), center=(0.0, 0.0))
    right = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    left_end = make_rect(size=(0.85, 0.85), center=(-1.0, 0.0))
    right_end = make_rect(size=(0.85, 0.85), center=(1.0, 0.0))
    s0 = Section(
        (
            as_section(left).regions[0],
            as_section(center).regions[0],
            as_section(right).regions[0],
        )
    )
    s1 = Section((as_section(left_end).regions[0], as_section(right_end).regions[0]))
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(
        stations,
        samples=30,
        split_merge_mode="resolve",
        split_merge_steps=8,
        split_merge_bias=0.5,
    )
    mm_transitions = [t for t in plan.transitions if t.topology_case == "many_to_many_collapse"]
    assert len(mm_transitions) == 1
    actions = [pair.action for pair in mm_transitions[0].region_pairs]
    assert actions.count("merge_match") == 2
    assert actions.count("merge_death") == 1
    assert plan.metadata["region_topology_case_counts"]["many_to_many_collapse"] == 1
    assert plan.metadata["region_action_counts"]["merge_death"] >= 1

    mesh = loft_sections(
        stations,
        samples=30,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=8,
        split_merge_bias=0.5,
    )
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_loft_plan_sections_many_to_many_region_2_to_3_branch_order_is_deterministic_under_reordering():
    left = make_rect(size=(0.85, 0.85), center=(-1.0, 0.0))
    right = make_rect(size=(0.85, 0.85), center=(1.0, 0.0))
    left_end = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    center_end = make_rect(size=(0.7, 0.7), center=(0.0, 0.0))
    right_end = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    s0_a = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s0_b = Section((as_section(right).regions[0], as_section(left).regions[0]))
    s1_a = Section(
        (
            as_section(left_end).regions[0],
            as_section(center_end).regions[0],
            as_section(right_end).regions[0],
        )
    )
    s1_b = Section(
        (
            as_section(right_end).regions[0],
            as_section(center_end).regions[0],
            as_section(left_end).regions[0],
        )
    )
    stations_a = [
        Station(t=0.0, section=s0_a, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1_a, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    stations_b = [
        Station(t=0.0, section=s0_b, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1_b, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]

    plan_a = loft_plan_sections(
        stations_a,
        samples=30,
        split_merge_mode="resolve",
        split_merge_steps=8,
        split_merge_bias=0.5,
    )
    plan_b = loft_plan_sections(
        stations_b,
        samples=30,
        split_merge_mode="resolve",
        split_merge_steps=8,
        split_merge_bias=0.5,
    )

    assert _plan_transition_signature(plan_a) == _plan_transition_signature(plan_b)
    mesh_a = loft_execute_plan(plan_a, cap_ends=True)
    mesh_b = loft_execute_plan(plan_b, cap_ends=True)
    assert np.array_equal(mesh_a.vertices, mesh_b.vertices)
    assert np.array_equal(mesh_a.faces, mesh_b.faces)
    _assert_mesh_quality(mesh_a)


def test_loft_plan_sections_classifies_many_to_many_region_ambiguity():
    left = make_rect(size=(0.85, 0.85), center=(-1.0, 0.0))
    right = make_rect(size=(0.85, 0.85), center=(1.0, 0.0))
    left_end = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    center_end = make_rect(size=(0.7, 0.7), center=(0.0, 0.0))
    right_end = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    s0 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1 = Section(
        (
            as_section(left_end).regions[0],
            as_section(center_end).regions[0],
            as_section(right_end).regions[0],
        )
    )
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(
        stations,
        samples=30,
        split_merge_mode="resolve",
        ambiguity_mode="auto",
        ambiguity_cost_profile="balanced",
        ambiguity_max_branches=64,
    )
    mm_transitions = [t for t in plan.transitions if t.topology_case == "many_to_many_expand"]
    assert len(mm_transitions) == 1
    assert mm_transitions[0].ambiguity_class in {"symmetry", "permutation"}
    assert plan.metadata["ambiguity_resolved_intervals_count"] >= 1
    class_counts = plan.metadata["ambiguity_class_counts"]
    assert class_counts["symmetry"] + class_counts["permutation"] >= 1


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


def test_loft_plan_sections_ambiguity_mode_fail_rejects_auto_resolvable_ambiguity_with_structured_diagnostics():
    left = make_rect(size=(0.85, 0.85), center=(-1.0, 0.0))
    right = make_rect(size=(0.85, 0.85), center=(1.0, 0.0))
    left_end = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    center_end = make_rect(size=(0.7, 0.7), center=(0.0, 0.0))
    right_end = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    s0 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1 = Section(
        (
            as_section(left_end).regions[0],
            as_section(center_end).regions[0],
            as_section(right_end).regions[0],
        )
    )
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    with pytest.raises(
        ValueError,
        match=(
            r"unsupported_topology_ambiguity.*interval 0->1.*"
            r"ambiguity_class=(symmetry|permutation).*"
            r"tie_break_stage=ambiguity_mode_fail.*"
            r"candidate_count_after_pruning=unknown"
        ),
    ):
        loft_plan_sections(
            stations,
            samples=30,
            split_merge_mode="resolve",
            ambiguity_mode="fail",
        )


def test_loft_plan_sections_reports_structured_residual_ambiguity_diagnostics_for_branch_budget_exhaustion():
    left = make_rect(size=(0.85, 0.85), center=(-1.0, 0.0))
    right = make_rect(size=(0.85, 0.85), center=(1.0, 0.0))
    left_end = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    center_end = make_rect(size=(0.7, 0.7), center=(0.0, 0.0))
    right_end = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    s0 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1 = Section(
        (
            as_section(left_end).regions[0],
            as_section(center_end).regions[0],
            as_section(right_end).regions[0],
        )
    )
    stations = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    with pytest.raises(
        ValueError,
        match=(
            r"unsupported_topology_ambiguity.*interval 0->1.*"
            r"ambiguity_class=(symmetry|permutation).*"
            r"tie_break_stage=candidate_enumeration_limit.*"
            r"candidate_count_after_pruning=\d+"
        ),
    ):
        loft_plan_sections(
            stations,
            samples=30,
            split_merge_mode="resolve",
            ambiguity_mode="auto",
            ambiguity_max_branches=1,
        )


def test_loft_sections_fail_mode_rejects_many_to_many_region_ambiguity():
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
            split_merge_mode="fail",
            split_merge_steps=8,
            split_merge_bias=0.5,
        )


def test_loft_sections_resolve_mode_supports_many_to_many_hole_3_to_2():
    outer = make_rect(size=(2.0, 1.5)).outer
    h0 = make_circle(radius=0.15, center=(-0.55, 0.0)).outer
    h1 = make_circle(radius=0.15, center=(0.0, 0.0)).outer
    h2 = make_circle(radius=0.15, center=(0.55, 0.0)).outer
    k0 = make_circle(radius=0.24, center=(-0.28, 0.0)).outer
    k1 = make_circle(radius=0.24, center=(0.28, 0.0)).outer
    start = PlanarShape2D(outer=outer, holes=[h0, h1, h2])
    end = PlanarShape2D(outer=outer, holes=[k0, k1])
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=as_section(end), origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(
        stations,
        samples=28,
        split_merge_mode="resolve",
        split_merge_steps=1,
        split_merge_bias=0.5,
        ambiguity_mode="auto",
    )
    assert len(plan.transitions) == 1
    pair = plan.transitions[0].region_pairs[0]
    roles = [loop_pair.role for loop_pair in pair.loop_pairs]
    assert roles.count("stable") == 3  # outer + 2 matched holes
    assert roles.count("synthetic_death") == 1  # collapsed unmatched hole
    mesh = loft_sections(
        stations,
        samples=28,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=8,
        split_merge_bias=0.5,
        ambiguity_mode="auto",
    )
    _assert_mesh_quality(mesh)


def test_loft_sections_resolve_mode_supports_many_to_many_hole_2_to_3():
    outer = make_rect(size=(2.0, 1.5)).outer
    h0 = make_circle(radius=0.24, center=(-0.28, 0.0)).outer
    h1 = make_circle(radius=0.24, center=(0.28, 0.0)).outer
    k0 = make_circle(radius=0.15, center=(-0.55, 0.0)).outer
    k1 = make_circle(radius=0.15, center=(0.0, 0.0)).outer
    k2 = make_circle(radius=0.15, center=(0.55, 0.0)).outer
    start = PlanarShape2D(outer=outer, holes=[h0, h1])
    end = PlanarShape2D(outer=outer, holes=[k0, k1, k2])
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=as_section(end), origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(
        stations,
        samples=28,
        split_merge_mode="resolve",
        split_merge_steps=1,
        split_merge_bias=0.5,
        ambiguity_mode="auto",
    )
    assert len(plan.transitions) == 1
    pair = plan.transitions[0].region_pairs[0]
    roles = [loop_pair.role for loop_pair in pair.loop_pairs]
    assert roles.count("stable") == 3  # outer + 2 matched holes
    assert roles.count("synthetic_birth") == 1  # synthetic seed for new hole
    mesh = loft_sections(
        stations,
        samples=28,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=8,
        split_merge_bias=0.5,
        ambiguity_mode="auto",
    )
    _assert_mesh_quality(mesh)


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


def test_loft_plan_sections_region_split_branch_order_is_deterministic_under_reordering():
    source = make_rect(size=(1.2, 1.0), center=(0.0, 0.0))
    left = make_rect(size=(0.8, 0.9), center=(-0.8, 0.0))
    right = make_rect(size=(0.8, 0.9), center=(0.8, 0.0))
    s0 = Section((as_section(source).regions[0],))
    s1_ordered = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1_swapped = Section((as_section(right).regions[0], as_section(left).regions[0]))

    stations_a = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1_ordered, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    stations_b = [
        Station(t=0.0, section=s0, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1_swapped, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]

    plan_a = loft_plan_sections(
        stations_a,
        samples=30,
        split_merge_mode="resolve",
        split_merge_steps=8,
        split_merge_bias=0.5,
    )
    plan_b = loft_plan_sections(
        stations_b,
        samples=30,
        split_merge_mode="resolve",
        split_merge_steps=8,
        split_merge_bias=0.5,
    )

    assert _plan_transition_signature(plan_a) == _plan_transition_signature(plan_b)
    mesh_a = loft_execute_plan(plan_a, cap_ends=True)
    mesh_b = loft_execute_plan(plan_b, cap_ends=True)
    assert np.array_equal(mesh_a.vertices, mesh_b.vertices)
    assert np.array_equal(mesh_a.faces, mesh_b.faces)


def test_loft_plan_sections_region_merge_branch_order_is_deterministic_under_reordering():
    left = make_rect(size=(0.8, 0.9), center=(-0.8, 0.0))
    right = make_rect(size=(0.8, 0.9), center=(0.8, 0.0))
    target = make_rect(size=(1.2, 1.0), center=(0.0, 0.0))
    s0_ordered = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s0_swapped = Section((as_section(right).regions[0], as_section(left).regions[0]))
    s1 = Section((as_section(target).regions[0],))

    stations_a = [
        Station(t=0.0, section=s0_ordered, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    stations_b = [
        Station(t=0.0, section=s0_swapped, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]

    plan_a = loft_plan_sections(
        stations_a,
        samples=30,
        split_merge_mode="resolve",
        split_merge_steps=8,
        split_merge_bias=0.5,
    )
    plan_b = loft_plan_sections(
        stations_b,
        samples=30,
        split_merge_mode="resolve",
        split_merge_steps=8,
        split_merge_bias=0.5,
    )

    assert _plan_transition_signature(plan_a) == _plan_transition_signature(plan_b)
    mesh_a = loft_execute_plan(plan_a, cap_ends=True)
    mesh_b = loft_execute_plan(plan_b, cap_ends=True)
    assert np.array_equal(mesh_a.vertices, mesh_b.vertices)
    assert np.array_equal(mesh_a.faces, mesh_b.faces)


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


def test_loft_real_world_splitter_manifold_example_meshes_are_watertight():
    module_path = Path("docs/examples/loft/real_world/loft_splitter_manifold_example.py")
    spec = importlib.util.spec_from_file_location("loft_splitter_manifold_example", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    scene = module.build()
    meshes = scene if isinstance(scene, list) else [scene]
    assert meshes, "Expected at least one mesh from splitter manifold build()."
    for mesh in meshes:
        _assert_mesh_quality(mesh)
