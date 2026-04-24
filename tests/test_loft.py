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
    SurfaceBody,
    SurfaceConsumerCollection,
    export_tessellation_request,
    loft,
    loft_execute_plan,
    loft_endcaps,
    loft_plan_ambiguities,
    loft_plan_sections,
    loft_sections,
    make_surface_mesh_adapter,
    mesh_from_surface_body,
    preview_tessellation_request,
    tessellate_surface_body,
    as_section,
)
from impression.modeling.drawing2d import Path2D, PlanarShape2D, make_circle, make_rect
from impression.modeling.loft import (
    LoftPlanningBlockedError,
    _loft_execute_plan_surface,
    _loft_profiles_surface,
    _loft_surface_consumer_handoff,
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


def test_private_surface_loft_executor_maps_simple_plan_to_surface_body() -> None:
    start = make_rect(size=(1.0, 1.0))
    end = make_rect(size=(0.6, 1.4))
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

    plan = loft_plan_sections(stations, samples=24)
    body = _loft_execute_plan_surface(plan, cap_ends=False)

    assert isinstance(body, SurfaceBody)
    assert body.shell_count == 1
    assert body.patch_count == 1
    assert [patch.family for patch in body.iter_patches(world=False)] == ["ruled"]
    assert len(body.shells[0].seams) == 1
    wrap_pairs = {(boundary.patch_index, boundary.boundary_id) for boundary in body.shells[0].seams[0].boundaries}
    assert wrap_pairs == {(0, "bottom"), (0, "top")}

    result = tessellate_surface_body(body, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_private_surface_loft_executor_adds_planar_caps_when_requested() -> None:
    start = make_rect(size=(1.0, 1.0))
    end = make_rect(size=(0.6, 1.4))
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0.0, 0.0, 0.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
        Station(t=1.0, section=as_section(end), origin=[0.0, 0.0, 1.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
    ]

    plan = loft_plan_sections(stations, samples=24)
    body = _loft_execute_plan_surface(plan, cap_ends=True)

    assert isinstance(body, SurfaceBody)
    assert body.patch_count == 3
    assert [patch.family for patch in body.iter_patches(world=False)].count("ruled") == 1
    assert [patch.family for patch in body.iter_patches(world=False)].count("planar") == 2


def test_private_surface_loft_executor_reuses_station_seams_across_adjacent_intervals() -> None:
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(0.8, 1.2)),
        make_rect(size=(0.6, 1.4)),
    ]
    stations = [
        Station(t=float(index), section=as_section(profile), origin=[0.0, 0.0, float(index)], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0])
        for index, profile in enumerate(profiles)
    ]

    plan = loft_plan_sections(stations, samples=24)
    body = _loft_execute_plan_surface(plan, cap_ends=False)

    assert body.patch_count == 2
    assert body.shells[0].seams
    assert len(body.shells[0].seams) == 3
    seam_pairs = [
        {(boundary.patch_index, boundary.boundary_id) for boundary in seam.boundaries}
        for seam in body.shells[0].seams
    ]
    assert {(0, "right"), (1, "left")} in seam_pairs
    assert {(0, "bottom"), (0, "top")} in seam_pairs
    assert {(1, "bottom"), (1, "top")} in seam_pairs


def test_private_surface_loft_executor_emits_loop_closure_cap_for_synthetic_hole_birth() -> None:
    outer = make_rect(size=(1.2, 1.2)).outer
    hole = make_rect(size=(0.4, 0.4)).outer
    start = PlanarShape2D(outer=outer, holes=[])
    end = PlanarShape2D(outer=outer, holes=[hole])
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0.0, 0.0, 0.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
        Station(t=1.0, section=as_section(end), origin=[0.0, 0.0, 1.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
    ]

    plan = loft_plan_sections(stations, samples=28)
    body = _loft_execute_plan_surface(plan, cap_ends=False)

    assert isinstance(body, SurfaceBody)
    assert body.patch_count == 3
    planar_roles = [
        patch.metadata.get("kernel", {}).get("surface_role")
        for patch in body.iter_patches(world=False)
        if patch.family == "planar"
    ]
    assert planar_roles == ["closure-cap"]


def test_private_surface_loft_executor_emits_region_closure_cap_for_region_death() -> None:
    base = as_section(make_rect(size=(1.0, 1.0)))
    extra = as_section(make_rect(size=(0.5, 0.5), center=(2.0, 0.0)))
    s0 = Section((base.regions[0], extra.regions[0]))
    s1 = Section((base.regions[0],))
    stations = [
        Station(t=0.0, section=s0, origin=[0.0, 0.0, 0.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
        Station(t=1.0, section=s1, origin=[0.0, 0.0, 1.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
    ]

    plan = loft_plan_sections(stations, samples=24)
    body = _loft_execute_plan_surface(plan, cap_ends=False)

    assert isinstance(body, SurfaceBody)
    planar_patches = [patch for patch in body.iter_patches(world=False) if patch.family == "planar"]
    assert planar_patches
    assert any(
        patch.metadata.get("kernel", {}).get("surface_role") == "closure-cap"
        and patch.metadata.get("kernel", {}).get("closure_scope") == "region"
        for patch in planar_patches
    )


def test_private_surface_loft_consumer_handoff_uses_standard_surface_collection_and_tessellation() -> None:
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(0.8, 1.2)),
        make_rect(size=(0.6, 1.4)),
    ]
    stations = [
        Station(t=float(index), section=as_section(profile), origin=[0.0, 0.0, float(index)], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0])
        for index, profile in enumerate(profiles)
    ]
    plan = loft_plan_sections(stations, samples=24)

    collection = _loft_surface_consumer_handoff(plan, cap_ends=True, metadata={"fixture": "simple"})

    assert isinstance(collection, SurfaceConsumerCollection)
    assert len(collection.items) == 1
    assert collection.metadata["producer"] == "loft"
    assert collection.metadata["executor"] == "surface"
    assert collection.metadata["fixture"] == "simple"

    body = collection.items[0].body
    preview = tessellate_surface_body(body, preview_tessellation_request(require_watertight=False))
    export = tessellate_surface_body(body, export_tessellation_request(require_watertight=False))
    adapter = make_surface_mesh_adapter(export_tessellation_request(require_watertight=False))
    adapted = adapter.convert(body)
    direct_mesh = mesh_from_surface_body(body, export_tessellation_request(require_watertight=False))

    assert preview.body_identity == export.body_identity == body.stable_identity
    assert preview.mesh.n_faces > 0
    assert export.mesh.n_faces > 0
    assert adapted.mesh.n_faces == direct_mesh.n_faces
    assert np.array_equal(adapted.mesh.faces, direct_mesh.faces)


def test_private_surface_loft_consumer_handoff_supports_staged_split_merge_output() -> None:
    left = make_rect(size=(0.9, 0.9), center=(-0.2, 0.0))
    right = make_rect(size=(0.9, 0.9), center=(0.2, 0.0))
    merged = make_rect(size=(1.2, 1.0), center=(0.0, 0.0))
    s0 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1 = Section((as_section(merged).regions[0],))
    stations = [
        Station(t=0.0, section=s0, origin=[0.0, 0.0, 0.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
        Station(t=1.0, section=s1, origin=[0.0, 0.0, 1.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
    ]
    plan = loft_plan_sections(
        stations,
        samples=32,
        split_merge_mode="resolve",
        split_merge_steps=10,
        split_merge_bias=0.5,
    )

    collection = _loft_surface_consumer_handoff(plan, cap_ends=True, metadata={"fixture": "merge"})
    body = collection.items[0].body
    preview = tessellate_surface_body(body, preview_tessellation_request(require_watertight=False))
    export = tessellate_surface_body(body, export_tessellation_request())
    adapter = make_surface_mesh_adapter(export_tessellation_request())
    adapted = adapter.convert(body)

    assert collection.metadata["fixture"] == "merge"
    assert preview.mesh.n_faces > 0
    assert export.classification == "closed"
    assert export.analysis.is_watertight is True
    assert adapted.analysis.is_watertight is True


def test_private_surface_loft_executor_with_end_caps_tessellates_closed_simple_loft() -> None:
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(0.8, 1.2)),
    ]
    stations = [
        Station(t=float(index), section=as_section(profile), origin=[0.0, 0.0, float(index)], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0])
        for index, profile in enumerate(profiles)
    ]
    plan = loft_plan_sections(stations, samples=24)

    body = _loft_execute_plan_surface(plan, cap_ends=True)
    result = tessellate_surface_body(body, export_tessellation_request())

    assert result.classification == "closed"
    assert result.analysis.is_watertight is True
    assert result.analysis.boundary_edges == 0


@pytest.mark.parametrize(
    ("start_cap", "end_cap"),
    [
        ("taper", "none"),
        ("none", "dome"),
        ("slope", "slope"),
    ],
)
def test_private_surface_loft_profiles_surface_supports_nonflat_caps(
    start_cap: str,
    end_cap: str,
) -> None:
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(1.0, 1.0)),
    ]

    body = _loft_profiles_surface(
        profiles,
        samples=24,
        start_cap=start_cap,
        end_cap=end_cap,
        cap_steps=4,
        start_cap_length=0.5,
        end_cap_length=0.75,
        cap_scale_dims="both",
    )

    assert isinstance(body, SurfaceBody)
    assert body.patch_count > 3
    result = tessellate_surface_body(body, export_tessellation_request())
    assert result.classification == "closed"
    assert result.analysis.is_watertight is True
    assert result.analysis.boundary_edges == 0


def test_private_surface_loft_executor_handles_split_birth_as_patch_group_with_closure_cap() -> None:
    outer = make_rect(size=(1.2, 1.2)).outer
    hole = make_rect(size=(0.4, 0.4)).outer
    start = PlanarShape2D(outer=outer, holes=[])
    end = PlanarShape2D(outer=outer, holes=[hole])
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0.0, 0.0, 0.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
        Station(t=1.0, section=as_section(end), origin=[0.0, 0.0, 1.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
    ]

    plan = loft_plan_sections(stations, samples=28, split_merge_mode="resolve")
    body = _loft_execute_plan_surface(plan, cap_ends=False)

    ruled_patches = [patch for patch in body.iter_patches(world=False) if patch.family == "ruled"]
    planar_patches = [patch for patch in body.iter_patches(world=False) if patch.family == "planar"]
    assert len(ruled_patches) >= 2
    assert len(planar_patches) == 1
    loop_roles = {
        patch.metadata.get("kernel", {}).get("loop_role")
        for patch in ruled_patches
    }
    assert "synthetic_birth" in loop_roles
    assert any(
        patch.metadata.get("kernel", {}).get("closure_scope") == "loop"
        for patch in planar_patches
    )

    result = tessellate_surface_body(body, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_private_surface_loft_executor_handles_split_birth_closed_with_end_caps() -> None:
    outer = make_rect(size=(1.2, 1.2)).outer
    hole = make_rect(size=(0.4, 0.4)).outer
    start = PlanarShape2D(outer=outer, holes=[])
    end = PlanarShape2D(outer=outer, holes=[hole])
    stations = [
        Station(t=0.0, section=as_section(start), origin=[0.0, 0.0, 0.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
        Station(t=1.0, section=as_section(end), origin=[0.0, 0.0, 1.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
    ]

    plan = loft_plan_sections(stations, samples=28, split_merge_mode="resolve")
    body = _loft_execute_plan_surface(plan, cap_ends=True)
    result = tessellate_surface_body(body, export_tessellation_request())

    assert result.classification == "closed"
    assert result.analysis.is_watertight is True
    assert result.analysis.boundary_edges == 0


def test_private_surface_loft_executor_handles_merge_death_as_patch_group_with_region_closure() -> None:
    left = make_rect(size=(0.9, 0.9), center=(-0.2, 0.0))
    right = make_rect(size=(0.9, 0.9), center=(0.2, 0.0))
    merged = make_rect(size=(1.2, 1.0), center=(0.0, 0.0))
    s0 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1 = Section((as_section(merged).regions[0],))
    stations = [
        Station(t=0.0, section=s0, origin=[0.0, 0.0, 0.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
        Station(t=1.0, section=s1, origin=[0.0, 0.0, 1.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
    ]

    plan = loft_plan_sections(
        stations,
        samples=32,
        split_merge_mode="resolve",
        split_merge_steps=10,
        split_merge_bias=0.5,
    )
    body = _loft_execute_plan_surface(plan, cap_ends=False)

    ruled_patches = [patch for patch in body.iter_patches(world=False) if patch.family == "ruled"]
    planar_patches = [patch for patch in body.iter_patches(world=False) if patch.family == "planar"]
    assert len(ruled_patches) >= 2
    region_actions = {
        patch.metadata.get("kernel", {}).get("region_action")
        for patch in ruled_patches
    }
    assert "merge_death" in region_actions
    assert any(
        patch.metadata.get("kernel", {}).get("closure_scope") == "region"
        for patch in planar_patches
    )

    result = tessellate_surface_body(body, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_private_surface_loft_executor_handles_merge_death_closed_with_end_caps() -> None:
    left = make_rect(size=(0.9, 0.9), center=(-0.2, 0.0))
    right = make_rect(size=(0.9, 0.9), center=(0.2, 0.0))
    merged = make_rect(size=(1.2, 1.0), center=(0.0, 0.0))
    s0 = Section((as_section(left).regions[0], as_section(right).regions[0]))
    s1 = Section((as_section(merged).regions[0],))
    stations = [
        Station(t=0.0, section=s0, origin=[0.0, 0.0, 0.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
        Station(t=1.0, section=s1, origin=[0.0, 0.0, 1.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0], n=[0.0, 0.0, 1.0]),
    ]

    plan = loft_plan_sections(
        stations,
        samples=32,
        split_merge_mode="resolve",
        split_merge_steps=10,
        split_merge_bias=0.5,
    )
    body = _loft_execute_plan_surface(plan, cap_ends=True)
    result = tessellate_surface_body(body, export_tessellation_request())

    assert result.classification == "closed"
    assert result.analysis.is_watertight is True
    assert result.analysis.boundary_edges == 0


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

    assert plan.schema_version == 1
    assert plan.planner_name == "loft_plan_sections"
    assert plan.plan_header == {
        "schema_version": 1,
        "planner": "loft_plan_sections",
        "samples": 24,
        "station_count": 2,
        "interval_count": 1,
    }
    assert plan.sequence_metadata == {
        "split_merge_mode": "fail",
        "split_merge_steps": 8,
        "split_merge_bias": 0.5,
        "ambiguity_mode": "fail",
        "ambiguity_selection_policy": "required",
        "ambiguity_cost_profile": "balanced",
        "ambiguity_max_branches": 64,
        "disambiguation_mode": "deterministic",
        "disambiguation_seed": plan.metadata["disambiguation_seed"],
        "probabilistic_trials": 64,
        "probabilistic_temperature": 0.25,
        "probabilistic_min_confidence": 0.65,
        "probabilistic_fallback": "deterministic",
        "fairness_mode": "local",
        "fairness_weight": 0.2,
        "fairness_iterations": 12,
        "skeleton_mode": "auto",
    }
    assert plan.summary_metadata["ambiguity_resolved_intervals_count"] == 0
    assert plan.summary_metadata["ambiguity_failed_intervals_count"] == 0
    assert plan.summary_metadata["probabilistic_selected_confidence"] == 1.0
    assert plan.summary_metadata["probabilistic_candidate_count"] == 0
    assert plan.summary_metadata["probabilistic_selected_candidate_ids"] == {}
    assert plan.is_executable is True
    assert plan.blocking_status == "none"

    station = stations[0]
    assert station.progression == 0.0
    assert station.topology_state is station.normalized_topology_state
    assert station.topology_state is not None
    assert len(station.topology_state.regions) == len(base.regions)
    assert np.allclose(
        station.topology_state.regions[0].outer.points,
        station.normalized_topology_state.regions[0].outer.points,
    )
    origin, u_axis, v_axis, n_axis = station.placement_frame
    assert np.allclose(origin, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(u_axis, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(v_axis, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(n_axis, np.array([0.0, 0.0, 1.0]))

    planned_station = plan.stations[0]
    assert planned_station.progression == 0.0
    assert planned_station.normalized_regions == planned_station.regions
    origin, u_axis, v_axis, n_axis = planned_station.placement_frame
    assert np.allclose(origin, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(u_axis, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(v_axis, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(n_axis, np.array([0.0, 0.0, 1.0]))

    transition = plan.transitions[0]
    assert transition.planned_state_indices == (0, 1)
    assert transition.execution_eligibility == "executable"
    assert transition.blocking_status == "none"
    assert transition.executor_operator_families == ("continuity",)
    assert transition.executor_operator_payloads[0]["operator_family"] == "continuity"
    assert transition.executor_operator_payloads[0]["branch_id"] == transition.region_pairs[0].branch_id


def test_station_normalizes_topology_order_and_reorders_directional_correspondence() -> None:
    left = as_section(make_rect(size=(0.5, 0.5), center=(-2.0, 0.0))).regions[0]
    right = as_section(make_rect(size=(1.0, 1.0), center=(2.0, 0.0))).regions[0]
    authored = Section((right, left))

    station = Station(
        t=0.0,
        section=authored,
        origin=[0.0, 0.0, 0.0],
        u=[1.0, 0.0, 0.0],
        v=[0.0, 1.0, 0.0],
        n=[0.0, 0.0, 1.0],
        predecessor_ids=(("right-prev",), ("left-prev",)),
        successor_ids=(("right-next",), ("left-next",)),
    )

    assert station.normalized_topology_state is not None
    centroids = [
        float(np.mean(region.outer.points[:, 0]))
        for region in station.normalized_topology_state.regions
    ]
    assert centroids[0] < centroids[1]
    assert station.predecessor_ids == (frozenset({"left-prev"}), frozenset({"right-prev"}))
    assert station.successor_ids == (frozenset({"left-next"}), frozenset({"right-next"}))
    assert station.directional_correspondence == (
        {"predecessor_ids": frozenset({"left-prev"}), "successor_ids": frozenset({"left-next"})},
        {"predecessor_ids": frozenset({"right-prev"}), "successor_ids": frozenset({"right-next"})},
    )


def test_station_directional_correspondence_requires_section_and_valid_lengths() -> None:
    with pytest.raises(ValueError, match="Directional correspondence requires section topology"):
        Station(
            t=0.0,
            section=None,
            origin=[0.0, 0.0, 0.0],
            u=[1.0, 0.0, 0.0],
            v=[0.0, 1.0, 0.0],
            n=[0.0, 0.0, 1.0],
            predecessor_ids=(("a",),),
        )

    base = as_section(make_rect(size=(1.0, 1.0)))
    with pytest.raises(ValueError, match="predecessor_ids must have one entry per normalized region"):
        Station(
            t=0.0,
            section=base,
            origin=[0.0, 0.0, 0.0],
            u=[1.0, 0.0, 0.0],
            v=[0.0, 1.0, 0.0],
            n=[0.0, 0.0, 1.0],
            predecessor_ids=(("a",), ("b",)),
        )


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


def test_loft_execute_plan_rejects_invalid_plan_sample_count():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=24)
    broken_plan = replace(plan, samples=2)

    with pytest.raises(ValueError, match="samples must be >= 3"):
        loft_execute_plan(broken_plan, cap_ends=True)


def test_loft_execute_plan_rejects_negative_failed_ambiguity_count():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=24)
    broken_metadata = dict(plan.metadata)
    broken_metadata["ambiguity_failed_intervals_count"] = -1
    broken_plan = replace(plan, metadata=broken_metadata)

    with pytest.raises(ValueError, match="ambiguity_failed_intervals_count must be >= 0"):
        loft_execute_plan(broken_plan, cap_ends=True)


def test_loft_execute_plan_rejects_negative_fairness_diagnostic_value():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    plan = loft_plan_sections(stations, samples=24)
    broken_metadata = dict(plan.metadata)
    broken_metadata["fairness_diagnostics"] = dict(plan.metadata["fairness_diagnostics"])
    broken_metadata["fairness_diagnostics"]["continuity_score"] = -1.0
    broken_plan = replace(plan, metadata=broken_metadata)

    with pytest.raises(ValueError, match=r"fairness_diagnostics\['continuity_score'\] must be >= 0"):
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


def test_loft_sections_rejects_samples_below_three():
    base = as_section(make_rect(size=(1.0, 1.0)))
    stations = [
        Station(t=0.0, section=base, origin=[0, 0, 0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=base, origin=[0, 0, 1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    with pytest.raises(ValueError, match="samples must be >= 3"):
        loft_sections(stations, samples=2)


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


def test_loft_plan_sections_exposes_many_to_many_candidate_set_and_decomposition_order():
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

    transition = next(t for t in plan.transitions if t.topology_case == "many_to_many_expand")
    assert transition.is_many_to_many is True
    assert transition.many_to_many_decomposition_order == (
        "isolate_candidate_set",
        "directional_correspondence",
        "direct_correspondence",
        "birth_death",
        "one_to_many_or_many_to_one",
    )

    candidate_set = transition.many_to_many_candidate_set
    assert candidate_set is not None
    assert candidate_set.prev_region_indices == (0, 1)
    assert candidate_set.curr_region_indices == (0, 1, 2)
    assert candidate_set.matched_prev_region_indices == (0, 1)
    assert candidate_set.matched_curr_region_indices == (0, 2)
    assert candidate_set.residual_prev_region_indices == ()
    assert candidate_set.residual_curr_region_indices == (1,)

    decomposability = transition.many_to_many_decomposability
    assert decomposability is not None
    assert decomposability.continues_automatically is True
    assert decomposability.gate_reached is False
    assert decomposability.residual_prev_region_indices == ()
    assert decomposability.residual_curr_region_indices == (1,)


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


def test_loft_plan_sections_raises_structured_blocked_error_with_locator_and_request() -> None:
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
    with pytest.raises(LoftPlanningBlockedError) as excinfo:
        loft_plan_sections(
            stations,
            samples=30,
            split_merge_mode="resolve",
            ambiguity_mode="fail",
        )

    error = excinfo.value
    assert error.ambiguity_record.interval == (0, 1)
    assert error.ambiguity_record.topology_state_index == 1
    assert error.ambiguity_record.ambiguous_region_indices == (0, 1, 2)
    assert error.ambiguity_record.relationship_group == "many_to_many_regions"
    assert error.constraint_request.interval == (0, 1)
    assert error.constraint_request.topology_state_index == 1
    assert error.constraint_request.ambiguous_region_indices == (0, 1, 2)
    assert error.constraint_request.requested_ties == ("predecessor_ids", "successor_ids")
    assert error.constraint_request.relationship_group == "many_to_many_regions"


def test_auto_resolved_ambiguous_transition_remains_executable() -> None:
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
        ambiguity_max_branches=64,
    )
    assert plan.transitions[0].ambiguity_class in {"symmetry", "permutation"}
    assert plan.transitions[0].execution_eligibility == "executable"
    assert plan.transitions[0].blocking_status == "none"
    assert plan.is_executable is True
    plan.require_executable()


def test_loft_plan_ambiguities_reports_stable_candidate_ids_for_ambiguous_interval() -> None:
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
    report_a = loft_plan_ambiguities(stations, samples=30, split_merge_mode="resolve")
    report_b = loft_plan_ambiguities(stations, samples=30, split_merge_mode="resolve")

    assert len(report_a.intervals) == 1
    assert report_a.intervals[0].interval == (0, 1)
    assert report_a.intervals[0].ambiguity_class in {"symmetry", "permutation"}
    assert tuple(candidate.candidate_id for candidate in report_a.intervals[0].candidates) == tuple(
        candidate.candidate_id for candidate in report_b.intervals[0].candidates
    )


def test_loft_plan_sections_interactive_mode_requires_selection_for_ambiguous_interval() -> None:
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

    with pytest.raises(ValueError, match=r"invalid_ambiguity_selection.*missing selection for interval 0->1"):
        loft_plan_sections(
            stations,
            samples=30,
            split_merge_mode="resolve",
            ambiguity_mode="interactive",
        )


def test_loft_plan_sections_interactive_mode_accepts_stable_candidate_selection() -> None:
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
    report = loft_plan_ambiguities(stations, samples=30, split_merge_mode="resolve")
    selection = {(0, 1): report.intervals[0].candidates[-1].candidate_id}

    plan_a = loft_plan_sections(
        stations,
        samples=30,
        split_merge_mode="resolve",
        ambiguity_mode="interactive",
        ambiguity_selection=selection,
    )
    plan_b = loft_plan_sections(
        stations,
        samples=30,
        split_merge_mode="resolve",
        ambiguity_mode="interactive",
        ambiguity_selection=selection,
    )

    assert _plan_transition_signature(plan_a) == _plan_transition_signature(plan_b)


def test_loft_plan_sections_interactive_best_effort_falls_back_deterministically() -> None:
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
    auto_plan = loft_plan_sections(stations, samples=30, split_merge_mode="resolve", ambiguity_mode="auto")
    best_effort_plan = loft_plan_sections(
        stations,
        samples=30,
        split_merge_mode="resolve",
        ambiguity_mode="interactive",
        ambiguity_selection_policy="best_effort",
    )
    assert _plan_transition_signature(auto_plan) == _plan_transition_signature(best_effort_plan)


def test_loft_plan_sections_interactive_mode_rejects_unknown_candidate_id() -> None:
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
    with pytest.raises(ValueError, match=r"invalid_ambiguity_selection.*unknown candidate_id"):
        loft_plan_sections(
            stations,
            samples=30,
            split_merge_mode="resolve",
            ambiguity_mode="interactive",
            ambiguity_selection={(0, 1): "many_to_many_expand:not-a-real-candidate"},
        )


def test_loft_plan_sections_probabilistic_mode_replays_same_seed() -> None:
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
    plan_a = loft_plan_sections(
        stations,
        samples=30,
        split_merge_mode="resolve",
        ambiguity_mode="auto",
        disambiguation_mode="probabilistic",
        disambiguation_seed=7,
        probabilistic_trials=48,
        probabilistic_temperature=0.35,
    )
    plan_b = loft_plan_sections(
        stations,
        samples=30,
        split_merge_mode="resolve",
        ambiguity_mode="auto",
        disambiguation_mode="probabilistic",
        disambiguation_seed=7,
        probabilistic_trials=48,
        probabilistic_temperature=0.35,
    )

    assert plan_a.metadata["disambiguation_mode"] == "probabilistic"
    assert plan_a.metadata["disambiguation_seed"] == 7
    assert plan_a.metadata["probabilistic_trials"] == 48
    assert _plan_transition_signature(plan_a) == _plan_transition_signature(plan_b)
    assert plan_a.metadata["probabilistic_selected_candidate_ids"] == plan_b.metadata["probabilistic_selected_candidate_ids"]


def test_loft_plan_sections_probabilistic_low_confidence_can_fail() -> None:
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
    with pytest.raises(ValueError, match=r"probabilistic_disambiguation_failed.*selected_confidence="):
        loft_plan_sections(
            stations,
            samples=30,
            split_merge_mode="resolve",
            ambiguity_mode="auto",
            disambiguation_mode="probabilistic",
            disambiguation_seed=11,
            probabilistic_trials=16,
            probabilistic_min_confidence=1.1,
            probabilistic_fallback="fail",
        )


def test_loft_plan_sections_probabilistic_low_confidence_can_fallback_deterministically() -> None:
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
        disambiguation_mode="probabilistic",
        disambiguation_seed=11,
        probabilistic_trials=16,
        probabilistic_min_confidence=1.1,
        probabilistic_fallback="deterministic",
    )
    assert plan.is_executable is True
    assert plan.metadata["probabilistic_selected_confidence"] < 1.1


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


def test_loft_plan_ambiguities_reports_hole_candidate_ids_for_many_to_many_holes() -> None:
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
    report = loft_plan_ambiguities(stations, samples=28, split_merge_mode="resolve")
    assert len(report.intervals) == 1
    interval = report.intervals[0]
    assert interval.relationship_group == "hole_many_to_many:r0"
    assert interval.candidates
    assert all(candidate.candidate_id.startswith("hole:r0:") for candidate in interval.candidates)


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


def test_loft_plan_sections_interactive_mode_accepts_hole_candidate_selection() -> None:
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
    report = loft_plan_ambiguities(stations, samples=28, split_merge_mode="resolve")
    selection = {(0, 1): report.intervals[0].candidates[-1].candidate_id}
    plan_a = loft_plan_sections(
        stations,
        samples=28,
        split_merge_mode="resolve",
        ambiguity_mode="interactive",
        ambiguity_selection=selection,
    )
    plan_b = loft_plan_sections(
        stations,
        samples=28,
        split_merge_mode="resolve",
        ambiguity_mode="interactive",
        ambiguity_selection=selection,
    )
    assert _plan_transition_signature(plan_a) == _plan_transition_signature(plan_b)


def test_loft_plan_sections_probabilistic_mode_replays_same_seed_for_hole_ambiguity() -> None:
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
    plan_a = loft_plan_sections(
        stations,
        samples=28,
        split_merge_mode="resolve",
        ambiguity_mode="auto",
        disambiguation_mode="probabilistic",
        disambiguation_seed=19,
        probabilistic_trials=32,
        probabilistic_temperature=0.3,
    )
    plan_b = loft_plan_sections(
        stations,
        samples=28,
        split_merge_mode="resolve",
        ambiguity_mode="auto",
        disambiguation_mode="probabilistic",
        disambiguation_seed=19,
        probabilistic_trials=32,
        probabilistic_temperature=0.3,
    )
    assert _plan_transition_signature(plan_a) == _plan_transition_signature(plan_b)


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


def test_loft_real_world_hourglass_vessel_example_meshes_are_watertight():
    module_path = Path("docs/examples/loft/real_world/loft_hourglass_vessel_example.py")
    spec = importlib.util.spec_from_file_location("loft_hourglass_vessel_example", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    scene = module.build(module.TEST_PARAMETERS, module.TEST_QUALITY)
    meshes = scene if isinstance(scene, list) else [scene]
    assert meshes, "Expected at least one mesh from hourglass vessel build()."
    for mesh in meshes:
        _assert_mesh_quality(mesh)
