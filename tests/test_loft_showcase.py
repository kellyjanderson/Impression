from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from impression.mesh import analyze_mesh
from impression.modeling import Loft, export_tessellation_request, loft_plan_ambiguities, tessellate_surface_body

from tests.loft_showcases import (
    build_ambiguous_hole_cluster_profiles,
    build_branching_manifold_profiles,
    build_perforated_vessel_profiles,
)


def _assert_mesh_quality(mesh) -> None:
    analysis = analyze_mesh(mesh)
    assert analysis.boundary_edges == 0, analysis.issues()
    assert analysis.nonmanifold_edges == 0, analysis.issues()
    assert analysis.degenerate_faces == 0, analysis.issues()


def _load_real_world_example(module_name: str, relative_path: str):
    module_path = Path(relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_branching_manifold_showcase_loft_is_watertight() -> None:
    profiles, path = build_branching_manifold_profiles()
    body = Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        samples=42,
        split_merge_mode="resolve",
        split_merge_steps=10,
        cap_ends=True,
    )
    mesh = tessellate_surface_body(body, export_tessellation_request()).mesh
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    assert (x_max - x_min) > 1.8
    assert (y_max - y_min) > 0.5
    assert (z_max - z_min) > 2.5
    _assert_mesh_quality(mesh)


def test_perforated_vessel_showcase_handles_hole_birth_split_merge_and_death() -> None:
    profiles, path = build_perforated_vessel_profiles()
    body = Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        samples=44,
        split_merge_mode="resolve",
        split_merge_steps=10,
        cap_ends=True,
    )
    mesh = tessellate_surface_body(body, export_tessellation_request()).mesh
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    assert (x_max - x_min) > 1.5
    assert (y_max - y_min) > 1.5
    assert (z_max - z_min) > 3.0
    _assert_mesh_quality(mesh)


def test_ambiguous_hole_cluster_showcase_supports_interactive_selection() -> None:
    profiles, path, stations = build_ambiguous_hole_cluster_profiles()
    report = loft_plan_ambiguities(stations, samples=34, split_merge_mode="resolve")
    assert report.intervals
    selections = {
        entry.interval: entry.candidates[0].candidate_id
        for entry in report.intervals
        if entry.relationship_group and entry.relationship_group.startswith("hole_many_to_many:")
    }
    assert selections

    body = Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        samples=34,
        split_merge_mode="resolve",
        ambiguity_mode="interactive",
        ambiguity_selection=selections,
        cap_ends=True,
    )
    mesh = tessellate_surface_body(body, export_tessellation_request()).mesh

    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    assert np.isfinite(mesh.vertices).all()
    _assert_mesh_quality(mesh)


def test_hourglass_vessel_showcase_uses_interactive_selection_and_is_watertight() -> None:
    module = _load_real_world_example(
        "loft_hourglass_vessel_example",
        "docs/examples/loft/real_world/loft_hourglass_vessel_example.py",
    )

    stations = module.build_stations()
    selections = module.build_ambiguity_selection(stations)
    assert selections
    assert set(selections.values()) == {module.PILLAR_CANDIDATE_ID}

    body = module.build_surface_body()
    mesh = tessellate_surface_body(body, export_tessellation_request()).mesh

    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    assert (x_max - x_min) > 55.0
    assert (y_max - y_min) > 55.0
    assert (z_max - z_min) > 65.0
    _assert_mesh_quality(mesh)
