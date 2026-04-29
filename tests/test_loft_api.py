from __future__ import annotations

import numpy as np

from impression.mesh import analyze_mesh
from impression.modeling import Section, Station, as_section, loft, loft_plan_ambiguities, loft_profiles
from impression.modeling.drawing2d import make_rect


def _assert_mesh_quality(mesh) -> None:
    analysis = analyze_mesh(mesh)
    assert analysis.boundary_edges == 0, analysis.issues()
    assert analysis.nonmanifold_edges == 0, analysis.issues()
    assert analysis.degenerate_faces == 0, analysis.issues()


def test_public_loft_api_threads_interactive_selection() -> None:
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
    path = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    stations = [
        Station(t=0.0, section=s0, origin=path[0], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
        Station(t=1.0, section=s1, origin=path[1], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1]),
    ]
    report = loft_plan_ambiguities(stations, samples=30, split_merge_mode="resolve")
    selection = {(0, 1): report.intervals[0].candidates[0].candidate_id}

    mesh = loft(
        [s0, s1],
        path=path,
        samples=30,
        split_merge_mode="resolve",
        ambiguity_mode="interactive",
        ambiguity_selection=selection,
        cap_ends=True,
    )
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)


def test_public_loft_profiles_api_threads_probabilistic_controls_reproducibly() -> None:
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
    path = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    mesh_a = loft_profiles(
        [s0, s1],
        path=path,
        samples=30,
        split_merge_mode="resolve",
        ambiguity_mode="auto",
        disambiguation_mode="probabilistic",
        disambiguation_seed=17,
        probabilistic_trials=40,
        probabilistic_temperature=0.3,
        cap_ends=True,
    )
    mesh_b = loft_profiles(
        [s0, s1],
        path=path,
        samples=30,
        split_merge_mode="resolve",
        ambiguity_mode="auto",
        disambiguation_mode="probabilistic",
        disambiguation_seed=17,
        probabilistic_trials=40,
        probabilistic_temperature=0.3,
        cap_ends=True,
    )

    assert np.array_equal(mesh_a.vertices, mesh_b.vertices)
    assert np.array_equal(mesh_a.faces, mesh_b.faces)
    _assert_mesh_quality(mesh_a)


def test_public_loft_api_interactive_best_effort_falls_back_without_selection() -> None:
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
    path = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    mesh = loft(
        [s0, s1],
        path=path,
        samples=30,
        split_merge_mode="resolve",
        ambiguity_mode="interactive",
        ambiguity_selection_policy="best_effort",
        cap_ends=True,
    )
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    _assert_mesh_quality(mesh)
