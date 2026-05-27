from __future__ import annotations

import numpy as np

from impression.modeling import (
    Section,
    Station,
    SurfaceBody,
    as_section,
    export_tessellation_request,
    loft,
    loft_plan_ambiguities,
    loft_profiles,
    tessellate_surface_body,
)
from impression.modeling.drawing2d import make_rect


def _assert_surface_body_tessellates(body: SurfaceBody) -> None:
    assert isinstance(body, SurfaceBody)
    assert body.patch_count > 0
    result = tessellate_surface_body(body, export_tessellation_request(require_watertight=False))
    assert result.mesh.n_vertices > 0
    assert result.mesh.n_faces > 0


def test_public_loft_api_passes_interactive_selection() -> None:
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

    body = loft(
        [s0, s1],
        path=path,
        samples=30,
        split_merge_mode="resolve",
        ambiguity_mode="interactive",
        ambiguity_selection=selection,
        cap_ends=True,
    )
    _assert_surface_body_tessellates(body)


def test_public_loft_profiles_api_passes_probabilistic_controls_reproducibly() -> None:
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

    body_a = loft_profiles(
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
    body_b = loft_profiles(
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

    assert body_a.stable_identity == body_b.stable_identity
    _assert_surface_body_tessellates(body_a)


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

    body = loft(
        [s0, s1],
        path=path,
        samples=30,
        split_merge_mode="resolve",
        ambiguity_mode="interactive",
        ambiguity_selection_policy="best_effort",
        cap_ends=True,
    )
    _assert_surface_body_tessellates(body)
