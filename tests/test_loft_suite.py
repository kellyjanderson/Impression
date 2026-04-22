from __future__ import annotations

import numpy as np

from impression.modeling import Loft, Section, Station, SurfaceBody, as_section, tessellate_surface_body
from impression.modeling.drawing2d import make_circle, make_rect
from tests.loft_showcases import build_branching_manifold_profiles


def test_Loft_returns_surface_body_from_progression_points_and_topology() -> None:
    profiles, path = build_branching_manifold_profiles()
    progression = np.linspace(0.0, 1.0, len(profiles))

    body = Loft(progression=progression, stations=path, topology=profiles, cap_ends=True, split_merge_mode="resolve")

    assert isinstance(body, SurfaceBody)
    result = tessellate_surface_body(body)
    assert result.mesh.n_faces > 0


def test_Loft_accepts_explicit_station_frames() -> None:
    topology = [
        as_section(make_circle(radius=0.8)),
        as_section(make_rect(size=(1.2, 0.8))),
        as_section(make_circle(radius=0.6)),
    ]
    progression = [0.0, 0.4, 1.0]
    stations = [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.2, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.5, 0.2, 2.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ]

    body = Loft(progression=progression, stations=stations, topology=topology, cap_ends=True)

    assert isinstance(body, SurfaceBody)
    assert body.patch_count > 0


def test_Loft_accepts_station_objects_and_normalizes_progression() -> None:
    section_a = Section((as_section(make_rect(size=(1.0, 1.0))).regions[0],))
    section_b = Section((as_section(make_rect(size=(0.8, 1.2))).regions[0],))
    stations = [
        Station(t=0.0, section=section_a, origin=(0.0, 0.0, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=1.0, section=section_b, origin=(0.0, 0.0, 1.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
    ]

    body = Loft(progression=[0.25, 0.75], stations=stations, topology=[section_a, section_b])

    assert isinstance(body, SurfaceBody)
    assert body.shell_count == 1


def test_Loft_rejects_length_mismatch() -> None:
    try:
        Loft(
            progression=[0.0, 1.0],
            stations=[(0.0, 0.0, 0.0)],
            topology=[as_section(make_rect(size=(1.0, 1.0))), as_section(make_rect(size=(1.0, 1.0)))],
        )
    except ValueError as exc:
        assert "length mismatch" in str(exc)
    else:
        raise AssertionError("Loft should reject mismatched progression/station/topology lengths.")
