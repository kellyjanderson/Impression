from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from impression.mesh import analyze_mesh
from impression.modeling import (
    Loft,
    Loop,
    PlanarSurfacePatch,
    Region,
    Section,
    Station,
    export_tessellation_request,
    tessellate_surface_body,
)


@dataclass(frozen=True)
class _Opening:
    u_center: float
    z_center: float
    width: float
    height: float


_ORIGINAL_AUDIO_CUBE_OPENINGS = (
    _Opening(u_center=-7.0, z_center=27.2, width=3.4, height=1.8),
    _Opening(u_center=7.0, z_center=27.2, width=3.4, height=1.8),
    _Opening(u_center=0.0, z_center=9.512, width=3.0, height=3.0),
)


def _rectangle(*, center: tuple[float, float], size: tuple[float, float]) -> Loop:
    cx, cy = center
    width, height = size
    return Loop(
        np.asarray(
            (
                (cx - width / 2.0, cy - height / 2.0),
                (cx + width / 2.0, cy - height / 2.0),
                (cx + width / 2.0, cy + height / 2.0),
                (cx - width / 2.0, cy + height / 2.0),
            ),
            dtype=float,
        )
    )


def _wall_section(openings: tuple[_Opening, ...]) -> Section:
    return Section(
        regions=(
            Region(
                outer=_rectangle(center=(0.0, 15.2), size=(32.0, 30.4)),
                holes=tuple(
                    _rectangle(
                        center=(opening.u_center, opening.z_center),
                        size=(opening.width, opening.height),
                    )
                    for opening in openings
                ),
            ),
        )
    )


def _direct_wall_loft(openings: tuple[_Opening, ...]):
    section = _wall_section(openings)
    stations = tuple(
        Station(
            t=float(index),
            section=section,
            origin=(0.0, y, 0.0),
            u=(-1.0, 0.0, 0.0),
            v=(0.0, 0.0, 1.0),
            n=(0.0, 1.0, 0.0),
        )
        for index, y in enumerate((14.4, 16.0))
    )
    return Loft(
        progression=(0.0, 1.0),
        stations=stations,
        topology=(section, section),
        samples=64,
        cap_ends=True,
    )


def _signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray, *, epsilon: float = 1e-9) -> bool:
    def cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    values = tuple(cross(triangle[index], triangle[(index + 1) % 3], point) for index in range(3))
    return all(value >= -epsilon for value in values) or all(value <= epsilon for value in values)


def _opening_witnesses(opening: _Opening) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(
            (
                opening.u_center + u_offset * opening.width,
                opening.z_center + z_offset * opening.height,
            ),
            dtype=float,
        )
        for u_offset in (-0.25, 0.0, 0.25)
        for z_offset in (-0.25, 0.0, 0.25)
    )


@pytest.mark.parametrize(
    "openings",
    (
        (),
        _ORIGINAL_AUDIO_CUBE_OPENINGS[:1],
        _ORIGINAL_AUDIO_CUBE_OPENINGS[:2],
        _ORIGINAL_AUDIO_CUBE_OPENINGS,
    ),
    ids=("solid", "one-opening", "two-openings", "original-audio-cube-wall"),
)
def test_direct_multi_opening_wall_preserves_trims_and_emits_clean_mesh(openings: tuple[_Opening, ...]) -> None:
    body = _direct_wall_loft(openings)

    cap_patches = tuple(patch for patch in body.shells[0].patches if isinstance(patch, PlanarSurfacePatch))
    assert len(cap_patches) == 2
    for cap in cap_patches:
        assert [trim.category for trim in cap.trim_loops] == ["outer", *(["inner"] * len(openings))]
        if cap.outer_trim is not None:
            assert _signed_area(cap.outer_trim.normalized().points_uv) > 0.0
        assert all(_signed_area(trim.normalized().points_uv) < 0.0 for trim in cap.inner_trims)

    mesh = tessellate_surface_body(body, export_tessellation_request()).mesh
    analysis = analyze_mesh(mesh)
    assert analysis.degenerate_faces == 0, analysis.issues()
    assert analysis.boundary_edges == 0, analysis.issues()
    assert analysis.nonmanifold_edges == 0, analysis.issues()
    assert analysis.invalid_vertices == 0, analysis.issues()
    assert np.all(np.isfinite(mesh.vertices))
    normals = np.cross(
        mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]],
        mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 0]],
    )
    assert np.all(np.isfinite(normals))

    cap_face_mask = np.ptp(mesh.vertices[mesh.faces, 1], axis=1) <= 1e-9
    cap_triangles_world = mesh.vertices[mesh.faces[cap_face_mask]]
    cap_triangles_local = np.stack((-cap_triangles_world[:, :, 0], cap_triangles_world[:, :, 2]), axis=2)
    for opening in openings:
        for witness in _opening_witnesses(opening):
            assert not any(_point_in_triangle(witness, triangle) for triangle in cap_triangles_local)


@pytest.mark.parametrize(
    ("openings", "diagnostic"),
    (
        (
            (_Opening(0.0, 10.0, 6.0, 6.0), _Opening(2.0, 10.0, 6.0, 6.0)),
            "overlapping or nested holes",
        ),
        (
            (_Opening(0.0, 10.0, 8.0, 8.0), _Opening(0.0, 10.0, 2.0, 2.0)),
            "overlapping or nested holes",
        ),
    ),
    ids=("overlapping", "nested"),
)
def test_invalid_multi_opening_wall_refuses_before_geometry_emission(
    openings: tuple[_Opening, ...], diagnostic: str
) -> None:
    with pytest.raises(ValueError, match=diagnostic):
        _direct_wall_loft(openings)

    assert not _wall_section(openings).regions[0].is_valid()
