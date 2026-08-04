from __future__ import annotations

import numpy as np
import pytest

from impression.modeling import (
    Loft,
    Station,
    TopologyPath,
    export_tessellation_request,
    preview_tessellation_request,
    tessellate_surface_body,
)


def _path(points: tuple[tuple[str, tuple[float, float]], ...]) -> TopologyPath:
    builder = TopologyPath.closed()
    for name, coordinates in points:
        builder = builder.point(name, coordinates, correspond=name)
    return builder.build()


def _station(z: float, height: float) -> Station:
    return Station(
        t=z / height,
        section=None,
        origin=(0.0, 0.0, z),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
    )


@pytest.mark.parametrize("samples", [8, 17, 64])
@pytest.mark.parametrize("request_factory", [preview_tessellation_request, export_tessellation_request])
def test_protected_diagonal_vertices_are_stable_across_sampling_policies(samples, request_factory) -> None:
    southeast = _path(
        (
            ("diagonal-negative-corner", (-16.0, -16.0)),
            ("negative-y-positive-corner", (16.0, -16.0)),
            ("diagonal-positive-corner", (16.0, 16.0)),
        )
    )
    northwest = _path(
        (
            ("diagonal-negative-corner", (-16.0, -16.0)),
            ("diagonal-positive-corner", (16.0, 16.0)),
            ("positive-y-negative-corner", (-16.0, 16.0)),
        )
    )
    stations = (_station(0.0, 2.0), _station(2.0, 2.0))

    results = []
    for topology in (southeast, northwest):
        body = Loft(
            (0.0, 1.0),
            stations,
            (topology, topology),
            samples=samples,
            cap_ends=True,
            fairness_mode="off",
            fairness_weight=0.0,
        )
        results.append(tessellate_surface_body(body, request_factory()))

    for result in results:
        for z in (0.0, 2.0):
            for point in ((-16.0, -16.0, z), (16.0, 16.0, z)):
                distance = np.linalg.norm(result.mesh.vertices - np.asarray(point), axis=1).min()
                assert distance <= 1e-9
        assert result.mesh.bounds == (-16.0, 16.0, -16.0, 16.0, 0.0, 2.0)
        assert result.analysis.is_watertight
        assert result.analysis.degenerate_faces == 0
        assert result.analysis.nonmanifold_edges == 0


def _audio_cube_half(cap_points, wall_points):
    cap = _path(cap_points)
    walls = _path(wall_points)
    levels = ((0.0, cap), (1.76, cap), (1.8, walls), (30.2, walls), (30.24, cap), (32.0, cap))
    stations = tuple(_station(z, 32.0) for z, _path_value in levels)
    return Loft(
        tuple(station.t for station in stations),
        stations,
        tuple(path for _z, path in levels),
        samples=64,
        cap_ends=True,
        split_merge_mode="resolve",
        ambiguity_mode="auto",
        fairness_mode="off",
        fairness_weight=0.0,
    )


def test_full_audio_cube_diagonal_halves_keep_shared_corner_and_mesh_integrity() -> None:
    southeast = _audio_cube_half(
        (
            ("diagonal-negative-corner", (-16.0, -16.0)),
            ("negative-y-positive-corner", (16.0, -16.0)),
            ("diagonal-positive-corner", (16.0, 16.0)),
            ("positive-snap-seam-upper", (9.0, 9.0)),
            ("positive-snap-head-upper", (7.7, 9.1)),
            ("positive-snap-head-lower", (2.9, 4.3)),
            ("positive-snap-seam-lower", (3.0, 3.0)),
            ("negative-snap-seam-upper", (-3.0, -3.0)),
            ("negative-snap-head-upper", (-4.3, -2.9)),
            ("negative-snap-head-lower", (-9.1, -7.7)),
            ("negative-snap-seam-lower", (-9.0, -9.0)),
        ),
        (
            ("diagonal-negative-corner", (-16.0, -16.0)),
            ("negative-y-positive-corner", (16.0, -16.0)),
            ("diagonal-positive-corner", (16.0, 16.0)),
            ("inner-diagonal-positive-corner", (14.2, 14.2)),
            ("inner-positive-x-negative-corner", (14.2, -14.2)),
            ("inner-diagonal-negative-corner", (-14.2, -14.2)),
        ),
    )
    northwest = _audio_cube_half(
        (
            ("diagonal-negative-corner", (-16.0, -16.0)),
            ("negative-snap-pocket-lower", (-9.15, -9.15)),
            ("negative-snap-pocket-head-lower", (-9.38, -7.62)),
            ("negative-snap-pocket-head-upper", (-4.38, -2.62)),
            ("negative-snap-pocket-upper", (-2.85, -2.85)),
            ("positive-snap-pocket-lower", (2.85, 2.85)),
            ("positive-snap-pocket-head-lower", (2.62, 4.38)),
            ("positive-snap-pocket-head-upper", (7.62, 9.38)),
            ("positive-snap-pocket-upper", (9.15, 9.15)),
            ("diagonal-positive-corner", (16.0, 16.0)),
            ("positive-y-negative-corner", (-16.0, 16.0)),
        ),
        (
            ("diagonal-negative-corner", (-16.0, -16.0)),
            ("inner-diagonal-negative-corner", (-14.2, -14.2)),
            ("inner-negative-x-positive-corner", (-14.2, 14.2)),
            ("inner-diagonal-positive-corner", (14.2, 14.2)),
            ("diagonal-positive-corner", (16.0, 16.0)),
            ("positive-y-negative-corner", (-16.0, 16.0)),
        ),
    )

    for body in (southeast, northwest):
        result = tessellate_surface_body(body, export_tessellation_request())
        assert result.mesh.bounds == (-16.0, 16.0, -16.0, 16.0, 0.0, 32.0)
        assert result.analysis.is_watertight
        assert result.analysis.degenerate_faces == 0
        for z in (0.0, 32.0):
            distance = np.linalg.norm(result.mesh.vertices - np.asarray((-16.0, -16.0, z)), axis=1).min()
            assert distance <= 1e-9


def test_unprotected_path_can_increase_sampling_density() -> None:
    path = TopologyPath.from_points(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
    stations = (_station(0.0, 1.0), _station(1.0, 1.0))
    vertex_counts = []
    for samples in (8, 24):
        body = Loft((0.0, 1.0), stations, (path, path), samples=samples, cap_ends=True)
        vertex_counts.append(tessellate_surface_body(body, preview_tessellation_request()).mesh.n_vertices)
    assert vertex_counts[1] > vertex_counts[0]
