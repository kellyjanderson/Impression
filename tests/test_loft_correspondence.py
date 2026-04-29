from __future__ import annotations

import numpy as np
import pytest

from impression.mesh import Mesh, MeshSectionResult, section_mesh_with_plane
from impression.modeling import Loft, Section, as_section, export_tessellation_request, tessellate_surface_body
from tests.loft_showcases import (
    build_anchor_shift_rectangle_profiles,
    build_cylinder_correspondence_profiles,
    build_dual_cylinder_correspondence_profiles,
    build_perforated_cylinder_correspondence_profiles,
    build_phase_shift_cylinder_profiles,
    build_square_correspondence_profiles,
)


def _tessellated_loft_mesh(profiles: list[Section], path: np.ndarray) -> Mesh:
    progression = np.linspace(0.0, 1.0, len(profiles))
    stations = [(tuple(point), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) for point in np.asarray(path, dtype=float)]
    body = Loft(
        progression=progression,
        stations=stations,
        topology=profiles,
        cap_ends=True,
    )
    return tessellate_surface_body(body, export_tessellation_request()).mesh


def _slice_closed_loop_xy(mesh: Mesh, z_value: float) -> tuple[np.ndarray, MeshSectionResult]:
    result = section_mesh_with_plane(
        mesh,
        origin=(0.0, 0.0, float(z_value)),
        normal=(0.0, 0.0, 1.0),
        stitch_epsilon=1e-5,
    )
    closed = [polyline for polyline in result.polylines if polyline.closed]
    assert closed, "Expected a closed section loop."
    loop = max(closed, key=lambda polyline: _polygon_area(polyline.points[:, :2]))
    return loop.points[:, :2], result


def _polygon_area(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _polygon_perimeter(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    closed = np.vstack([points, points[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def _bbox(points_xy: np.ndarray) -> tuple[float, float, float, float]:
    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]))


def _roundness_ratio(points_xy: np.ndarray) -> float:
    centered = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    centered = centered - centered.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(centered, axis=1)
    mean_radius = float(radii.mean())
    if mean_radius == 0.0:
        return 0.0
    return float(radii.std() / mean_radius)


def _aspect_ratio(points_xy: np.ndarray) -> float:
    xmin, xmax, ymin, ymax = _bbox(points_xy)
    x_span = xmax - xmin
    y_span = ymax - ymin
    major = max(x_span, y_span)
    minor = min(x_span, y_span)
    if minor == 0.0:
        return float("inf")
    return float(major / minor)


def _principal_axis_angle_deg(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    if angle < -90.0:
        angle += 180.0
    if angle >= 90.0:
        angle -= 180.0
    return angle


def _expected_world_loop(section: Section, station_origin: np.ndarray) -> np.ndarray:
    region = section.regions[0]
    return np.asarray(region.outer.points, dtype=float) + np.asarray(station_origin, dtype=float)[:2]


def _expected_world_loops(section_like: Section | object, station_origin: np.ndarray) -> list[np.ndarray]:
    section = section_like if isinstance(section_like, Section) else as_section(section_like)
    offset = np.asarray(station_origin, dtype=float)[:2]
    loops: list[np.ndarray] = []
    for region in section.regions:
        loops.append(np.asarray(region.outer.points, dtype=float) + offset)
        for hole in region.holes:
            loops.append(np.asarray(hole.points, dtype=float) + offset)
    return loops


def _actual_closed_loops_xy(mesh: Mesh, z_value: float) -> tuple[list[np.ndarray], MeshSectionResult]:
    result = section_mesh_with_plane(
        mesh,
        origin=(0.0, 0.0, float(z_value)),
        normal=(0.0, 0.0, 1.0),
        stitch_epsilon=1e-5,
    )
    closed = [polyline.points[:, :2] for polyline in result.polylines if polyline.closed]
    assert closed, "Expected at least one closed section loop."
    return closed, result


def _loop_centroid(points_xy: np.ndarray) -> np.ndarray:
    return np.asarray(points_xy, dtype=float).reshape(-1, 2).mean(axis=0)


def _assert_station_slice_matches_expected_loops(
    mesh: Mesh,
    section_like: Section | object,
    station_origin: np.ndarray,
) -> None:
    actual_loops, result = _actual_closed_loops_xy(mesh, float(station_origin[2]))
    expected_loops = _expected_world_loops(section_like, station_origin)

    assert result.closed_count == len(expected_loops)
    assert result.polyline_count == len(expected_loops)

    unmatched = list(actual_loops)
    for expected_loop in expected_loops:
        expected_centroid = _loop_centroid(expected_loop)
        match_index = min(
            range(len(unmatched)),
            key=lambda idx: float(np.linalg.norm(_loop_centroid(unmatched[idx]) - expected_centroid)),
        )
        actual_loop = unmatched.pop(match_index)
        assert np.allclose(_loop_centroid(actual_loop), expected_centroid, atol=2e-2)
        assert np.allclose(_bbox(actual_loop), _bbox(expected_loop), atol=3e-2)
        assert _polygon_area(actual_loop) == pytest.approx(_polygon_area(expected_loop), rel=3e-2)
        assert _polygon_perimeter(actual_loop) == pytest.approx(_polygon_perimeter(expected_loop), rel=5e-2)


def _assert_station_slice_matches_input(
    mesh: Mesh,
    section: Section,
    station_origin: np.ndarray,
) -> None:
    actual_loop, result = _slice_closed_loop_xy(mesh, float(station_origin[2]))
    expected_loop = _expected_world_loop(section, station_origin)

    assert result.closed_count == 1
    assert result.polyline_count == 1
    assert np.allclose(actual_loop.mean(axis=0), expected_loop.mean(axis=0), atol=1e-3)
    assert np.allclose(_bbox(actual_loop), _bbox(expected_loop), atol=2e-3)
    assert _polygon_area(actual_loop) == pytest.approx(_polygon_area(expected_loop), rel=2e-3)
    assert _polygon_perimeter(actual_loop) == pytest.approx(_polygon_perimeter(expected_loop), rel=1e-2)
    if _aspect_ratio(expected_loop) > 1.2:
        assert _principal_axis_angle_deg(actual_loop) == pytest.approx(
            _principal_axis_angle_deg(expected_loop),
            abs=0.5,
        )


@pytest.mark.parametrize(
    ("builder", "fixture_name"),
    [
        (build_square_correspondence_profiles, "square"),
        (build_anchor_shift_rectangle_profiles, "anchor_shift_rectangle"),
    ],
)
def test_rectangular_station_slices_reconstruct_input_sections(builder, fixture_name: str) -> None:
    profiles, path = builder()
    mesh = _tessellated_loft_mesh(profiles, path)

    for station_index in range(1, len(profiles) - 1):
        _assert_station_slice_matches_input(mesh, profiles[station_index], path[station_index])


@pytest.mark.parametrize(
    ("builder", "fixture_name"),
    [
        (build_cylinder_correspondence_profiles, "cylinder"),
        (build_phase_shift_cylinder_profiles, "phase_shift_cylinder"),
    ],
)
def test_circular_station_slices_reconstruct_input_sections(builder, fixture_name: str) -> None:
    profiles, path = builder()
    mesh = _tessellated_loft_mesh(profiles, path)

    for station_index in range(1, len(profiles) - 1):
        _assert_station_slice_matches_input(mesh, profiles[station_index], path[station_index])


def test_dual_cylinder_station_slices_reconstruct_multi_region_topology() -> None:
    profiles, path = build_dual_cylinder_correspondence_profiles()
    mesh = _tessellated_loft_mesh(profiles, path)

    for station_index in range(1, len(profiles) - 1):
        _assert_station_slice_matches_expected_loops(mesh, profiles[station_index], path[station_index])


def test_perforated_cylinder_station_slices_reconstruct_hole_topology() -> None:
    profiles, path = build_perforated_cylinder_correspondence_profiles()
    mesh = _tessellated_loft_mesh([as_section(profile) for profile in profiles], path)

    for station_index in range(1, len(profiles) - 1):
        _assert_station_slice_matches_expected_loops(mesh, profiles[station_index], path[station_index])


@pytest.mark.parametrize(
    ("builder", "fixture_name"),
    [
        (build_square_correspondence_profiles, "square"),
        (build_anchor_shift_rectangle_profiles, "anchor_shift_rectangle"),
    ],
)
def test_rectangular_mid_slices_stay_rectangular(builder, fixture_name: str) -> None:
    profiles, path = builder()
    mesh = _tessellated_loft_mesh(profiles, path)

    for interval_index in range(len(path) - 1):
        midpoint_z = float(path[interval_index, 2] + path[interval_index + 1, 2]) / 2.0
        actual_loop, result = _slice_closed_loop_xy(mesh, midpoint_z)

        assert result.closed_count == 1
        assert result.polyline_count == 1
        assert _aspect_ratio(actual_loop) > 2.0
        assert _roundness_ratio(actual_loop) > 0.2


@pytest.mark.parametrize(
    ("builder", "fixture_name"),
    [
        (build_cylinder_correspondence_profiles, "cylinder"),
        (build_phase_shift_cylinder_profiles, "phase_shift_cylinder"),
    ],
)
def test_circular_mid_slices_stay_round(builder, fixture_name: str) -> None:
    profiles, path = builder()
    mesh = _tessellated_loft_mesh(profiles, path)

    for interval_index in range(len(path) - 1):
        midpoint_z = float(path[interval_index, 2] + path[interval_index + 1, 2]) / 2.0
        actual_loop, result = _slice_closed_loop_xy(mesh, midpoint_z)

        assert result.closed_count == 1
        assert result.polyline_count == 1
        assert _aspect_ratio(actual_loop) < 1.02
        assert _roundness_ratio(actual_loop) < 0.01
