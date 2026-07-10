from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from impression.modeling import (
    Bezier3D,
    Loft,
    Path3D,
    Section,
    Station,
    SurfaceBody,
    as_section,
    boolean_difference,
    boolean_intersection,
    boolean_union,
    make_box,
    make_cone,
    make_cylinder,
    make_ngon,
    make_prism,
    make_surface_body,
    make_sphere,
    make_torus,
    loft,
    loft_sections,
    rotate,
    scale,
    translate,
)
from impression.modeling.drawing2d import make_circle, make_polygon, make_rect
from tests.loft_showcases import (
    build_branching_manifold_profiles,
    build_dual_cylinder_correspondence_profiles,
    build_notched_cylinder_correspondence_profiles,
    build_notched_phase_shift_cylinder_profiles,
    build_perforated_cylinder_correspondence_profiles,
    build_perforated_vessel_profiles,
)
from tests.reference_images import ensure_reference_stl, stl_signal_stats, write_surface_body_stl


_RELEASE_STL_ROOT = Path(__file__).resolve().parents[1] / "project/release-0.1.0a/reference-stl"


def _assert_stl_signal(path: Path, *, min_facets: int, min_vertices: int = 0) -> None:
    stats = stl_signal_stats(path)
    assert stats["facet_count"] >= min_facets
    assert stats["vertex_count"] >= min_vertices
    assert stats["file_size"] > 256


def _ensure_stl_reference(
    *,
    model: SurfaceBody,
    fixture_name: str,
    tmp_path: Path,
    update_dirty_reference_images: bool,
    min_facets: int,
    min_vertices: int = 0,
) -> None:
    stl_path = tmp_path / f"{fixture_name.replace('/', '-')}.stl"
    write_surface_body_stl(model, stl_path)
    _assert_stl_signal(stl_path, min_facets=min_facets, min_vertices=min_vertices)
    reference = ensure_reference_stl(
        rendered_path=stl_path,
        reference_root=_RELEASE_STL_ROOT,
        name=fixture_name,
        update_dirty_reference_images=update_dirty_reference_images,
    )
    assert reference.exists()


def _surface_boolean_body(result: object) -> SurfaceBody:
    assert getattr(result, "status", None) == "succeeded"
    body = getattr(result, "body", None)
    assert isinstance(body, SurfaceBody)
    return body


def _loft_from_profiles(builder: Callable[[], tuple[list[object], np.ndarray]]) -> SurfaceBody:
    profiles, path = builder()
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=True,
        split_merge_mode="resolve",
    )


def _combine_surface_bodies(bodies: list[SurfaceBody]) -> SurfaceBody:
    return make_surface_body(tuple(shell for body in bodies for shell in body.iter_shells(world=True)))


def _rounded_rectangle_profile(width: float, height: float, radius: float, *, samples_per_corner: int = 8):
    points: list[tuple[float, float]] = []
    corners = (
        (width / 2.0 - radius, height / 2.0 - radius, 0.0, np.pi / 2.0),
        (-width / 2.0 + radius, height / 2.0 - radius, np.pi / 2.0, np.pi),
        (-width / 2.0 + radius, -height / 2.0 + radius, np.pi, 3.0 * np.pi / 2.0),
        (width / 2.0 - radius, -height / 2.0 + radius, 3.0 * np.pi / 2.0, 2.0 * np.pi),
    )
    for cx, cy, start, stop in corners:
        for angle in np.linspace(start, stop, samples_per_corner, endpoint=False):
            points.append((cx + radius * float(np.cos(angle)), cy + radius * float(np.sin(angle))))
    return make_polygon(points)


def _star_profile(outer_radius: float, inner_radius: float, points: int = 5):
    coords: list[tuple[float, float]] = []
    for index in range(points * 2):
        radius = outer_radius if index % 2 == 0 else inner_radius
        angle = (np.pi / 2.0) + index * np.pi / points
        coords.append((radius * float(np.cos(angle)), radius * float(np.sin(angle))))
    return make_polygon(coords)


def _transformed_primitive_group() -> SurfaceBody:
    box = rotate(make_box(size=(0.75, 0.45, 0.35), color="#d9822b"), axis=(0.0, 0.0, 1.0), angle_deg=28.0)
    box = translate(box, (-0.55, -0.08, 0.0))
    cylinder = rotate(make_cylinder(radius=0.18, height=0.95, color="#5b84b1"), axis=(1.0, 0.0, 0.0), angle_deg=72.0)
    cylinder = translate(cylinder, (0.38, 0.22, 0.08))
    frustum = scale(make_cone(bottom_diameter=0.46, top_diameter=0.22, height=0.62, color="#72a276"), (1.0, 0.7, 1.2))
    frustum = translate(frustum, (0.08, -0.42, 0.05))
    return _combine_surface_bodies([box, cylinder, frustum])


def _mixed_scale_primitives() -> SurfaceBody:
    tiny = translate(make_box(size=(0.04, 0.04, 0.04), color="#ffcf5a"), (-0.72, -0.42, 0.0))
    slender = translate(make_cylinder(radius=0.035, height=1.4, color="#8f6bb8"), (0.0, 0.0, 0.0))
    plate = translate(make_box(size=(1.5, 0.08, 0.18), color="#4f9d9a"), (0.15, 0.35, 0.0))
    large = translate(make_sphere(radius=0.28, color="#d95d39"), (0.72, -0.2, 0.0))
    return _combine_surface_bodies([tiny, slender, plate, large])


def _thin_stable_primitives() -> SurfaceBody:
    wafer = translate(make_box(size=(1.1, 0.7, 0.025), color="#a8c686"), (0.0, 0.0, -0.05))
    pin = translate(make_cylinder(radius=0.025, height=0.62, color="#f2a65a"), (-0.38, 0.0, 0.0))
    needle = rotate(make_cone(bottom_diameter=0.08, top_diameter=0.01, height=0.85, color="#6c91bf"), axis=(0.0, 1.0, 0.0), angle_deg=86.0)
    needle = translate(needle, (0.32, 0.0, 0.02))
    return _combine_surface_bodies([wafer, pin, needle])


def _authored_color_primitive_smoke() -> SurfaceBody:
    red = translate(make_box(size=(0.42, 0.42, 0.42), color="#d95d39"), (-0.48, 0.0, 0.0))
    blue = translate(make_sphere(radius=0.28, color="#5b84b1"), (0.0, 0.0, 0.0))
    green = translate(make_cylinder(radius=0.22, height=0.48, color="#72a276"), (0.48, 0.0, 0.0))
    return _combine_surface_bodies([red, blue, green])


def _circle_to_circle_loft() -> SurfaceBody:
    profiles = [
        make_circle(radius=0.48),
        make_circle(radius=0.62, center=(0.04, -0.03)),
        make_circle(radius=0.54, center=(-0.03, 0.02)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.08, 0.03, 0.9), (0.16, -0.02, 1.8)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _square_to_square_loft() -> SurfaceBody:
    profiles = [
        make_rect(size=(0.9, 0.9)),
        make_rect(size=(1.1, 0.7), center=(0.04, -0.02)),
        make_rect(size=(0.8, 1.0), center=(-0.02, 0.03)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.1, -0.04, 0.85), (0.18, 0.02, 1.7)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _circle_to_square_loft() -> SurfaceBody:
    profiles = [
        make_circle(radius=0.55),
        make_polygon([(-0.55, -0.35), (0.55, -0.35), (0.55, 0.35), (-0.55, 0.35)]),
        make_rect(size=(0.9, 0.9), center=(0.03, -0.02)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.06, 0.03, 0.85), (0.12, -0.02, 1.7)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _triangle_to_hexagon_loft() -> SurfaceBody:
    triangle = make_polygon([(0.0, 0.62), (-0.58, -0.36), (0.58, -0.36)])
    hexagon_points = [
        (0.62 * float(np.cos(angle)), 0.62 * float(np.sin(angle)))
        for angle in np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    ]
    profiles = [triangle, make_polygon(hexagon_points), make_rect(size=(0.75, 0.55))]
    path = np.asarray([(0.0, 0.0, 0.0), (0.04, 0.04, 0.78), (0.1, -0.03, 1.65)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _tapered_circle_loft() -> SurfaceBody:
    profiles = [make_circle(radius=0.72), make_circle(radius=0.48), make_circle(radius=0.28)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.03, 0.0, 0.78), (0.04, 0.0, 1.65)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _bulb_circle_loft() -> SurfaceBody:
    profiles = [
        make_circle(radius=0.34),
        make_circle(radius=0.78, center=(0.03, 0.0)),
        make_circle(radius=0.42, center=(-0.02, 0.02)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.03, 0.75), (0.05, -0.03, 1.6)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _s_curve_station_loft() -> SurfaceBody:
    profiles = [
        make_circle(radius=0.38),
        make_circle(radius=0.48, center=(0.02, -0.01)),
        make_circle(radius=0.44, center=(-0.03, 0.02)),
        make_circle(radius=0.36),
    ]
    path = np.asarray(
        [(0.0, 0.0, 0.0), (0.32, 0.16, 0.58), (-0.28, -0.14, 1.38), (0.18, 0.08, 2.25)],
        dtype=float,
    )
    return _loft_from_profiles(lambda: (profiles, path))


def _rectangle_to_rounded_rectangle_loft() -> SurfaceBody:
    profiles = [
        make_rect(size=(1.0, 0.62)),
        _rounded_rectangle_profile(1.06, 0.72, 0.12),
        _rounded_rectangle_profile(0.84, 0.58, 0.18),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.06, 0.02, 0.82), (0.1, -0.02, 1.68)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _open_ended_loft() -> SurfaceBody:
    profiles = [make_circle(radius=0.42), make_circle(radius=0.58), make_circle(radius=0.38)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.04, 0.06, 0.78), (0.08, -0.04, 1.62)], dtype=float)
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=False,
        split_merge_mode="resolve",
    )


def _capped_circle_planar_caps_loft() -> SurfaceBody:
    profiles = [make_circle(radius=0.46), make_circle(radius=0.5), make_circle(radius=0.42)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.01, 0.7), (0.04, -0.01, 1.4)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _high_twist_loft() -> SurfaceBody:
    profiles = []
    for angle in (0.0, 45.0, 105.0, 170.0):
        radians = np.deg2rad(angle)
        base = [(-0.45, -0.28), (0.45, -0.28), (0.45, 0.28), (-0.45, 0.28)]
        rotated = [
            (
                x * float(np.cos(radians)) - y * float(np.sin(radians)),
                x * float(np.sin(radians)) + y * float(np.cos(radians)),
            )
            for x, y in base
        ]
        profiles.append(make_polygon(rotated))
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.02, 0.55), (0.0, -0.04, 1.12), (0.05, 0.0, 1.72)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _smooth_profile_sequence_loft() -> SurfaceBody:
    profiles = [
        make_circle(radius=0.44),
        _rounded_rectangle_profile(0.88, 0.72, 0.28),
        make_circle(radius=0.5, center=(0.02, -0.02)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.08, 0.02, 0.82), (0.12, -0.02, 1.6)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _mixed_sharp_smooth_loft() -> SurfaceBody:
    profiles = [
        _star_profile(0.48, 0.26),
        _rounded_rectangle_profile(0.86, 0.64, 0.18),
        make_circle(radius=0.46),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.04, 0.04, 0.74), (0.1, -0.02, 1.52)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _helical_station_loft() -> SurfaceBody:
    profiles = [make_circle(radius=0.2 + 0.03 * (index % 2)) for index in range(6)]
    angles = np.linspace(0.0, 1.5 * np.pi, len(profiles))
    path = np.asarray(
        [(0.34 * float(np.cos(angle)), 0.34 * float(np.sin(angle)), 0.34 * index) for index, angle in enumerate(angles)],
        dtype=float,
    )
    return _loft_from_profiles(lambda: (profiles, path))


def _very_short_span_loft() -> SurfaceBody:
    profiles = [make_rect(size=(0.72, 0.5)), make_rect(size=(0.64, 0.44), center=(0.02, -0.01))]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.0, 0.08)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _nonuniform_station_spacing_loft() -> SurfaceBody:
    profiles = [
        make_circle(radius=0.34),
        make_circle(radius=0.48),
        make_circle(radius=0.52),
        make_circle(radius=0.38),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.01, 0.12), (0.08, -0.03, 1.35), (0.1, 0.0, 1.62)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _branching_manifold_open_loft() -> SurfaceBody:
    profiles, path = build_branching_manifold_profiles()
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=False,
        split_merge_mode="resolve",
    )


def _sharp_corner_star_loft() -> SurfaceBody:
    profiles = [
        _star_profile(0.55, 0.28),
        _star_profile(0.48, 0.22),
        _star_profile(0.6, 0.32),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.03, -0.03, 0.76), (0.0, 0.02, 1.5)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _tiny_to_normal_profile_loft() -> SurfaceBody:
    profiles = [make_circle(radius=0.08), make_circle(radius=0.28), make_circle(radius=0.62)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.0, 0.55), (0.04, 0.0, 1.55)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _hole_to_solid_transition_loft() -> SurfaceBody:
    from tests.loft_showcases import make_perforated_disc

    profiles = [
        make_perforated_disc(outer_radius=0.72, holes=[(0.14, (0.0, 0.0))]),
        make_circle(radius=0.64),
        make_circle(radius=0.56),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.03, 0.02, 0.72), (0.08, -0.02, 1.5)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _explicit_station_frames_loft() -> SurfaceBody:
    profiles = [as_section(make_circle(radius=0.46)), as_section(make_rect(size=(0.88, 0.58))), as_section(make_circle(radius=0.38))]
    half_root = float(np.sqrt(0.5))
    cos_30 = float(np.sqrt(3.0) / 2.0)
    stations = [
        Station(t=0.0, section=profiles[0], origin=(0.0, 0.0, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=0.45, section=profiles[1], origin=(0.2, 0.0, 0.8), u=(cos_30, 0.5, 0.0), v=(-0.5, cos_30, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=1.0, section=profiles[2], origin=(0.44, 0.18, 1.65), u=(half_root, half_root, 0.0), v=(-half_root, half_root, 0.0), n=(0.0, 0.0, 1.0)),
    ]
    return Loft(progression=[0.0, 0.45, 1.0], stations=stations, topology=profiles, cap_ends=True)


def _tight_curvature_frame_transport_loft() -> SurfaceBody:
    profiles = [make_circle(radius=0.18 + 0.025 * (index % 2)) for index in range(7)]
    angles = np.linspace(0.0, 1.75 * np.pi, len(profiles))
    path = np.asarray(
        [(0.28 * float(np.cos(angle)), 0.28 * float(np.sin(angle)), 0.18 * index) for index, angle in enumerate(angles)],
        dtype=float,
    )
    return _loft_from_profiles(lambda: (profiles, path))


def _branch_merge_many_to_one_loft() -> SurfaceBody:
    profiles, path = build_branching_manifold_profiles()
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path[::-1].copy(),
        topology=list(reversed(profiles)),
        cap_ends=True,
        split_merge_mode="resolve",
    )


def _asymmetric_branch_lengths_loft() -> SurfaceBody:
    trunk = Section((as_section(make_circle(radius=0.58)).regions[0],))
    split = Section((as_section(make_circle(radius=0.34, center=(-0.55, 0.0))).regions[0], as_section(make_circle(radius=0.3, center=(0.55, 0.0))).regions[0]))
    stretched = Section((as_section(make_circle(radius=0.28, center=(-1.18, 0.38))).regions[0], as_section(make_circle(radius=0.34, center=(0.72, -0.1))).regions[0]))
    profiles = [trunk, split, stretched]
    path = np.asarray([(0.0, 0.0, 0.0), (0.08, 0.04, 0.9), (0.38, 0.18, 2.1)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _near_coincident_station_loft() -> SurfaceBody:
    profiles = [make_circle(radius=0.3), make_circle(radius=0.32), make_circle(radius=0.5)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.002, 0.0, 0.015), (0.04, 0.0, 1.05)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def _curved_bezier_path_loft() -> SurfaceBody:
    path = Path3D(
        [
            Bezier3D(
                np.asarray([0.0, 0.0, 0.0], dtype=float),
                np.asarray([0.45, 0.2, 0.55], dtype=float),
                np.asarray([-0.35, 0.35, 1.25], dtype=float),
                np.asarray([0.2, -0.1, 1.9], dtype=float),
            )
        ]
    )
    profiles = [make_circle(radius=0.34), make_circle(radius=0.48), make_circle(radius=0.32)]
    return loft(profiles, path=path, samples=48, bezier_samples=48, cap_ends=True)


def _many_to_many_region_expand_loft() -> SurfaceBody:
    left = make_rect(size=(0.85, 0.85), center=(-1.0, 0.0))
    right = make_rect(size=(0.85, 0.85), center=(1.0, 0.0))
    left_end = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    center_end = make_rect(size=(0.7, 0.7), center=(0.0, 0.0))
    right_end = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    start = Section((as_section(left).regions[0], as_section(right).regions[0]))
    end = Section(
        (
            as_section(left_end).regions[0],
            as_section(center_end).regions[0],
            as_section(right_end).regions[0],
        )
    )
    stations = [
        Station(t=0.0, section=start, origin=(0.0, 0.0, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=1.0, section=end, origin=(0.0, 0.0, 1.5), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
    ]
    return loft_sections(stations, samples=30, cap_ends=True, split_merge_mode="resolve", split_merge_steps=8)


def _many_to_many_region_collapse_loft() -> SurfaceBody:
    left = make_rect(size=(0.8, 0.8), center=(-1.0, 0.0))
    center = make_rect(size=(0.7, 0.7), center=(0.0, 0.0))
    right = make_rect(size=(0.8, 0.8), center=(1.0, 0.0))
    left_end = make_rect(size=(0.85, 0.85), center=(-1.0, 0.0))
    right_end = make_rect(size=(0.85, 0.85), center=(1.0, 0.0))
    start = Section(
        (
            as_section(left).regions[0],
            as_section(center).regions[0],
            as_section(right).regions[0],
        )
    )
    end = Section((as_section(left_end).regions[0], as_section(right_end).regions[0]))
    stations = [
        Station(t=0.0, section=start, origin=(0.0, 0.0, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=1.0, section=end, origin=(0.0, 0.0, 1.5), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
    ]
    return loft_sections(stations, samples=30, cap_ends=True, split_merge_mode="resolve", split_merge_steps=8)


def _deterministic_resampling_loft() -> SurfaceBody:
    circle_points = [
        (0.5 * float(np.cos(angle)), 0.5 * float(np.sin(angle)))
        for angle in np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False)
    ]
    profiles = [make_polygon(circle_points), make_rect(size=(0.75, 0.75)), make_circle(radius=0.42)]
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.1, 0.05, 0.8), (0.16, -0.02, 1.6)])
    return loft(profiles, path=path, samples=44, cap_ends=True)


def _domed_tapered_caps_loft() -> SurfaceBody:
    profiles = [make_circle(radius=0.44), make_circle(radius=0.56), make_circle(radius=0.38)]
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.08, 0.02, 0.9), (0.12, -0.02, 1.8)])
    return loft(
        profiles,
        path=path,
        samples=40,
        start_cap="dome",
        end_cap="taper",
        start_cap_length=0.5,
        end_cap_length=0.45,
        cap_scale_dims="both",
    )


def _slope_dome_caps_loft() -> SurfaceBody:
    profiles = [make_rect(size=(0.8, 0.55)), make_rect(size=(0.95, 0.75)), make_rect(size=(0.58, 0.45))]
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.05, 0.06, 0.8), (0.0, 0.0, 1.55)])
    return loft(
        profiles,
        path=path,
        samples=36,
        start_cap="slope",
        end_cap="dome",
        start_cap_length=0.38,
        end_cap_length=0.5,
        cap_scale_dims="smallest",
    )


def _reversed_winding_repair_loft() -> SurfaceBody:
    square = [(-0.45, -0.45), (-0.45, 0.45), (0.45, 0.45), (0.45, -0.45)]
    profiles = [make_polygon(square), make_polygon(list(reversed(square))), make_polygon(square)]
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.02, 0.0, 0.7), (0.05, 0.0, 1.4)])
    return loft(profiles, path=path, samples=32, cap_ends=True)


def _surface_box_union_overlap() -> SurfaceBody:
    base = make_box(size=(1.4, 1.0, 0.8))
    post = make_box(size=(0.7, 0.55, 0.9), center=(0.55, 0.1, 0.05))
    return _surface_boolean_body(boolean_union([base, post]))


def _surface_box_intersection_overlap() -> SurfaceBody:
    left = make_box(size=(1.4, 1.0, 0.8))
    right = make_box(size=(0.9, 0.8, 0.9), center=(0.35, 0.12, 0.04))
    return _surface_boolean_body(boolean_intersection([left, right]))


def _surface_box_difference_corner_notch() -> SurfaceBody:
    base = make_box(size=(1.6, 1.2, 0.9))
    cutter = make_box(size=(0.85, 0.7, 1.1), center=(0.5, 0.3, 0.0))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def _surface_box_difference_end_recess() -> SurfaceBody:
    base = make_box(size=(2.0, 1.2, 0.8))
    cutter = make_box(size=(0.85, 0.72, 1.0), center=(0.75, 0.0, 0.0))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def _surface_box_difference_side_recess() -> SurfaceBody:
    base = make_box(size=(2.0, 1.2, 0.8))
    cutter = make_box(size=(0.85, 0.72, 1.0), center=(0.0, 0.45, 0.0))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def _surface_box_difference_top_pocket() -> SurfaceBody:
    base = make_box(size=(1.6, 1.2, 0.9))
    cutter = make_box(size=(0.72, 0.54, 0.5), center=(0.0, 0.0, 0.35))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def _surface_box_difference_shallow_step() -> SurfaceBody:
    base = make_box(size=(1.8, 1.1, 0.8))
    cutter = make_box(size=(0.9, 1.3, 0.46), center=(0.45, 0.0, 0.28))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def _surface_box_difference_coincident_face() -> SurfaceBody:
    base = make_box(size=(1.0, 1.0, 1.0))
    cutter = make_box(size=(1.0, 1.0, 1.0), center=(1.0, 0.0, 0.0))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def _surface_box_union_disjoint() -> SurfaceBody:
    left = make_box(size=(0.8, 0.8, 0.8), center=(-0.6, 0.0, 0.0))
    right = make_box(size=(0.6, 0.6, 0.6), center=(0.55, 0.0, 0.0))
    return _surface_boolean_body(boolean_union([left, right]))


@pytest.mark.stl
@pytest.mark.parametrize(
    ("fixture_name", "builder", "min_facets"),
    [
        ("surfacebody/primitives/sphere", lambda: make_sphere(radius=0.75), 100),
        (
            "surfacebody/primitives/cylinder",
            lambda: make_cylinder(radius=0.55, height=1.4, center=(0.0, 0.0, 0.0)),
            80,
        ),
        (
            "surfacebody/primitives/cone_frustum",
            lambda: make_cone(bottom_diameter=1.2, top_diameter=0.35, height=1.5),
            80,
        ),
        (
            "surfacebody/primitives/torus",
            lambda: make_torus(major_radius=0.9, minor_radius=0.22),
            100,
        ),
        (
            "surfacebody/primitives/ngon_prism",
            lambda: make_ngon(sides=7, radius=0.65, height=0.9),
            20,
        ),
        (
            "surfacebody/primitives/pyramid_frustum",
            lambda: make_prism(base_size=(1.2, 0.9), top_size=(0.28, 0.42), height=1.15),
            12,
        ),
        ("surfacebody/primitives/transformed_group", _transformed_primitive_group, 60),
        ("surfacebody/primitives/mixed_scale", _mixed_scale_primitives, 80),
        ("surfacebody/primitives/thin_stable_dimensions", _thin_stable_primitives, 40),
        ("surfacebody/primitives/authored_color_smoke", _authored_color_primitive_smoke, 80),
    ],
)
def test_surface_primitive_reference_stls(
    fixture_name: str,
    builder: Callable[[], SurfaceBody],
    min_facets: int,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    _ensure_stl_reference(
        model=builder(),
        fixture_name=fixture_name,
        tmp_path=tmp_path,
        update_dirty_reference_images=update_dirty_reference_images,
        min_facets=min_facets,
    )


@pytest.mark.stl
@pytest.mark.parametrize(
    ("fixture_name", "builder", "min_facets"),
    [
        ("surfacebody/csg/box_union_overlap", _surface_box_union_overlap, 20),
        ("surfacebody/csg/box_intersection_overlap", _surface_box_intersection_overlap, 12),
        ("surfacebody/csg/box_difference_corner_notch", _surface_box_difference_corner_notch, 20),
        ("surfacebody/csg/box_difference_end_recess", _surface_box_difference_end_recess, 20),
        ("surfacebody/csg/box_difference_side_recess", _surface_box_difference_side_recess, 20),
        ("surfacebody/csg/box_difference_top_pocket", _surface_box_difference_top_pocket, 40),
        ("surfacebody/csg/box_difference_shallow_step", _surface_box_difference_shallow_step, 12),
        ("surfacebody/csg/box_difference_coincident_face", _surface_box_difference_coincident_face, 12),
        ("surfacebody/csg/box_union_disjoint", _surface_box_union_disjoint, 24),
    ],
)
def test_surface_csg_reference_stls(
    fixture_name: str,
    builder: Callable[[], SurfaceBody],
    min_facets: int,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    _ensure_stl_reference(
        model=builder(),
        fixture_name=fixture_name,
        tmp_path=tmp_path,
        update_dirty_reference_images=update_dirty_reference_images,
        min_facets=min_facets,
    )


@pytest.mark.stl
@pytest.mark.parametrize(
    ("fixture_name", "builder", "min_facets"),
    [
        ("loft/matrix/circle_to_circle", _circle_to_circle_loft, 100),
        ("loft/matrix/square_to_square", _square_to_square_loft, 50),
        ("loft/matrix/circle_to_square", _circle_to_square_loft, 80),
        ("loft/matrix/rectangle_to_rounded_rectangle", _rectangle_to_rounded_rectangle_loft, 80),
        ("loft/matrix/triangle_to_hexagon", _triangle_to_hexagon_loft, 50),
        ("loft/matrix/open_ended_circle", _open_ended_loft, 60),
        ("loft/matrix/capped_circle_planar_caps", _capped_circle_planar_caps_loft, 80),
        ("loft/matrix/tapered_circle", _tapered_circle_loft, 80),
        ("loft/matrix/bulb_circle", _bulb_circle_loft, 80),
        ("loft/matrix/high_twist_rectangle", _high_twist_loft, 50),
        ("loft/matrix/smooth_profile_sequence", _smooth_profile_sequence_loft, 80),
        ("loft/matrix/mixed_sharp_smooth", _mixed_sharp_smooth_loft, 80),
        ("loft/matrix/s_curve_station_path", _s_curve_station_loft, 100),
        ("loft/matrix/helical_station_path", _helical_station_loft, 120),
        ("loft/matrix/very_short_span", _very_short_span_loft, 20),
        ("loft/matrix/nonuniform_station_spacing", _nonuniform_station_spacing_loft, 100),
        ("loft/matrix/branching_manifold_open", _branching_manifold_open_loft, 80),
        ("loft/matrix/sharp_corner_star", _sharp_corner_star_loft, 80),
        ("loft/matrix/tiny_to_normal_profile", _tiny_to_normal_profile_loft, 80),
        ("loft/matrix/hole_to_solid_transition", _hole_to_solid_transition_loft, 80),
        ("loft/matrix/explicit_station_frames", _explicit_station_frames_loft, 80),
        ("loft/matrix/tight_curvature_frame_transport", _tight_curvature_frame_transport_loft, 120),
        ("loft/matrix/branch_merge_many_to_one", _branch_merge_many_to_one_loft, 80),
        ("loft/matrix/asymmetric_branch_lengths", _asymmetric_branch_lengths_loft, 80),
        ("loft/matrix/near_coincident_stations", _near_coincident_station_loft, 40),
        ("loft/matrix/curved_bezier_path", _curved_bezier_path_loft, 100),
        ("loft/matrix/many_to_many_region_expand", _many_to_many_region_expand_loft, 100),
        ("loft/matrix/many_to_many_region_collapse", _many_to_many_region_collapse_loft, 100),
        ("loft/matrix/deterministic_resampling", _deterministic_resampling_loft, 80),
        ("loft/matrix/domed_tapered_caps", _domed_tapered_caps_loft, 100),
        ("loft/matrix/slope_dome_caps", _slope_dome_caps_loft, 80),
        ("loft/matrix/reversed_winding_repair", _reversed_winding_repair_loft, 40),
        ("loft/matrix/notched_cylinder_correspondence", lambda: _loft_from_profiles(build_notched_cylinder_correspondence_profiles), 100),
        ("loft/matrix/notched_phase_shift_cylinder", lambda: _loft_from_profiles(build_notched_phase_shift_cylinder_profiles), 100),
        ("loft/matrix/dual_cylinder_correspondence", lambda: _loft_from_profiles(build_dual_cylinder_correspondence_profiles), 100),
        ("loft/matrix/perforated_cylinder_correspondence", lambda: _loft_from_profiles(build_perforated_cylinder_correspondence_profiles), 100),
        ("loft/matrix/perforated_vessel", lambda: _loft_from_profiles(build_perforated_vessel_profiles), 100),
    ],
)
def test_loft_matrix_reference_stls(
    fixture_name: str,
    builder: Callable[[], SurfaceBody],
    min_facets: int,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    _ensure_stl_reference(
        model=builder(),
        fixture_name=fixture_name,
        tmp_path=tmp_path,
        update_dirty_reference_images=update_dirty_reference_images,
        min_facets=min_facets,
    )
