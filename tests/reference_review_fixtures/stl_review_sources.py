"""Loadable source entrypoints for dirty STL reference review fixtures."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

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
    handoff_hinge_surface,
    make_bistable_hinge,
    make_box,
    make_cone,
    make_cylinder,
    make_living_hinge,
    make_ngon,
    make_prism,
    make_surface_body,
    make_traditional_hinge_pair,
    make_sphere,
    make_torus,
    loft,
    loft_sections,
    rotate,
    scale,
    translate,
)
from impression.modeling.drawing2d import make_circle, make_polygon, make_rect
from impression.modeling.drafting import make_arrow
from impression.modeling.heightmap import heightmap
from impression.modeling.text import make_text
from tests.csg_reference_fixtures import (
    build_csg_difference_slot_fixture,
    build_csg_union_box_post_fixture,
)
from tests.loft_showcases import (
    build_anchor_shift_rectangle_profiles,
    build_branching_manifold_profiles,
    build_cylinder_correspondence_profiles,
    build_dual_cylinder_correspondence_profiles,
    build_notched_cylinder_correspondence_profiles,
    build_notched_phase_shift_cylinder_profiles,
    build_perforated_cylinder_correspondence_profiles,
    build_perforated_vessel_profiles,
    build_phase_shift_cylinder_profiles,
    build_square_correspondence_profiles,
)
from tests.text_font_fixtures import require_glyph_capable_font

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _surface_boolean_body(result: object) -> SurfaceBody:
    if getattr(result, "status", None) != "succeeded":
        raise ValueError(f"Surface CSG fixture did not succeed: {getattr(result, 'failure_reason', None)}")
    body = getattr(result, "body", None)
    if not isinstance(body, SurfaceBody):
        raise TypeError("Surface CSG fixture did not return a SurfaceBody.")
    return body


def _loft_from_profiles(builder):
    profiles, path = builder()
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=True,
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


def build_loft_anchor_shift_rectangle():
    return _loft_from_profiles(build_anchor_shift_rectangle_profiles)


def build_loft_branching_manifold():
    profiles, path = build_branching_manifold_profiles()
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=True,
        split_merge_mode="resolve",
    )


def build_loft_cylinder_correspondence():
    return _loft_from_profiles(build_cylinder_correspondence_profiles)


def build_loft_hourglass_vessel():
    module_path = _PROJECT_ROOT / "docs/examples/loft/real_world/loft_hourglass_vessel_example.py"
    spec = importlib.util.spec_from_file_location("loft_hourglass_vessel_example", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_surface_body(module.TEST_PARAMETERS, module.TEST_QUALITY)


def build_loft_phase_shift_cylinder():
    return _loft_from_profiles(build_phase_shift_cylinder_profiles)


def build_loft_square_correspondence():
    return _loft_from_profiles(build_square_correspondence_profiles)


def build_loft_matrix_circle_to_circle():
    profiles = [
        make_circle(radius=0.48),
        make_circle(radius=0.62, center=(0.04, -0.03)),
        make_circle(radius=0.54, center=(-0.03, 0.02)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.08, 0.03, 0.9), (0.16, -0.02, 1.8)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_square_to_square():
    profiles = [
        make_rect(size=(0.9, 0.9)),
        make_rect(size=(1.1, 0.7), center=(0.04, -0.02)),
        make_rect(size=(0.8, 1.0), center=(-0.02, 0.03)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.1, -0.04, 0.85), (0.18, 0.02, 1.7)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_circle_to_square():
    profiles = [
        make_circle(radius=0.55),
        make_polygon([(-0.55, -0.35), (0.55, -0.35), (0.55, 0.35), (-0.55, 0.35)]),
        make_rect(size=(0.9, 0.9), center=(0.03, -0.02)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.06, 0.03, 0.85), (0.12, -0.02, 1.7)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_rectangle_to_rounded_rectangle():
    profiles = [
        make_rect(size=(1.0, 0.62)),
        _rounded_rectangle_profile(1.06, 0.72, 0.12),
        _rounded_rectangle_profile(0.84, 0.58, 0.18),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.06, 0.02, 0.82), (0.1, -0.02, 1.68)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_triangle_to_hexagon():
    triangle = make_polygon([(0.0, 0.62), (-0.58, -0.36), (0.58, -0.36)])
    hexagon_points = [
        (0.62 * float(np.cos(angle)), 0.62 * float(np.sin(angle)))
        for angle in np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    ]
    profiles = [triangle, make_polygon(hexagon_points), make_rect(size=(0.75, 0.55))]
    path = np.asarray([(0.0, 0.0, 0.0), (0.04, 0.04, 0.78), (0.1, -0.03, 1.65)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_open_ended_circle():
    profiles = [make_circle(radius=0.42), make_circle(radius=0.58), make_circle(radius=0.38)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.04, 0.06, 0.78), (0.08, -0.04, 1.62)], dtype=float)
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=False,
        split_merge_mode="resolve",
    )


def build_loft_matrix_capped_circle_planar_caps():
    profiles = [make_circle(radius=0.46), make_circle(radius=0.5), make_circle(radius=0.42)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.01, 0.7), (0.04, -0.01, 1.4)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_tapered_circle():
    profiles = [make_circle(radius=0.72), make_circle(radius=0.48), make_circle(radius=0.28)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.03, 0.0, 0.78), (0.04, 0.0, 1.65)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_bulb_circle():
    profiles = [
        make_circle(radius=0.34),
        make_circle(radius=0.78, center=(0.03, 0.0)),
        make_circle(radius=0.42, center=(-0.02, 0.02)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.03, 0.75), (0.05, -0.03, 1.6)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_high_twist_rectangle():
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


def build_loft_matrix_s_curve_station_path():
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


def build_loft_matrix_smooth_profile_sequence():
    profiles = [
        make_circle(radius=0.44),
        _rounded_rectangle_profile(0.88, 0.72, 0.28),
        make_circle(radius=0.5, center=(0.02, -0.02)),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.08, 0.02, 0.82), (0.12, -0.02, 1.6)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_mixed_sharp_smooth():
    profiles = [
        _star_profile(0.48, 0.26),
        _rounded_rectangle_profile(0.86, 0.64, 0.18),
        make_circle(radius=0.46),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.04, 0.04, 0.74), (0.1, -0.02, 1.52)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_helical_station_path():
    profiles = [make_circle(radius=0.2 + 0.03 * (index % 2)) for index in range(6)]
    angles = np.linspace(0.0, 1.5 * np.pi, len(profiles))
    path = np.asarray(
        [(0.34 * float(np.cos(angle)), 0.34 * float(np.sin(angle)), 0.34 * index) for index, angle in enumerate(angles)],
        dtype=float,
    )
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_very_short_span():
    profiles = [make_rect(size=(0.72, 0.5)), make_rect(size=(0.64, 0.44), center=(0.02, -0.01))]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.0, 0.08)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_nonuniform_station_spacing():
    profiles = [
        make_circle(radius=0.34),
        make_circle(radius=0.48),
        make_circle(radius=0.52),
        make_circle(radius=0.38),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.01, 0.12), (0.08, -0.03, 1.35), (0.1, 0.0, 1.62)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_branching_manifold_open():
    profiles, path = build_branching_manifold_profiles()
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=False,
        split_merge_mode="resolve",
    )


def build_loft_matrix_sharp_corner_star():
    profiles = [
        _star_profile(0.55, 0.28),
        _star_profile(0.48, 0.22),
        _star_profile(0.6, 0.32),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.03, -0.03, 0.76), (0.0, 0.02, 1.5)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_tiny_to_normal_profile():
    profiles = [make_circle(radius=0.08), make_circle(radius=0.28), make_circle(radius=0.62)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.02, 0.0, 0.55), (0.04, 0.0, 1.55)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_hole_to_solid_transition():
    from tests.loft_showcases import make_perforated_disc

    profiles = [
        make_perforated_disc(outer_radius=0.72, holes=[(0.14, (0.0, 0.0))]),
        make_circle(radius=0.64),
        make_circle(radius=0.56),
    ]
    path = np.asarray([(0.0, 0.0, 0.0), (0.03, 0.02, 0.72), (0.08, -0.02, 1.5)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_explicit_station_frames():
    profiles = [as_section(make_circle(radius=0.46)), as_section(make_rect(size=(0.88, 0.58))), as_section(make_circle(radius=0.38))]
    half_root = float(np.sqrt(0.5))
    cos_30 = float(np.sqrt(3.0) / 2.0)
    stations = [
        Station(t=0.0, section=profiles[0], origin=(0.0, 0.0, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=0.45, section=profiles[1], origin=(0.2, 0.0, 0.8), u=(cos_30, 0.5, 0.0), v=(-0.5, cos_30, 0.0), n=(0.0, 0.0, 1.0)),
        Station(t=1.0, section=profiles[2], origin=(0.44, 0.18, 1.65), u=(half_root, half_root, 0.0), v=(-half_root, half_root, 0.0), n=(0.0, 0.0, 1.0)),
    ]
    return Loft(progression=[0.0, 0.45, 1.0], stations=stations, topology=profiles, cap_ends=True)


def build_loft_matrix_tight_curvature_frame_transport():
    profiles = [make_circle(radius=0.18 + 0.025 * (index % 2)) for index in range(7)]
    angles = np.linspace(0.0, 1.75 * np.pi, len(profiles))
    path = np.asarray(
        [(0.28 * float(np.cos(angle)), 0.28 * float(np.sin(angle)), 0.18 * index) for index, angle in enumerate(angles)],
        dtype=float,
    )
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_branch_merge_many_to_one():
    profiles, path = build_branching_manifold_profiles()
    return Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path[::-1].copy(),
        topology=list(reversed(profiles)),
        cap_ends=True,
        split_merge_mode="resolve",
    )


def build_loft_matrix_asymmetric_branch_lengths():
    trunk = Section((as_section(make_circle(radius=0.58)).regions[0],))
    split = Section((as_section(make_circle(radius=0.34, center=(-0.55, 0.0))).regions[0], as_section(make_circle(radius=0.3, center=(0.55, 0.0))).regions[0]))
    stretched = Section((as_section(make_circle(radius=0.28, center=(-1.18, 0.38))).regions[0], as_section(make_circle(radius=0.34, center=(0.72, -0.1))).regions[0]))
    profiles = [trunk, split, stretched]
    path = np.asarray([(0.0, 0.0, 0.0), (0.08, 0.04, 0.9), (0.38, 0.18, 2.1)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_near_coincident_stations():
    profiles = [make_circle(radius=0.3), make_circle(radius=0.32), make_circle(radius=0.5)]
    path = np.asarray([(0.0, 0.0, 0.0), (0.002, 0.0, 0.015), (0.04, 0.0, 1.05)], dtype=float)
    return _loft_from_profiles(lambda: (profiles, path))


def build_loft_matrix_curved_bezier_path():
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


def build_loft_matrix_many_to_many_region_expand():
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


def build_loft_matrix_many_to_many_region_collapse():
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


def build_loft_matrix_deterministic_resampling():
    circle_points = [
        (0.5 * float(np.cos(angle)), 0.5 * float(np.sin(angle)))
        for angle in np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False)
    ]
    profiles = [make_polygon(circle_points), make_rect(size=(0.75, 0.75)), make_circle(radius=0.42)]
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.1, 0.05, 0.8), (0.16, -0.02, 1.6)])
    return loft(profiles, path=path, samples=44, cap_ends=True)


def build_loft_matrix_domed_tapered_caps():
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


def build_loft_matrix_slope_dome_caps():
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


def build_loft_matrix_reversed_winding_repair():
    square = [(-0.45, -0.45), (-0.45, 0.45), (0.45, 0.45), (0.45, -0.45)]
    profiles = [make_polygon(square), make_polygon(list(reversed(square))), make_polygon(square)]
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.02, 0.0, 0.7), (0.05, 0.0, 1.4)])
    return loft(profiles, path=path, samples=32, cap_ends=True)


def build_loft_matrix_notched_cylinder_correspondence():
    return _loft_from_profiles(build_notched_cylinder_correspondence_profiles)


def build_loft_matrix_notched_phase_shift_cylinder():
    return _loft_from_profiles(build_notched_phase_shift_cylinder_profiles)


def build_loft_matrix_dual_cylinder_correspondence():
    return _loft_from_profiles(build_dual_cylinder_correspondence_profiles)


def build_loft_matrix_perforated_cylinder_correspondence():
    return _loft_from_profiles(build_perforated_cylinder_correspondence_profiles)


def build_loft_matrix_perforated_vessel():
    return _loft_from_profiles(build_perforated_vessel_profiles)


def build_surfacebody_box():
    return make_box(size=(2.0, 3.0, 1.5), center=(0.0, 0.0, 0.0))


def build_surfacebody_primitive_sphere():
    return make_sphere(radius=0.75)


def build_surfacebody_primitive_cylinder():
    return make_cylinder(radius=0.55, height=1.4, center=(0.0, 0.0, 0.0))


def build_surfacebody_primitive_cone_frustum():
    return make_cone(bottom_diameter=1.2, top_diameter=0.35, height=1.5)


def build_surfacebody_primitive_torus():
    return make_torus(major_radius=0.9, minor_radius=0.22)


def build_surfacebody_primitive_ngon_prism():
    return make_ngon(sides=7, radius=0.65, height=0.9)


def build_surfacebody_primitive_pyramid_frustum():
    return make_prism(base_size=(1.2, 0.9), top_size=(0.28, 0.42), height=1.15)


def build_surfacebody_primitive_transformed_group():
    box = rotate(make_box(size=(0.75, 0.45, 0.35), color="#d9822b"), axis=(0.0, 0.0, 1.0), angle_deg=28.0)
    box = translate(box, (-0.55, -0.08, 0.0))
    cylinder = rotate(make_cylinder(radius=0.18, height=0.95, color="#5b84b1"), axis=(1.0, 0.0, 0.0), angle_deg=72.0)
    cylinder = translate(cylinder, (0.38, 0.22, 0.08))
    frustum = scale(make_cone(bottom_diameter=0.46, top_diameter=0.22, height=0.62, color="#72a276"), (1.0, 0.7, 1.2))
    frustum = translate(frustum, (0.08, -0.42, 0.05))
    return _combine_surface_bodies([box, cylinder, frustum])


def build_surfacebody_primitive_mixed_scale():
    tiny = translate(make_box(size=(0.04, 0.04, 0.04), color="#ffcf5a"), (-0.72, -0.42, 0.0))
    slender = translate(make_cylinder(radius=0.035, height=1.4, color="#8f6bb8"), (0.0, 0.0, 0.0))
    plate = translate(make_box(size=(1.5, 0.08, 0.18), color="#4f9d9a"), (0.15, 0.35, 0.0))
    large = translate(make_sphere(radius=0.28, color="#d95d39"), (0.72, -0.2, 0.0))
    return _combine_surface_bodies([tiny, slender, plate, large])


def build_surfacebody_primitive_thin_stable_dimensions():
    wafer = translate(make_box(size=(1.1, 0.7, 0.025), color="#a8c686"), (0.0, 0.0, -0.05))
    pin = translate(make_cylinder(radius=0.025, height=0.62, color="#f2a65a"), (-0.38, 0.0, 0.0))
    needle = rotate(make_cone(bottom_diameter=0.08, top_diameter=0.01, height=0.85, color="#6c91bf"), axis=(0.0, 1.0, 0.0), angle_deg=86.0)
    needle = translate(needle, (0.32, 0.0, 0.02))
    return _combine_surface_bodies([wafer, pin, needle])


def build_surfacebody_primitive_authored_color_smoke():
    red = translate(make_box(size=(0.42, 0.42, 0.42), color="#d95d39"), (-0.48, 0.0, 0.0))
    blue = translate(make_sphere(radius=0.28, color="#5b84b1"), (0.0, 0.0, 0.0))
    green = translate(make_cylinder(radius=0.22, height=0.48, color="#72a276"), (0.48, 0.0, 0.0))
    return _combine_surface_bodies([red, blue, green])


def build_surfacebody_csg_difference_slot():
    return build_csg_difference_slot_fixture()["result_body"]


def build_surfacebody_csg_union_box_post():
    return build_csg_union_box_post_fixture()["result_body"]


def build_surfacebody_csg_box_union_overlap():
    base = make_box(size=(1.4, 1.0, 0.8))
    post = make_box(size=(0.7, 0.55, 0.9), center=(0.55, 0.1, 0.05))
    return _surface_boolean_body(boolean_union([base, post]))


def build_surfacebody_csg_box_intersection_overlap():
    left = make_box(size=(1.4, 1.0, 0.8))
    right = make_box(size=(0.9, 0.8, 0.9), center=(0.35, 0.12, 0.04))
    return _surface_boolean_body(boolean_intersection([left, right]))


def build_surfacebody_csg_box_difference_corner_notch():
    base = make_box(size=(1.6, 1.2, 0.9))
    cutter = make_box(size=(0.85, 0.7, 1.1), center=(0.5, 0.3, 0.0))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def build_surfacebody_csg_box_difference_end_recess():
    base = make_box(size=(2.0, 1.2, 0.8))
    cutter = make_box(size=(0.85, 0.72, 1.0), center=(0.75, 0.0, 0.0))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def build_surfacebody_csg_box_difference_side_recess():
    base = make_box(size=(2.0, 1.2, 0.8))
    cutter = make_box(size=(0.85, 0.72, 1.0), center=(0.0, 0.45, 0.0))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def build_surfacebody_csg_box_difference_top_pocket():
    base = make_box(size=(1.6, 1.2, 0.9))
    cutter = make_box(size=(0.72, 0.54, 0.5), center=(0.0, 0.0, 0.35))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def build_surfacebody_csg_box_difference_shallow_step():
    base = make_box(size=(1.8, 1.1, 0.8))
    cutter = make_box(size=(0.9, 1.3, 0.46), center=(0.45, 0.0, 0.28))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def build_surfacebody_csg_box_difference_coincident_face():
    base = make_box(size=(1.0, 1.0, 1.0))
    cutter = make_box(size=(1.0, 1.0, 1.0), center=(1.0, 0.0, 0.0))
    return _surface_boolean_body(boolean_difference(base, [cutter]))


def build_surfacebody_csg_box_union_disjoint():
    left = make_box(size=(0.8, 0.8, 0.8), center=(-0.6, 0.0, 0.0))
    right = make_box(size=(0.6, 0.6, 0.6), center=(0.55, 0.0, 0.0))
    return _surface_boolean_body(boolean_union([left, right]))


def build_surfacebody_drafting_arrow():
    return make_arrow((0.0, 0.0, 0.0), (1.25, 0.25, 0.35), color="#ffb703")


def build_surfacebody_heightmap_surface():
    return heightmap(
        np.asarray(
            [
                [0.0, 0.2, 0.6, 0.9],
                [0.1, 0.5, 0.8, 0.7],
                [0.2, 0.7, 1.0, 0.4],
                [0.0, 0.3, 0.5, 0.2],
            ],
            dtype=float,
        ),
        height=0.6,
        xy_scale=0.3,
        alpha_mode="ignore",
    )


def build_surfacebody_hinge_bistable_blank():
    return handoff_hinge_surface(make_bistable_hinge(width=40.0, preload_offset=2.0))


def build_surfacebody_hinge_living_panel():
    return handoff_hinge_surface(
        make_living_hinge(width=48.0, height=20.0, hinge_band_width=12.0, slit_pitch=1.8)
    )


def build_surfacebody_hinge_traditional_pair():
    return handoff_hinge_surface(
        make_traditional_hinge_pair(width=24.0, knuckle_count=5, opened_angle_deg=32.0)
    )


def build_surfacebody_text_surface():
    return make_text(
        "SURFACE",
        depth=0.08,
        font_size=0.3,
        font_path=str(require_glyph_capable_font()),
        color="#5b84b1",
    )
