from __future__ import annotations

import numpy as np

from impression.modeling import Section, Station, as_section
from impression.modeling.drawing2d import PlanarShape2D, make_circle, make_polygon, make_rect


def _circle_hole(radius: float, center: tuple[float, float]) -> object:
    return make_circle(radius=radius, center=center).outer


def make_perforated_disc(
    outer_radius: float,
    holes: list[tuple[float, tuple[float, float]]] | None = None,
) -> PlanarShape2D:
    shape = make_circle(radius=outer_radius)
    if holes:
        shape.holes.extend(_circle_hole(radius, center) for radius, center in holes)
    return shape


def build_branching_manifold_profiles() -> tuple[list[Section], np.ndarray]:
    trunk = Section((as_section(make_circle(radius=0.9)).regions[0],))
    transition = Section((as_section(make_rect(size=(1.4, 1.0))).regions[0],))
    left = as_section(make_circle(radius=0.52, center=(-0.8, 0.0))).regions[0]
    right = as_section(make_circle(radius=0.52, center=(0.8, 0.0))).regions[0]
    branches_mid = Section((left, right))
    left_far = as_section(make_circle(radius=0.48, center=(-1.0, 0.25))).regions[0]
    right_far = as_section(make_circle(radius=0.48, center=(1.0, -0.2))).regions[0]
    branches_end = Section((left_far, right_far))
    path = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.15, 0.0, 0.9],
            [0.35, 0.15, 1.9],
            [0.8, 0.45, 3.1],
        ],
        dtype=float,
    )
    return [trunk, transition, branches_mid, branches_end], path


def build_perforated_vessel_profiles() -> tuple[list[PlanarShape2D], np.ndarray]:
    profiles = [
        make_perforated_disc(outer_radius=1.25),
        make_perforated_disc(outer_radius=1.15, holes=[(0.16, (-0.32, 0.0))]),
        make_perforated_disc(
            outer_radius=1.2,
            holes=[(0.15, (-0.42, 0.0)), (0.15, (0.42, 0.0))],
        ),
        make_perforated_disc(outer_radius=1.1, holes=[(0.16, (0.34, 0.0))]),
        make_perforated_disc(outer_radius=1.3),
    ]
    path = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.8],
            [0.2, 0.1, 1.8],
            [0.35, -0.1, 2.8],
            [0.55, -0.2, 3.8],
        ],
        dtype=float,
    )
    return profiles, path


def build_ambiguous_hole_cluster_profiles() -> tuple[list[PlanarShape2D], np.ndarray, list[Station]]:
    outer = make_rect(size=(2.6, 1.9)).outer
    start = PlanarShape2D(
        outer=outer,
        holes=[
            _circle_hole(0.2, (-0.55, 0.0)),
            _circle_hole(0.2, (0.55, 0.0)),
        ],
    )
    middle = PlanarShape2D(
        outer=outer,
        holes=[
            _circle_hole(0.18, (-0.75, 0.0)),
            _circle_hole(0.18, (0.0, 0.0)),
            _circle_hole(0.18, (0.75, 0.0)),
        ],
    )
    end = PlanarShape2D(
        outer=outer,
        holes=[
            _circle_hole(0.19, (-0.65, 0.1)),
            _circle_hole(0.19, (0.65, -0.1)),
        ],
    )
    profiles = [start, middle, end]
    path = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 1.2],
            [0.25, 0.1, 2.4],
        ],
        dtype=float,
    )
    stations = [
        Station(t=float(idx), section=as_section(profile), origin=path[idx], u=[1, 0, 0], v=[0, 1, 0], n=[0, 0, 1])
        for idx, profile in enumerate(profiles)
    ]
    return profiles, path, stations


def build_square_correspondence_profiles() -> tuple[list[Section], np.ndarray]:
    profiles = [
        as_section(make_rect(size=(1.55, 0.62))),
        as_section(make_rect(size=(1.45, 0.68), center=(0.05, -0.02))),
        as_section(make_rect(size=(1.70, 0.58), center=(-0.03, 0.04))),
        as_section(make_rect(size=(1.60, 0.64), center=(0.02, 0.01))),
    ]
    path = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.15, 0.05, 1.0],
            [0.28, -0.08, 2.0],
            [0.36, 0.02, 3.0],
        ],
        dtype=float,
    )
    return profiles, path


def build_cylinder_correspondence_profiles() -> tuple[list[Section], np.ndarray]:
    profiles = [
        as_section(make_circle(radius=0.55)),
        as_section(make_circle(radius=0.62, center=(0.03, -0.02))),
        as_section(make_circle(radius=0.48, center=(-0.04, 0.03))),
        as_section(make_circle(radius=0.58, center=(0.02, 0.01))),
    ]
    path = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.10, -0.04, 0.9],
            [0.24, 0.06, 1.9],
            [0.33, -0.02, 3.0],
        ],
        dtype=float,
    )
    return profiles, path


def _rectangle_points(width: float, height: float, *, center: tuple[float, float] = (0.0, 0.0)) -> list[tuple[float, float]]:
    cx, cy = center
    hx = width / 2.0
    hy = height / 2.0
    return [
        (cx - hx, cy - hy),
        (cx + hx, cy - hy),
        (cx + hx, cy + hy),
        (cx - hx, cy + hy),
    ]


def _rotated_start(points: list[tuple[float, float]], shift: int) -> list[tuple[float, float]]:
    shift = int(shift) % len(points)
    return points[shift:] + points[:shift]


def _reversed_with_shift(points: list[tuple[float, float]], shift: int) -> list[tuple[float, float]]:
    reversed_points = list(reversed(points))
    return _rotated_start(reversed_points, shift)


def _phased_circle_polygon(radius: float, *, phase_deg: float, samples: int, center: tuple[float, float]) -> PlanarShape2D:
    angles = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False) + np.deg2rad(float(phase_deg))
    cx, cy = center
    points = [(cx + radius * float(np.cos(angle)), cy + radius * float(np.sin(angle))) for angle in angles]
    return make_polygon(points)


def _notched_circle_polygon(
    radius: float,
    *,
    notch_depth: float,
    notch_center_deg: float,
    notch_half_width_deg: float,
    phase_deg: float,
    samples: int,
    center: tuple[float, float],
) -> PlanarShape2D:
    angles = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False) + np.deg2rad(float(phase_deg))
    notch_center = np.deg2rad(float(notch_center_deg))
    notch_half_width = np.deg2rad(float(notch_half_width_deg))
    cx, cy = center
    points: list[tuple[float, float]] = []
    for angle in angles:
        wrapped = float(np.angle(np.exp(1j * (angle - notch_center))))
        if abs(wrapped) <= notch_half_width:
            local = abs(wrapped) / max(notch_half_width, 1e-9)
            radius_scale = 1.0 - float(notch_depth) * (1.0 - local)
        else:
            radius_scale = 1.0
        local_radius = float(radius) * radius_scale
        points.append((cx + local_radius * float(np.cos(angle)), cy + local_radius * float(np.sin(angle))))
    return make_polygon(points)


def build_anchor_shift_rectangle_profiles() -> tuple[list[Section], np.ndarray]:
    base = _rectangle_points(1.55, 0.62)
    profiles = [
        as_section(make_polygon(base)),
        as_section(make_polygon(_rotated_start(_rectangle_points(1.45, 0.68, center=(0.05, -0.02)), 1))),
        as_section(make_polygon(_reversed_with_shift(_rectangle_points(1.70, 0.58, center=(-0.03, 0.04)), 2))),
        as_section(make_polygon(_rotated_start(_rectangle_points(1.60, 0.64, center=(0.02, 0.01)), 3))),
    ]
    path = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.15, 0.05, 1.0],
            [0.28, -0.08, 2.0],
            [0.36, 0.02, 3.0],
        ],
        dtype=float,
    )
    return profiles, path


def build_phase_shift_cylinder_profiles() -> tuple[list[Section], np.ndarray]:
    profiles = [
        as_section(_phased_circle_polygon(0.55, phase_deg=0.0, samples=28, center=(0.0, 0.0))),
        as_section(_phased_circle_polygon(0.62, phase_deg=26.0, samples=28, center=(0.03, -0.02))),
        as_section(_phased_circle_polygon(0.48, phase_deg=-33.0, samples=28, center=(-0.04, 0.03))),
        as_section(_phased_circle_polygon(0.58, phase_deg=57.0, samples=28, center=(0.02, 0.01))),
    ]
    path = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.10, -0.04, 0.9],
            [0.24, 0.06, 1.9],
            [0.33, -0.02, 3.0],
        ],
        dtype=float,
    )
    return profiles, path


def build_notched_cylinder_correspondence_profiles() -> tuple[list[Section], np.ndarray]:
    profiles = [
        as_section(
            _notched_circle_polygon(
                0.55,
                notch_depth=0.24,
                notch_center_deg=90.0,
                notch_half_width_deg=18.0,
                phase_deg=0.0,
                samples=36,
                center=(0.0, 0.0),
            )
        ),
        as_section(
            _notched_circle_polygon(
                0.62,
                notch_depth=0.24,
                notch_center_deg=90.0,
                notch_half_width_deg=18.0,
                phase_deg=0.0,
                samples=36,
                center=(0.03, -0.02),
            )
        ),
        as_section(
            _notched_circle_polygon(
                0.48,
                notch_depth=0.24,
                notch_center_deg=90.0,
                notch_half_width_deg=18.0,
                phase_deg=0.0,
                samples=36,
                center=(-0.04, 0.03),
            )
        ),
        as_section(
            _notched_circle_polygon(
                0.58,
                notch_depth=0.24,
                notch_center_deg=90.0,
                notch_half_width_deg=18.0,
                phase_deg=0.0,
                samples=36,
                center=(0.02, 0.01),
            )
        ),
    ]
    path = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.10, -0.04, 0.9],
            [0.24, 0.06, 1.9],
            [0.33, -0.02, 3.0],
        ],
        dtype=float,
    )
    return profiles, path


def build_notched_phase_shift_cylinder_profiles() -> tuple[list[Section], np.ndarray]:
    profiles = [
        as_section(
            _notched_circle_polygon(
                0.55,
                notch_depth=0.24,
                notch_center_deg=90.0,
                notch_half_width_deg=18.0,
                phase_deg=0.0,
                samples=36,
                center=(0.0, 0.0),
            )
        ),
        as_section(
            _notched_circle_polygon(
                0.62,
                notch_depth=0.24,
                notch_center_deg=90.0,
                notch_half_width_deg=18.0,
                phase_deg=26.0,
                samples=36,
                center=(0.03, -0.02),
            )
        ),
        as_section(
            _notched_circle_polygon(
                0.48,
                notch_depth=0.24,
                notch_center_deg=90.0,
                notch_half_width_deg=18.0,
                phase_deg=-33.0,
                samples=36,
                center=(-0.04, 0.03),
            )
        ),
        as_section(
            _notched_circle_polygon(
                0.58,
                notch_depth=0.24,
                notch_center_deg=90.0,
                notch_half_width_deg=18.0,
                phase_deg=57.0,
                samples=36,
                center=(0.02, 0.01),
            )
        ),
    ]
    path = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.10, -0.04, 0.9],
            [0.24, 0.06, 1.9],
            [0.33, -0.02, 3.0],
        ],
        dtype=float,
    )
    return profiles, path


def build_dual_cylinder_correspondence_profiles() -> tuple[list[Section], np.ndarray]:
    profiles = [
        Section(
            (
                as_section(make_circle(radius=0.42, center=(-0.72, 0.0))).regions[0],
                as_section(make_circle(radius=0.42, center=(0.72, 0.0))).regions[0],
            )
        ),
        Section(
            (
                as_section(make_circle(radius=0.47, center=(-0.84, 0.08))).regions[0],
                as_section(make_circle(radius=0.38, center=(0.62, -0.06))).regions[0],
            )
        ),
        Section(
            (
                as_section(make_circle(radius=0.36, center=(-0.64, 0.03))).regions[0],
                as_section(make_circle(radius=0.50, center=(0.88, 0.05))).regions[0],
            )
        ),
        Section(
            (
                as_section(make_circle(radius=0.44, center=(-0.78, -0.04))).regions[0],
                as_section(make_circle(radius=0.44, center=(0.78, 0.04))).regions[0],
            )
        ),
    ]
    path = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.12, 0.04, 0.9],
            [0.24, -0.03, 1.9],
            [0.36, 0.02, 3.0],
        ],
        dtype=float,
    )
    return profiles, path


def build_perforated_cylinder_correspondence_profiles() -> tuple[list[PlanarShape2D], np.ndarray]:
    profiles = [
        make_perforated_disc(
            outer_radius=1.12,
            holes=[(0.16, (-0.38, 0.0)), (0.14, (0.38, 0.02))],
        ),
        make_perforated_disc(
            outer_radius=1.18,
            holes=[(0.18, (-0.46, 0.08)), (0.13, (0.30, -0.05))],
        ),
        make_perforated_disc(
            outer_radius=1.08,
            holes=[(0.15, (-0.30, 0.06)), (0.18, (0.46, 0.06))],
        ),
        make_perforated_disc(
            outer_radius=1.15,
            holes=[(0.17, (-0.42, -0.02)), (0.15, (0.36, 0.0))],
        ),
    ]
    path = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.06, -0.02, 0.9],
            [0.18, 0.04, 1.9],
            [0.26, -0.03, 3.0],
        ],
        dtype=float,
    )
    return profiles, path
