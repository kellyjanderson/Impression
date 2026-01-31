"""Saddle loft extracted from soft_touche_interface_impression.py.

Run with:
  impression preview docs/examples/loft/saddle_example.py
"""

from __future__ import annotations

import numpy as np

from impression.modeling import Bezier3D, Path3D, loft, offset, rotate, translate
from impression.modeling.drawing2d import Arc2D, Bezier2D, Line2D, Path2D, Profile2D

# Saddle parameters
DEPTH = 100
WIDTH = 30
CURVE_DEPTH = 6

# U-shape parameters
U_ARM_HEIGHT = 4.0
U_BOTTOM_SAG = 2.0
U_CORNER_RADIUS = 1.0
U_WALL_THICKNESS = 3.0
LOFT_PROFILE_COUNT = 24


def make_u_profile(
    width: float = WIDTH,
    arm_height: float = U_ARM_HEIGHT,
    bottom_sag: float = U_BOTTOM_SAG,
    corner_radius: float = U_CORNER_RADIUS,
    wall_thickness: float = U_WALL_THICKNESS,
) -> Profile2D:
    half = width / 2.0
    radius = min(corner_radius, arm_height / 2.0, width / 4.0)
    top_y = arm_height
    center_y = (top_y - bottom_sag) / 2.0

    def pt2(x: float, y: float) -> np.ndarray:
        return np.array([x, y - center_y], dtype=float)

    segments: list = [
        Line2D(start=pt2(-half + radius, top_y), end=pt2(half - radius, top_y)),
        Arc2D(
            center=pt2(half - radius, top_y - radius),
            radius=radius,
            start_angle_deg=90,
            end_angle_deg=0,
            clockwise=True,
        ),
        Line2D(start=pt2(half, top_y - radius), end=pt2(half, 0.0)),
        Bezier2D(
            p0=pt2(half, 0.0),
            p1=pt2(half, -bottom_sag),
            p2=pt2(-half, -bottom_sag),
            p3=pt2(-half, 0.0),
        ),
        Line2D(start=pt2(-half, 0.0), end=pt2(-half, top_y - radius)),
        Arc2D(
            center=pt2(-half + radius, top_y - radius),
            radius=radius,
            start_angle_deg=180,
            end_angle_deg=90,
            clockwise=True,
        ),
    ]
    outer = Path2D(segments, closed=True)

    if wall_thickness <= 0.0:
        return Profile2D(outer=outer)

    inner_profiles = offset(Profile2D(outer=outer), delta=-wall_thickness)
    if isinstance(inner_profiles, list):
        if not inner_profiles:
            return Profile2D(outer=outer)
        inner = inner_profiles[0]
    else:
        inner = inner_profiles

    return Profile2D(outer=outer, holes=[inner.outer])


def make_soft_loft_path(depth: float = DEPTH, rise: float = CURVE_DEPTH) -> Path3D:
    def pt3(x: float, y: float, z: float) -> np.ndarray:
        return np.array([x, y, z], dtype=float)

    start = pt3(0.0, -depth / 2.0, 0.0)
    end = pt3(0.0, depth / 2.0, 0.0)
    ctrl_a = pt3(0.0, -depth / 4.0, rise)
    ctrl_b = pt3(0.0, depth / 4.0, rise)
    segments: list = [Bezier3D(p0=start, p1=ctrl_a, p2=ctrl_b, p3=end)]
    return Path3D(segments, closed=False)


def organic_pipe_saddle():
    profiles = [make_u_profile() for _ in range(LOFT_PROFILE_COUNT)]
    path = make_soft_loft_path()
    return loft(profiles, path=path, cap_ends=True)


def build():
    saddle = organic_pipe_saddle()
    rotate(saddle, [90.0, 0.0, -90.0])
    rotate(saddle, axis=(0.0, 0.0, 1.0), angle_deg=180.0)
    translate(saddle, (-22.0, 0.0, 0.0))
    return saddle


if __name__ == "__main__":
    build()
