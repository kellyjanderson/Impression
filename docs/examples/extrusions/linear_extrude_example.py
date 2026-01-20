"""Linear extrude example."""

from __future__ import annotations

from impression.modeling import linear_extrude
from impression.modeling.drawing2d import Arc2D, Path2D, Profile2D


def build():
    outer = Path2D([Arc2D(center=(0.0, 0.0), radius=1.0, start_angle_deg=0, end_angle_deg=360)], closed=True)
    inner = Path2D(
        [Arc2D(center=(0.0, 0.0), radius=0.45, start_angle_deg=0, end_angle_deg=360, clockwise=True)],
        closed=True,
    )
    profile = Profile2D(outer=outer, holes=[inner]).with_color("#5a7bff")
    return linear_extrude(profile, height=1.0)
