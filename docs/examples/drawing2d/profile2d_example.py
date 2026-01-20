"""Profile2D example with a hole."""

from __future__ import annotations

from impression.modeling.drawing2d import Arc2D, Path2D, Profile2D


def build():
    outer = Path2D([Arc2D(center=(0.0, 0.0), radius=1.2, start_angle_deg=0, end_angle_deg=360)], closed=True)
    inner = Path2D(
        [Arc2D(center=(0.0, 0.0), radius=0.5, start_angle_deg=0, end_angle_deg=360, clockwise=True)],
        closed=True,
    )
    return Profile2D(outer=outer, holes=[inner]).with_color("#7fbf7f")
