"""Arc2D example."""

from __future__ import annotations

from impression.modeling.drawing2d import Arc2D, Path2D


def build():
    arc = Arc2D(center=(0.0, 0.0), radius=1.2, start_angle_deg=20, end_angle_deg=300)
    path = Path2D([arc], closed=False).with_color("#5a7bff")
    return path
