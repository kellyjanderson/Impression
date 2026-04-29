"""Path2D example with mixed segments."""

from __future__ import annotations

from impression.modeling.drawing2d import Arc2D, Line2D, Path2D


def build():
    segments = [
        Line2D(start=(-1.0, 0.0), end=(-0.2, 0.6)),
        Arc2D(center=(0.3, 0.2), radius=0.7, start_angle_deg=160, end_angle_deg=-40),
        Line2D(start=(0.8, -0.2), end=(1.2, 0.5)),
    ]
    return Path2D(segments, closed=False).with_color("#6ab0ff")
