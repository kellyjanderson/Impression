"""2D drawing primitives example."""

from __future__ import annotations

from impression.modeling.drawing2d import (
    Arc2D,
    Path2D,
    Profile2D,
    make_circle,
    make_ngon,
    make_polyline,
    make_rect,
)


def build():
    rect = make_rect(size=(2.0, 1.2), center=(-1.6, 0.0), color="#5a7bff")
    ngon = make_ngon(sides=6, radius=0.6, center=(1.2, 0.0), color="#ff7a18")

    outer = Path2D([Arc2D(center=(0.0, 1.6), radius=0.7, start_angle_deg=0, end_angle_deg=360)], closed=True)
    inner = Path2D(
        [Arc2D(center=(0.0, 1.6), radius=0.3, start_angle_deg=0, end_angle_deg=360, clockwise=True)],
        closed=True,
    )
    ring = Profile2D(outer=outer, holes=[inner]).with_color("#7fbf7f")

    path = make_polyline([(-2.2, -1.4), (-0.4, -1.1), (0.6, -1.6), (2.0, -1.1)], closed=False)
    path.with_color("#9aa6bf")

    outline = make_polyline([(-0.6, -0.6), (0.6, -0.6), (0.8, -1.4), (-0.8, -1.4)], closed=True)
    outline.with_color("#c9d1ff")

    return [rect, ngon, ring, path, outline]
