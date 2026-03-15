from __future__ import annotations

import sys
from pathlib import Path

from impression.modeling import Bezier3D, Path3D, loft, make_circle, make_rect, translate
from impression.modeling.drawing2d import PlanarShape2D

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _common import make_label


def _ring_circle(outer_radius: float, inner_radius: float, color: str) -> PlanarShape2D:
    outer = make_circle(radius=outer_radius, color=color).outer
    inner = make_circle(radius=inner_radius).outer
    return PlanarShape2D(outer=outer, holes=[inner]).with_color(color)


def _ring_rect(outer_size: tuple[float, float], wall: float, color: str) -> PlanarShape2D:
    outer = make_rect(size=outer_size, color=color).outer
    inner = make_rect(size=(outer_size[0] - 2.0 * wall, outer_size[1] - 2.0 * wall)).outer
    return PlanarShape2D(outer=outer, holes=[inner]).with_color(color)


def build():
    start = _ring_circle(outer_radius=9.0, inner_radius=7.0, color="#8fb6d9")
    mid = _ring_rect(outer_size=(17.0, 13.0), wall=2.0, color="#8fb6d9")
    end = _ring_rect(outer_size=(24.0, 14.0), wall=2.0, color="#8fb6d9")

    path = Path3D(
        [
            Bezier3D(
                p0=(0.0, 0.0, 0.0),
                p1=(4.0, 2.0, 7.0),
                p2=(7.0, -2.5, 16.0),
                p3=(10.0, 0.0, 24.0),
            )
        ]
    )

    adapter = loft(
        [start, mid, end],
        path=path,
        samples=96,
        cap_ends=True,
        start_cap="taper",
        end_cap="flat",
        start_cap_length=1.2,
        cap_steps=8,
    )
    translate(adapter, (0.0, 0.0, -8.0))

    title = make_label("HVAC ROUND TO RECT", center=(10.0, 9.0, 16.0), font_size=6.4)
    return [adapter, title]
