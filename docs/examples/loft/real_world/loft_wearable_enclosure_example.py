from __future__ import annotations

import sys
from pathlib import Path

from impression.modeling import Bezier3D, Path3D, loft, make_circle, make_rect
from impression.modeling.drawing2d import PlanarShape2D

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _common import make_label


def _rounded_rect_with_port(
    width: float,
    height: float,
    port_radius: float,
    color: str,
) -> PlanarShape2D:
    outer = make_rect(size=(width, height), color=color).outer
    port = make_circle(radius=port_radius, center=(width * 0.2, 0.0)).outer
    return PlanarShape2D(outer=outer, holes=[port]).with_color(color)


def build():
    s0 = _rounded_rect_with_port(24.0, 16.0, 2.1, "#93bca9")
    s1 = _rounded_rect_with_port(22.0, 14.5, 2.1, "#93bca9")
    s2 = _rounded_rect_with_port(18.0, 12.0, 1.8, "#93bca9")
    s3 = _rounded_rect_with_port(14.0, 9.2, 1.3, "#93bca9")

    path = Path3D(
        [
            Bezier3D(
                p0=(0.0, 0.0, 0.0),
                p1=(3.5, 0.8, 8.0),
                p2=(5.2, -0.5, 16.0),
                p3=(7.0, 0.0, 24.0),
            )
        ]
    )

    enclosure = loft(
        [s0, s1, s2, s3],
        path=path,
        samples=96,
        cap_ends=True,
        start_cap="flat",
        end_cap="slope",
        end_cap_length=2.0,
        cap_steps=12,
        cap_scale_dims="both",
    )
    label = make_label("WEARABLE ENCLOSURE SHELL", center=(7.0, 10.5, 12.0), font_size=6.0)
    return [enclosure, label]
