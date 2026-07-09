from __future__ import annotations

import sys
from pathlib import Path

from impression.modeling import loft, make_circle
from impression.modeling.drawing2d import PlanarShape2D

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _common import make_label


def _ring(outer: float, inner: float, color: str) -> PlanarShape2D:
    outer_path = make_circle(radius=outer, color=color).outer
    inner_path = make_circle(radius=inner).outer
    return PlanarShape2D(outer=outer_path, holes=[inner_path]).with_color(color)


def build():
    sections = [
        _ring(outer=12.0, inner=10.2, color="#a6c7b2"),
        _ring(outer=10.8, inner=9.0, color="#a6c7b2"),
        _ring(outer=9.6, inner=7.8, color="#a6c7b2"),
        _ring(outer=7.2, inner=5.8, color="#a6c7b2"),
        _ring(outer=4.3, inner=2.7, color="#a6c7b2"),
    ]
    bottle = loft(
        sections,
        samples=120,
        cap_ends=True,
        start_cap="flat",
        end_cap="dome",
        end_cap_length=2.2,
        cap_steps=14,
    )
    label = make_label("BOTTLE TO NOZZLE", center=(0.0, 9.5, 2.0), font_size=6.8)
    return [bottle, label]
