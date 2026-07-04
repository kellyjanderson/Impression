"""Bezier2D example."""

from __future__ import annotations

from impression.modeling.drawing2d import Bezier2D, Path2D


def build():
    curve = Bezier2D(
        p0=(-1.5, -0.5),
        p1=(-0.5, 1.2),
        p2=(0.5, -1.2),
        p3=(1.5, 0.5),
    )
    return Path2D([curve], closed=False).with_color("#ff7a18")
