"""Line2D example."""

from __future__ import annotations

from impression.modeling.drawing2d import Line2D, Path2D


def build():
    line = Line2D(start=(-1.2, -0.8), end=(1.2, 0.8))
    return Path2D([line], closed=False).with_color("#9aa6bf")
