"""Polyline path example."""

from __future__ import annotations

from impression.modeling.drawing2d import make_polyline


def build():
    points = [(-1.5, -0.4), (-0.4, 0.6), (0.6, -0.2), (1.4, 0.8)]
    return make_polyline(points, closed=False, color="#9aa6bf")
