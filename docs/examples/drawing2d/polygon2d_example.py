"""Polygon profile example."""

from __future__ import annotations

from impression.modeling.drawing2d import make_polygon


def build():
    points = [(-1.2, -0.6), (0.2, -1.0), (1.0, -0.2), (0.6, 0.9), (-0.8, 0.8)]
    return make_polygon(points, color="#f6c343")
