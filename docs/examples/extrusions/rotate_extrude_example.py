"""Rotate extrude (lathe) example."""

from __future__ import annotations

from impression.modeling import rotate_extrude
from impression.modeling.drawing2d import make_polygon


def build():
    profile = make_polygon(
        [
            (0.4, -0.6),
            (0.8, -0.2),
            (0.7, 0.3),
            (0.4, 0.6),
            (0.2, 0.2),
        ],
        color="#ff7a18",
    )
    return rotate_extrude(profile, angle_deg=360)
