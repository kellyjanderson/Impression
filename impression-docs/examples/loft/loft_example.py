"""Loft example."""

from __future__ import annotations

from impression.modeling import Path3D, loft
from impression.modeling.drawing2d import make_rect


def build():
    profiles = [
        make_rect(size=(1.2, 1.2)),
        make_rect(size=(0.6, 1.8)),
        make_rect(size=(1.0, 0.8)),
    ]
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.2, 0.0, 1.0), (-0.1, 0.0, 2.0)])
    return loft(profiles, path=path, cap_ends=True)
