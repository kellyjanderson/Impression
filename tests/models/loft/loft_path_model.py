"""Loft along a path (test model)."""

from __future__ import annotations

from impression.modeling import loft, Path3D
from impression.modeling.drawing2d import make_rect


def build():
    profiles = [
        make_rect(size=(1.0, 0.6)),
        make_rect(size=(0.7, 0.9)),
        make_rect(size=(0.5, 0.5)),
    ]
    path = Path3D.from_points([(0, 0, 0), (0.4, 0.0, 0.8), (0.4, 0.6, 1.6)])
    return loft(profiles, path=path)


if __name__ == "__main__":
    build()
