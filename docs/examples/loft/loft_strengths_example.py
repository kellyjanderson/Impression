"""Loft strengths: multiple profiles + path-driven orientation."""

from __future__ import annotations

from impression.modeling import loft, Path3D
from impression.modeling.drawing2d import make_rect, make_circle


def build():
    profiles = [
        make_rect(size=(1.0, 0.6)),
        make_circle(radius=0.4),
        make_rect(size=(0.5, 0.9)),
    ]
    path = Path3D.from_points(
        [
            (0.0, 0.0, 0.0),
            (0.3, 0.0, 0.8),
            (0.6, 0.3, 1.6),
            (0.6, 0.6, 2.2),
        ]
    )
    return loft(profiles, path=path)


if __name__ == "__main__":
    build()
