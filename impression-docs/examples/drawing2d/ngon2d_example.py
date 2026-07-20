"""N-gon profile example."""

from __future__ import annotations

from impression.modeling.drawing2d import make_ngon


def build():
    return make_ngon(sides=5, radius=1.0, center=(0.0, 0.0), color="#8b5cf6")
