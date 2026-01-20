"""Circle profile example."""

from __future__ import annotations

from impression.modeling.drawing2d import make_circle


def build():
    return make_circle(radius=1.0, center=(0.0, 0.0), color="#ff7a18")
