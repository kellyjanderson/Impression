"""Rectangle profile example."""

from __future__ import annotations

from impression.modeling.drawing2d import make_rect


def build():
    return make_rect(size=(2.0, 1.0), center=(0.0, 0.0), color="#5a7bff")
