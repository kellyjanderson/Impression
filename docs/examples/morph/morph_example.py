"""Profile morph example."""

from __future__ import annotations

from impression.modeling import morph
from impression.modeling.drawing2d import make_circle, make_ngon


def build():
    start = make_circle(radius=1.0, color="#5a7bff")
    end = make_ngon(sides=6, radius=1.0, color="#ff7a18")
    return morph(start, end, t=0.5)
