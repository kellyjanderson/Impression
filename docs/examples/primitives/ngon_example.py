"""Regular n-gon prism example."""

from __future__ import annotations

from impression.modeling import make_ngon


def build():
    return make_ngon(sides=7, radius=1.0, height=0.8, color=(0.3, 0.6, 0.9, 1.0))
