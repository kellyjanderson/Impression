"""Regular polyhedron example."""

from __future__ import annotations

from impression.modeling import make_polyhedron


def build():
    return make_polyhedron(faces=12, radius=1.0, color=(0.9, 0.65, 0.2, 1.0))
