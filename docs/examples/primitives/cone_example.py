"""Cone/frustum example."""

from __future__ import annotations

from impression.modeling import make_cone


def build():
    return make_cone(bottom_diameter=1.5, top_diameter=0.4, height=2.0)


if __name__ == "__main__":
    mesh = build()
    print("Cells:", mesh.n_cells)
