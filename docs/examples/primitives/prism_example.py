"""Prism/pyramid example."""

from __future__ import annotations

from impression.modeling import make_prism


def build():
    return make_prism(base_size=(1.5, 1.0), top_size=(0.3, 0.6), height=1.8)


if __name__ == "__main__":
    mesh = build()
    print("Cells:", mesh.n_cells)
