"""Box primitive example compatible with `impression preview`."""

from __future__ import annotations

from pathlib import Path

from impression.modeling import make_box


def build():
    """Return a simple box mesh."""

    return make_box(size=(2.0, 1.0, 0.5), center=(0.0, 0.0, 0.25))


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    mesh.save(OUTPUT / "box_example.stl")
    print("Saved box_example.stl with", mesh.n_cells, "cells")
