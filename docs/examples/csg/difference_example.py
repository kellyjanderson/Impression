"""CSG difference example."""

from __future__ import annotations

from pathlib import Path

from impression.modeling import boolean_difference, make_box, make_cylinder


def build():
    base = make_box(size=(2, 2, 2), color="#5A7BFF")
    cutter = make_cylinder(radius=0.4, height=2.5, color="#FF7A18")
    return boolean_difference(base, [cutter])


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    mesh.save(OUTPUT / "difference_example.stl")
    print("Saved difference_example.stl with", mesh.n_cells, "cells")
