"""CSG union example."""

from __future__ import annotations

from pathlib import Path

from impression.modeling import boolean_union, make_box, make_cylinder


def build():
    box = make_box(size=(2, 2, 1))
    cyl = make_cylinder(radius=0.6, height=1.5)
    return boolean_union([box, cyl])


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    mesh.save(OUTPUT / "union_example.stl")
    print("Saved union_example.stl with", mesh.n_cells, "cells")
