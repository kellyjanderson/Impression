"""CSG intersection example."""

from __future__ import annotations

from pathlib import Path

from impression.modeling import boolean_intersection, make_box, make_sphere


def build():
    box = make_box(size=(2, 2, 2), color=(0.35, 0.55, 0.95))
    sphere = make_sphere(radius=1.2, color=(1.0, 0.55, 0.2))
    return boolean_intersection([box, sphere])


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    mesh.save(OUTPUT / "intersection_example.stl")
    print("Saved intersection_example.stl with", mesh.n_cells, "cells")
