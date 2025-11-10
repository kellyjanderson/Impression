"""CSG intersection example."""

from __future__ import annotations

from pathlib import Path

from impression.modeling import boolean_intersection, make_box, make_sphere

OUTPUT = Path("dist")
OUTPUT.mkdir(exist_ok=True)

box = make_box(size=(2, 2, 2))
sphere = make_sphere(radius=1.2)
result = boolean_intersection([box, sphere])
result.save(OUTPUT / "intersection_example.stl")
print("Saved intersection_example.stl with", result.n_cells, "cells")
