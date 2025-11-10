"""CSG difference example."""

from __future__ import annotations

from pathlib import Path

from impression.modeling import boolean_difference, make_box, make_cylinder

OUTPUT = Path("dist")
OUTPUT.mkdir(exist_ok=True)

base = make_box(size=(2, 2, 2))
cutter = make_cylinder(radius=0.4, height=2.5)
result = boolean_difference(base, [cutter])
result.save(OUTPUT / "difference_example.stl")
print("Saved difference_example.stl with", result.n_cells, "cells")
