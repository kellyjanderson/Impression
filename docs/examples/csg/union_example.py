"""CSG union example."""

from __future__ import annotations

from pathlib import Path

from impression.modeling import boolean_union, make_box, make_cylinder

OUTPUT = Path("dist")
OUTPUT.mkdir(exist_ok=True)

box = make_box(size=(2, 2, 1))
cyl = make_cylinder(radius=0.6, height=1.5)
union = boolean_union([box, cyl])
union.save(OUTPUT / "union_example.stl")
print("Saved union_example.stl with", union.n_cells, "cells")
