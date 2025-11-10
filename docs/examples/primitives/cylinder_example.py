"""Cylinder primitive demo."""

from __future__ import annotations

from pathlib import Path

from impression.modeling import make_cylinder

OUTPUT = Path("dist")
OUTPUT.mkdir(exist_ok=True)

cylinder = make_cylinder(radius=0.6, height=1.5, center=(0, 0, 0.75))
cylinder.save(OUTPUT / "cylinder_example.stl")
print("Saved cylinder_example.stl with", cylinder.n_cells, "cells")
