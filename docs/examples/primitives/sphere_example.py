"""Sphere primitive demo."""

from __future__ import annotations

from pathlib import Path

from impression.modeling import make_sphere

OUTPUT = Path("dist")
OUTPUT.mkdir(exist_ok=True)

sphere = make_sphere(radius=0.75, center=(0.5, 0.5, 0.75))
sphere.save(OUTPUT / "sphere_example.stl")
print("Saved sphere_example.stl with", sphere.n_cells, "cells")
