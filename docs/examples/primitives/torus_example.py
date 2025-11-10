"""Torus primitive demo."""

from __future__ import annotations

from pathlib import Path

from impression.modeling import make_torus

OUTPUT = Path("dist")
OUTPUT.mkdir(exist_ok=True)

torus = make_torus(major_radius=1.25, minor_radius=0.35, center=(0, 0, 0))
torus.save(OUTPUT / "torus_example.stl")
print("Saved torus_example.stl with", torus.n_cells, "cells")
