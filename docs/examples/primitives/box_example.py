"""Generate a box primitive and save it for inspection."""

from __future__ import annotations

from pathlib import Path

from impression.modeling import make_box

OUTPUT = Path("dist")
OUTPUT.mkdir(exist_ok=True)

mesh = make_box(size=(2.0, 1.0, 0.5), center=(0.0, 0.0, 0.25))
mesh.save(OUTPUT / "box_example.stl")
print("Saved box_example.stl with", mesh.n_cells, "cells")
