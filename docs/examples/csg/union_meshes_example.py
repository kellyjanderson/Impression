"""Union multiple meshes via union_meshes helper.

Run with:
  impression preview docs/examples/csg/union_meshes_example.py
"""

from __future__ import annotations

from pathlib import Path

from impression.modeling import make_box, make_cylinder, union_meshes


def build():
    box = make_box(size=(2, 2, 1), color=(0.35, 0.55, 0.95))
    cyl = make_cylinder(radius=0.8, height=1.5, color=(1.0, 0.6, 0.2))
    # union_meshes accepts any iterable or mapping
    return union_meshes({"box": box, "cyl": cyl})


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    mesh.save(OUTPUT / "union_meshes_example.stl")
    print("Saved union_meshes_example.stl with", mesh.n_cells, "cells")
