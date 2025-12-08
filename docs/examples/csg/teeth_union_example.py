"""Stress-test union of multiple aligned meshes (teeth-like layout).

Run with:
  impression preview docs/examples/csg/teeth_union_example.py
"""

from __future__ import annotations

import math
from pathlib import Path

from impression.modeling import make_cylinder, union_meshes


def build():
    meshes = {}
    radius = 4.0
    tooth_radius = 0.5
    tooth_height = 2.0
    count = 8
    span_deg = 180.0
    start = -span_deg / 2.0
    step = span_deg / max(count - 1, 1)

    for i in range(count):
        angle = math.radians(start + i * step)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        cyl = make_cylinder(radius=tooth_radius, height=tooth_height, center=(x, y, tooth_height / 2.0))
        meshes[f"tooth-{i}"] = cyl

    return union_meshes(meshes)


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    mesh.save(OUTPUT / "teeth_union_example.stl")
    print("Saved teeth_union_example.stl with", mesh.n_cells, "cells")
