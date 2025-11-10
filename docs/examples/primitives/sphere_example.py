"""Sphere primitive demo."""

from __future__ import annotations

from pathlib import Path

from impression.modeling import make_sphere


def build():
    return make_sphere(radius=0.75, center=(0.5, 0.5, 0.75))


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    mesh.save(OUTPUT / "sphere_example.stl")
    print("Saved sphere_example.stl with", mesh.n_cells, "cells")
