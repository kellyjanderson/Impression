"""Cylinder primitive demo usable with `impression preview`."""

from __future__ import annotations

from pathlib import Path

from impression.io import write_stl

from impression.modeling import make_cylinder


def build():
    return make_cylinder(radius=0.6, height=1.5, center=(0, 0, 0.75))


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    write_stl(mesh, OUTPUT / "cylinder_example.stl")
    print("Saved cylinder_example.stl with", mesh.n_faces, "faces")
