"""Torus primitive demo."""

from __future__ import annotations

from pathlib import Path

from impression.io import write_stl

from impression.modeling import make_torus


def build():
    return make_torus(major_radius=1.25, minor_radius=0.35, center=(0, 0, 0))


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    write_stl(mesh, OUTPUT / "torus_example.stl")
    print("Saved torus_example.stl with", mesh.n_faces, "faces")
