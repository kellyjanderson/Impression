"""CSG union example."""

from __future__ import annotations

from pathlib import Path

from impression.io import write_stl

from impression.modeling import boolean_union, make_box_mesh, make_cylinder_mesh


def build():
    box = make_box_mesh(size=(2, 2, 1), color=(0.35, 0.55, 0.95))
    cyl = make_cylinder_mesh(radius=0.6, height=1.5, color=(1.0, 0.6, 0.2))
    return boolean_union([box, cyl])


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    write_stl(mesh, OUTPUT / "union_example.stl")
    print("Saved union_example.stl with", mesh.n_faces, "faces")
