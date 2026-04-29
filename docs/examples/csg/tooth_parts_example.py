"""Tombstone parts only (no boolean): box plus cylinder aligned for comparison."""

from __future__ import annotations

from pathlib import Path

from impression.io import write_stl
from impression.modeling import make_box, make_cylinder
from impression.modeling.group import MeshGroup


def build():
    body_height = 6.0
    body = make_box(size=(5.0, 2.0, body_height), center=(0.0, 0.0, body_height / 2.0))

    cap_height = 2.0
    cap = make_cylinder(radius=2.5, height=cap_height, center=(0.0, 0.0, body_height + cap_height / 2.0))

    # Return both solids un-fused so the preview shows overlap and the export keeps both volumes.
    return MeshGroup([body, cap])


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    combined = mesh.to_mesh()
    write_stl(combined, OUTPUT / "tooth_parts_example.stl")
    print("Saved tooth_parts_example.stl with", combined.n_faces, "faces")
