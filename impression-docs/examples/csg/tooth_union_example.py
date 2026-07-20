"""Solid tombstone: box + rounded top, fused via boolean_union.

Run with:
  impression preview docs/examples/csg/tooth_example.py
"""

from __future__ import annotations

from pathlib import Path

from impression.io import write_stl

from impression.modeling import boolean_union, rotate, make_box, make_cylinder


def build():
    body_height = 6.0
    body = make_box(size=(5.0, 2.0, body_height), center=(0.0, 0.0, body_height / 2.0), color=(0.5,0.5,0.5,1.0))

    cap_height = 2.0
    cap = make_cylinder(radius=2.5, height=cap_height, center=(0.0, 0.0, 0.0), color=(0.5,0.5,0.5,1.0))
    rotate(cap, axis=(1.0, 0.0, 0.0), angle_deg=90)
    # Fuse into a single solid; tolerance kept small to avoid masking geometry issues.
    mesh = boolean_union([body, cap], tolerance=1e-4)

    return mesh


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    write_stl(mesh, OUTPUT / "tooth_example.stl")
    print("Saved tooth_example.stl with", mesh.n_faces, "faces")
