"""Surface CSG intersection example."""

from __future__ import annotations

from pathlib import Path

from impression.io import write_stl
from impression.modeling import (
    boolean_intersection,
    export_tessellation_request,
    make_box,
    tessellate_surface_body,
)


def build():
    left = make_box(size=(2, 2, 2), center=(-0.5, 0, 0), color=(0.35, 0.55, 0.95))
    right = make_box(size=(2, 2, 2), center=(0.5, 0, 0), color=(1.0, 0.55, 0.2))
    result = boolean_intersection([left, right])
    if result.status != "succeeded" or result.body is None:
        raise RuntimeError(result.failure_reason or "Surface intersection did not produce a body.")
    return result.body


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = tessellate_surface_body(build(), export_tessellation_request()).mesh
    write_stl(mesh, OUTPUT / "intersection_example.stl")
    print("Saved intersection_example.stl with", mesh.n_faces, "faces")
