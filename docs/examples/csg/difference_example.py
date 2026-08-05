"""Surface CSG difference example."""

from __future__ import annotations

from pathlib import Path

from impression.io import write_stl
from impression.modeling import (
    boolean_difference,
    export_tessellation_request,
    make_box,
    tessellate_surface_body,
)


def build():
    base = make_box(size=(3, 2, 2), color="#5A7BFF")
    cutter = make_box(size=(1.5, 1, 3), center=(1, 0, 0), color="#FF7A18")
    result = boolean_difference(base, [cutter])
    if result.status not in {"succeeded", "no-cut"} or result.body is None:
        raise RuntimeError(result.failure_reason or "Surface difference did not produce a body.")
    return result.body


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = tessellate_surface_body(build(), export_tessellation_request()).mesh
    write_stl(mesh, OUTPUT / "difference_example.stl")
    print("Saved difference_example.stl with", mesh.n_faces, "faces")
