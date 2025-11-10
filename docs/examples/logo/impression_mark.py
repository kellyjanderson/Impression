"""Impression mark: cube with embossed 👁️ on all faces."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from impression.modeling import make_box, make_text
from impression.modeling.csg import boolean_difference

BRAND_BLUE = "#5A7BFF"
EMOJI = "👁️"


def _face_text(normal: np.ndarray, depth: float, size: float) -> object:
    text = make_text(EMOJI, depth=depth, font_size=size, direction=normal, color="#FF7A18")
    offset = normal * (0.5 - depth / 2.0)
    text.translate(offset, inplace=True)
    return text


def build():
    cube = make_box(size=(1, 1, 1), color=BRAND_BLUE)
    depth = 0.08
    size = 0.35
    normals = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    cutters = [_face_text(normal, depth, size) for normal in normals]
    mark = boolean_difference(cube, cutters)
    return mark


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    mesh.save(OUTPUT / "impression_mark.stl")
    print("Saved impression_mark.stl with", mesh.n_cells, "cells")
