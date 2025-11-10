"""Impression mark: cube with embossed 👁️ on all faces."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from impression.modeling import make_box, make_cylinder, make_text
from impression.modeling.csg import boolean_difference

BRAND_BLUE = "#5A7BFF"
BRAND_ORANGE = "#FF7A18"
EMOJI = "👁️"


def _emoji_cutter(normal: np.ndarray, depth: float, size: float):
    text = make_text(EMOJI, depth=depth, font_size=size, direction=normal, color=BRAND_ORANGE)
    text.triangulate(inplace=True)
    if text.n_cells == 0:
        text = _fallback_eye(normal, depth, size)
    offset = normal * (0.5 - depth / 2.0)
    text.translate(offset, inplace=True)
    return text


def _fallback_eye(normal: np.ndarray, depth: float, size: float):
    disc = make_cylinder(radius=size * 0.45, height=depth, center=(0, 0, 0), direction=normal)
    disc.scale((1.6, 1.0, 1.0), inplace=True)
    pupil = make_cylinder(radius=size * 0.18, height=depth * 1.2, center=(0, 0, 0), direction=normal)
    pupil.translate(normal * 0.02, inplace=True)
    return boolean_difference(disc, [pupil])


def build():
    cube = make_box(size=(1, 1, 1), color=BRAND_BLUE)
    depth = 0.12
    size = 0.4
    normals = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    cutters = [_emoji_cutter(normal, depth, size) for normal in normals]
    return boolean_difference(cube, cutters)


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    mesh.save(OUTPUT / "impression_mark.stl")
    print("Saved impression_mark.stl with", mesh.n_cells, "cells")
