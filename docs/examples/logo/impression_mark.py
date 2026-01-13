"""Impression mark: cube with embossed 👁️ on all faces."""

from __future__ import annotations

from pathlib import Path

from impression.io import write_stl

import numpy as np

from impression.modeling import make_box, make_text
from impression.modeling.csg import boolean_difference

BRAND_BLUE = "#5A7BFF"
FONT_PATH = Path(__file__).resolve().parents[3] / "assets" / "fonts" / "NotoSansSymbols2-Regular.ttf"


def _glyph_cutter(normal: np.ndarray, depth: float, size: float):
    if not FONT_PATH.exists():  # pragma: no cover - runtime safety
        raise FileNotFoundError(f"Emoji font missing: {FONT_PATH}")

    center = normal * (0.5 - depth / 2.0)
    return make_text(
        "👁️",
        depth=depth,
        center=center,
        direction=normal,
        font_size=size,
        font_path=str(FONT_PATH),
        tolerance=0.03,
    )


def build():
    cube = make_box(size=(1, 1, 1), color=BRAND_BLUE)
    depth = 0.12
    size = 0.55
    normals = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    cutters = [_glyph_cutter(normal, depth, size) for normal in normals]
    return boolean_difference(cube, cutters)


if __name__ == "__main__":
    OUTPUT = Path("dist")
    OUTPUT.mkdir(exist_ok=True)
    mesh = build()
    write_stl(mesh, OUTPUT / "impression_mark.stl")
    print("Saved impression_mark.stl with", mesh.n_faces, "faces")
