"""Heightmap example with masking.

Run with:
  impression preview docs/examples/heightmaps/heightmap_basic.py
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from impression.modeling import heightmap


def build():
    size = 96
    coords = np.linspace(-1.0, 1.0, size)
    xv, yv = np.meshgrid(coords, coords)
    r = np.sqrt(xv**2 + yv**2)
    height = np.clip(1.0 - r, 0.0, 1.0)
    alpha = (r <= 1.0).astype(float)
    rgba = np.stack([
        (height * 255),
        (height * 255),
        (height * 255),
        (alpha * 255),
    ], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")
    return heightmap(img, height=3.0, xy_scale=0.08, alpha_mode="mask")


if __name__ == "__main__":
    build()
