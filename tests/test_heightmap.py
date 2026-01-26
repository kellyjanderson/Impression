from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from impression.modeling import heightmap, displace_heightmap
from impression.modeling.drafting import make_plane


def _write_test_image(path: Path, transparent_corner: bool = False) -> Path:
    arr = np.ones((2, 2, 4), dtype=np.uint8) * 255
    if transparent_corner:
        arr[0, 0, 3] = 0
    img = Image.fromarray(arr, mode="RGBA")
    img.save(path)
    return path


def test_heightmap_alpha_mask(tmp_path: Path):
    path = _write_test_image(tmp_path / "mask.png", transparent_corner=True)

    masked = heightmap(path, height=1.0, alpha_mode="mask")
    assert masked.n_faces == 0

    ignored = heightmap(path, height=1.0, alpha_mode="ignore")
    assert ignored.n_faces == 2
    assert np.isclose(ignored.vertices[0, 2], 0.0)


def test_displace_heightmap_planar():
    mesh = make_plane(size=(1.0, 1.0), center=(0.0, 0.0, 0.0))
    image = np.ones((2, 2), dtype=float)

    displaced = displace_heightmap(mesh, image, height=0.5, plane="xy", direction="z")
    assert np.allclose(displaced.vertices[:, 2], mesh.vertices[:, 2] + 0.5)

    arr = np.ones((2, 2, 4), dtype=np.uint8) * 255
    arr[0, 0, 3] = 0
    masked = displace_heightmap(mesh, arr, height=0.5, plane="xy", direction="z", alpha_mode="mask")
    assert masked.n_faces < mesh.n_faces
