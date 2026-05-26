from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from impression.modeling import HeightmapSurfacePatch, MeshQuality, heightmap, displace_heightmap, make_heightmap_surface_patch
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


def test_heightmap_quality_preview(tmp_path: Path):
    path = _write_test_image(tmp_path / "mask.png", transparent_corner=False)
    mesh = heightmap(path, height=1.0, alpha_mode="ignore", quality=MeshQuality(lod="preview"))
    assert mesh.n_faces >= 0


def test_heightmap_surface_backend_uses_sampled_surface_payload():
    samples = np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float)

    patch = make_heightmap_surface_patch(samples, height=2.0, xy_scale=(2.0, 3.0))
    body = heightmap(samples, height=2.0, xy_scale=(2.0, 3.0), backend="surface")

    assert isinstance(patch, HeightmapSurfacePatch)
    assert patch.height_samples.shape == (2, 2)
    assert patch.point_at(1.0, 0.0)[2] == 2.0
    assert body.patch_count == 1
    assert isinstance(body.iter_patches()[0], HeightmapSurfacePatch)


def test_displace_heightmap_planar():
    mesh = make_plane(size=(1.0, 1.0), center=(0.0, 0.0, 0.0), backend="mesh")
    image = np.ones((2, 2), dtype=float)

    displaced = displace_heightmap(mesh, image, height=0.5, plane="xy", direction="z")
    assert np.allclose(displaced.vertices[:, 2], mesh.vertices[:, 2] + 0.5)

    arr = np.ones((2, 2, 4), dtype=np.uint8) * 255
    arr[0, 0, 3] = 0
    masked = displace_heightmap(mesh, arr, height=0.5, plane="xy", direction="z", alpha_mode="mask")
    assert masked.n_faces < mesh.n_faces
