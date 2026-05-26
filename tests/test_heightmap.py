from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from impression.modeling import (
    HeightmapAlphaMaskPolicy,
    HeightmapCacheKeyRecord,
    HeightmapSurfacePatch,
    MeshQuality,
    heightmap,
    displace_heightmap,
    heightmap_cache_key_record,
    make_heightmap_surface_patch,
    resolve_heightmap_alpha_mask_policy,
    tessellate_surface_body,
    tessellate_surface_patch,
)
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


def test_heightmap_surface_alpha_policy_and_tessellation_preserve_mask(tmp_path: Path):
    path = _write_test_image(tmp_path / "mask.png", transparent_corner=True)

    masked_body = heightmap(path, height=1.0, alpha_mode="mask", backend="surface")
    ignored_body = heightmap(path, height=1.0, alpha_mode="ignore", backend="surface")
    masked_patch = masked_body.iter_patches()[0]
    ignored_patch = ignored_body.iter_patches()[0]
    masked_mesh = tessellate_surface_body(masked_body).mesh
    ignored_mesh = tessellate_surface_body(ignored_body).mesh
    ignored_patch_mesh = tessellate_surface_patch(ignored_patch)

    assert isinstance(masked_patch, HeightmapSurfacePatch)
    assert isinstance(ignored_patch, HeightmapSurfacePatch)
    assert masked_patch.alpha_mode == "mask"
    assert ignored_patch.alpha_mode == "ignore"
    assert masked_patch.kernel_metadata()["alpha_policy"]["masked_sample_count"] == 1
    assert masked_patch.kernel_metadata()["cache_key_policy"]["cacheable"] is True
    assert masked_mesh.n_faces == 0
    assert ignored_mesh.n_faces == 2
    assert ignored_patch_mesh.metadata["heightmap_alpha_mode"] == "ignore"


def test_heightmap_alpha_and_cache_policy_records_are_explicit(tmp_path: Path):
    path = _write_test_image(tmp_path / "cache.png", transparent_corner=True)
    mask = np.asarray([[False, True], [True, True]], dtype=bool)

    policy = resolve_heightmap_alpha_mask_policy(mask, alpha_mode="mask")
    cache_record = heightmap_cache_key_record(path, 1.0, 1.0, (0.0, 0.0, 0.0), "mask", None)
    array_cache_record = heightmap_cache_key_record(mask.astype(float), 1.0, 1.0, (0.0, 0.0, 0.0), "mask", None)

    assert isinstance(policy, HeightmapAlphaMaskPolicy)
    assert policy.has_masked_samples is True
    assert policy.canonical_payload()["masked_sample_count"] == 1
    assert isinstance(cache_record, HeightmapCacheKeyRecord)
    assert cache_record.cacheable is True
    assert array_cache_record.cacheable is False
    assert array_cache_record.reason == "uncacheable-source"


def test_displace_heightmap_planar():
    mesh = make_plane(size=(1.0, 1.0), center=(0.0, 0.0, 0.0), backend="mesh")
    image = np.ones((2, 2), dtype=float)

    displaced = displace_heightmap(mesh, image, height=0.5, plane="xy", direction="z")
    assert np.allclose(displaced.vertices[:, 2], mesh.vertices[:, 2] + 0.5)

    arr = np.ones((2, 2, 4), dtype=np.uint8) * 255
    arr[0, 0, 3] = 0
    masked = displace_heightmap(mesh, arr, height=0.5, plane="xy", direction="z", alpha_mode="mask")
    assert masked.n_faces < mesh.n_faces
