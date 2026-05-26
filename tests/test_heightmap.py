from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from impression.modeling import (
    HeightmapAlphaMaskPolicy,
    HeightmapCacheKeyRecord,
    HeightmapProjectionBoundsPolicy,
    HeightmapSampleCoordinateRecord,
    HeightmapSurfacePatch,
    DisplacementSurfacePatch,
    MeshQuality,
    ParameterDomain,
    PlanarSurfacePatch,
    heightmap,
    displace_heightmap,
    heightmap_cache_key_record,
    heightmap_sample_coordinate_record,
    make_surface_body,
    make_surface_shell,
    make_heightmap_surface_patch,
    resolve_heightmap_alpha_mask_policy,
    resolve_heightmap_projection_bounds_policy,
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


def test_displace_heightmap_surface_backend_uses_displacement_payload():
    surface = make_plane(size=(1.0, 1.0), center=(0.0, 0.0, 0.0))
    image = np.ones((2, 2), dtype=float)

    displaced = displace_heightmap(surface, image, height=0.5, plane="xy", direction="z", backend="surface")
    patch = displaced.iter_patches()[0]
    mesh = tessellate_surface_body(displaced).mesh
    patch_mesh = tessellate_surface_patch(patch)

    assert isinstance(patch, DisplacementSurfacePatch)
    assert patch.source_patch.family == "planar"
    assert patch.height_scale == 0.5
    assert mesh.n_faces > 0
    assert patch_mesh.metadata["heightmap_displacement_boundary"] == "surface-payload"
    assert np.isclose(mesh.bounds[5] - mesh.bounds[4], 0.0, atol=1e-9)
    assert np.isclose(mesh.bounds[4], 0.5, atol=1e-9)


def test_heightmap_projection_policy_resolves_planes_and_bounds():
    xy = resolve_heightmap_projection_bounds_policy(plane="xy", source_bounds=(0.0, 2.0, 10.0, 14.0, -1.0, 1.0))
    xz = resolve_heightmap_projection_bounds_policy(plane="xz", source_bounds=(0.0, 2.0, 10.0, 14.0, -1.0, 1.0))
    yz = resolve_heightmap_projection_bounds_policy(plane="yz", bounds=(10.0, 14.0, -1.0, 1.0))
    record = heightmap_sample_coordinate_record(
        np.asarray([[1.0, 12.0, 0.0]], dtype=float),
        yz,
    )

    assert isinstance(xy, HeightmapProjectionBoundsPolicy)
    assert xy.bounds == (0.0, 2.0, 10.0, 14.0)
    assert xz.bounds == (0.0, 2.0, -1.0, 1.0)
    assert yz.source == "explicit"
    assert isinstance(record, HeightmapSampleCoordinateRecord)
    assert np.allclose(record.u_normalized, [0.5])
    assert np.allclose(record.v_normalized, [0.5])
    assert record.canonical_payload()["sample_count"] == 1


def test_heightmap_projection_policy_refuses_degenerate_and_unsupported_projection():
    with pytest.raises(ValueError, match="projection bounds are degenerate"):
        resolve_heightmap_projection_bounds_policy(bounds=(0.0, 0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="Only planar projection"):
        resolve_heightmap_projection_bounds_policy(projection="cylindrical", bounds=(0.0, 1.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="plane must be"):
        resolve_heightmap_projection_bounds_policy(plane="uv", bounds=(0.0, 1.0, 0.0, 1.0))


def test_displace_heightmap_surface_projection_planes_and_explicit_bounds():
    source_patch = PlanarSurfacePatch(
        family="planar",
        domain=ParameterDomain((0.0, 1.0), (0.0, 1.0)),
        origin=np.asarray([0.0, 0.0, 0.0], dtype=float),
        u_axis=np.asarray([1.0, 0.0, 0.0], dtype=float),
        v_axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
    )
    surface = make_surface_body((make_surface_shell((source_patch,), connected=False),))
    image = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float)

    displaced = displace_heightmap(
        surface,
        image,
        height=0.25,
        plane="xz",
        direction="y",
        backend="surface",
    )
    patch = displaced.iter_patches()[0]

    assert isinstance(patch, DisplacementSurfacePatch)
    assert patch.plane == "xz"
    assert patch.projection_bounds == (0.0, 1.0, 0.0, 1.0)
    assert patch.kernel_metadata()["projection_policy"]["plane"] == "xz"
    assert np.isclose(patch.point_at(0.5, 0.0)[1], 0.25)
    assert np.isclose(patch.point_at(0.5, 1.0)[1], 0.0)


def test_displace_heightmap_surface_projection_bounds_refusal_and_mask():
    surface = make_plane(size=(1.0, 1.0), center=(0.0, 0.0, 0.0))
    image = np.ones((2, 2, 4), dtype=np.uint8) * 255
    image[0, 0, 3] = 0

    with pytest.raises(ValueError, match="projection bounds are degenerate"):
        displace_heightmap(
            surface,
            image,
            height=0.5,
            plane="xz",
            direction="z",
            backend="surface",
        )

    masked = displace_heightmap(
        surface,
        image,
        height=0.5,
        plane="xz",
        direction="z",
        bounds=(-0.5, 0.5, -0.5, 0.5),
        alpha_mode="mask",
        backend="surface",
    )
    ignored = displace_heightmap(
        surface,
        image,
        height=0.5,
        plane="xz",
        direction="z",
        bounds=(-0.5, 0.5, -0.5, 0.5),
        alpha_mode="ignore",
        backend="surface",
    )
    mesh = tessellate_surface_body(masked).mesh

    assert masked.iter_patches()[0].alpha_mode == "mask"
    assert mesh.n_faces < tessellate_surface_body(ignored).mesh.n_faces


def test_displace_heightmap_surface_backend_rejects_mesh_input() -> None:
    mesh = make_plane(size=(1.0, 1.0), center=(0.0, 0.0, 0.0), backend="mesh")

    with pytest.raises(ValueError, match="Surface displacement requires a SurfaceBody"):
        displace_heightmap(mesh, np.ones((2, 2), dtype=float), backend="surface")
