from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from impression.modeling import (
    HeightmapAlphaMaskPolicy,
    HeightmapCacheKeyRecord,
    HeightmapMeshCompatibilityResult,
    HeightmapClippingRecord,
    HeightmapGridAlignmentRecord,
    HeightmapProjectionDomainRecord,
    HeightmapProjectionRefusalDiagnostic,
    HeightmapProjectionBoundsPolicy,
    HeightmapSampleCoordinateRecord,
    HeightmapSurfacePatch,
    DisplacementSurfacePatch,
    HeightmapCompositionDiagnostic,
    HeightmapCompositionRecord,
    HeightmapCompositionResult,
    HeightmapOverhangDiagnostic,
    HeightmapPromotionDecision,
    HeightmapPromotionDiagnostic,
    HeightmapPromotionTriggerRecord,
    HeightmapRepresentabilityReport,
    MeshQuality,
    ParameterDomain,
    PlanarSurfacePatch,
    compose_heightmap_csg_result,
    heightmap_representability_report,
    plan_heightmap_promotion_route,
    select_heightmap_promotion_target,
    heightmap,
    displace_heightmap,
    heightmap_cache_key_record,
    heightmap_mesh_compatibility_result,
    heightmap_projection_domain_record,
    heightmap_sample_coordinate_record,
    make_surface_body,
    make_surface_shell,
    make_heightmap_surface_patch,
    plan_heightmap_grid_alignment,
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
    masked_mesh = tessellate_surface_body(masked).mesh
    assert masked_mesh.n_faces == 0

    ignored = heightmap(path, height=1.0, alpha_mode="ignore")
    ignored_mesh = tessellate_surface_body(ignored).mesh
    assert ignored_mesh.n_faces == 2
    assert np.isclose(ignored_mesh.vertices[0, 2], 0.0)


def test_heightmap_quality_preview(tmp_path: Path):
    image = np.ones((4, 4), dtype=float)
    body = heightmap(image, height=1.0, alpha_mode="ignore", quality=MeshQuality(lod="preview"))
    assert body.patch_count == 1


def test_heightmap_surface_backend_uses_sampled_surface_payload():
    samples = np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float)

    patch = make_heightmap_surface_patch(samples, height=2.0, xy_scale=(2.0, 3.0))
    body = heightmap(samples, height=2.0, xy_scale=(2.0, 3.0))

    assert isinstance(patch, HeightmapSurfacePatch)
    assert patch.height_samples.shape == (2, 2)
    assert patch.point_at(1.0, 0.0)[2] == 2.0
    assert body.patch_count == 1
    assert isinstance(body.iter_patches()[0], HeightmapSurfacePatch)


def test_heightmap_surface_alpha_policy_and_tessellation_preserve_mask(tmp_path: Path):
    path = _write_test_image(tmp_path / "mask.png", transparent_corner=True)

    masked_body = heightmap(path, height=1.0, alpha_mode="mask")
    ignored_body = heightmap(path, height=1.0, alpha_mode="ignore")
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


def test_heightmap_mesh_compatibility_result_marks_explicit_mesh_boundary():
    image = np.ones((2, 2), dtype=float)

    result = heightmap_mesh_compatibility_result(image, height=0.5, alpha_mode="ignore")

    assert isinstance(result, HeightmapMeshCompatibilityResult)
    assert result.boundary == "explicit-mesh-compatibility"
    assert result.mesh.metadata["heightmap_mesh_compatibility"]["boundary"] == "explicit-mesh-compatibility"
    assert result.mesh.metadata["heightmap_mesh_compatibility"]["mesh_faces"] == result.mesh.n_faces


def test_displace_heightmap_planar():
    mesh = make_plane(size=(1.0, 1.0), center=(0.0, 0.0, 0.0))
    image = np.ones((2, 2), dtype=float)

    displaced = displace_heightmap(mesh, image, height=0.5, plane="xy", direction="z")
    assert displaced.patch_count == 1
    assert isinstance(displaced.iter_patches()[0], DisplacementSurfacePatch)

    arr = np.ones((2, 2, 4), dtype=np.uint8) * 255
    arr[0, 0, 3] = 0
    masked = displace_heightmap(mesh, arr, height=0.5, plane="xy", direction="z", alpha_mode="mask")
    assert isinstance(masked.iter_patches()[0], DisplacementSurfacePatch)
    assert tessellate_surface_body(masked).mesh.n_faces > 0


def test_displace_heightmap_surface_backend_uses_displacement_payload():
    surface = make_plane(size=(1.0, 1.0), center=(0.0, 0.0, 0.0))
    image = np.ones((2, 2), dtype=float)

    displaced = displace_heightmap(surface, image, height=0.5, plane="xy", direction="z")
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


def test_heightmap_projection_domain_record_derives_xy_bounds_and_grid_spacing():
    patch = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.zeros((3, 5), dtype=float),
        alpha_mask=np.ones((3, 5), dtype=bool),
        xy_scale=(0.5, 2.0),
        center=(10.0, 20.0, 0.0),
    )

    record = heightmap_projection_domain_record(patch)

    assert isinstance(record, HeightmapProjectionDomainRecord)
    assert record.projection == "planar"
    assert record.plane == "xy"
    assert record.sample_shape == (3, 5)
    assert record.sample_spacing == (0.5, 2.0)
    assert record.bounds == (9.0, 11.0, 18.0, 22.0)
    assert record.origin == (9.0, 18.0)


def test_heightmap_grid_alignment_plans_aligned_overlap_and_clipping():
    left = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.zeros((4, 4), dtype=float),
        alpha_mask=np.ones((4, 4), dtype=bool),
        xy_scale=(1.0, 1.0),
        center=(0.0, 0.0, 0.0),
    )
    right = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((4, 4), dtype=float),
        alpha_mask=np.ones((4, 4), dtype=bool),
        xy_scale=(1.0, 1.0),
        center=(1.0, 0.0, 0.0),
    )

    plan = plan_heightmap_grid_alignment(left, right)

    assert isinstance(plan, HeightmapGridAlignmentRecord)
    assert plan.supported is True
    assert plan.alignment == "aligned"
    assert plan.resample_kernel == "none"
    assert isinstance(plan.clipping, HeightmapClippingRecord)
    assert plan.clipping.has_overlap is True
    assert plan.clipping.overlap_bounds == (-0.5, 1.5, -1.5, 1.5)
    assert plan.result_shape == (4, 3)
    assert plan.diagnostics == ()


def test_heightmap_grid_alignment_plans_bilinear_resampling_for_overlapping_misaligned_grids():
    left = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.zeros((3, 3), dtype=float),
        alpha_mask=np.ones((3, 3), dtype=bool),
        xy_scale=(1.0, 1.0),
        center=(0.0, 0.0, 0.0),
    )
    right = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((5, 5), dtype=float),
        alpha_mask=np.ones((5, 5), dtype=bool),
        xy_scale=(0.5, 0.5),
        center=(0.0, 0.0, 0.0),
    )

    plan = plan_heightmap_grid_alignment(left, right)

    assert plan.supported is True
    assert plan.alignment == "resample-required"
    assert plan.resample_kernel == "bilinear"
    assert plan.result_shape == (5, 5)
    assert plan.clipping is not None and plan.clipping.has_overlap is True


def test_heightmap_grid_alignment_refuses_disjoint_domains_and_projection_mismatch_without_mesh_fallback():
    left = HeightmapSurfacePatch(family="heightmap", height_samples=np.zeros((2, 2), dtype=float), center=(0.0, 0.0, 0.0))
    right = HeightmapSurfacePatch(family="heightmap", height_samples=np.ones((2, 2), dtype=float), center=(10.0, 0.0, 0.0))

    disjoint = plan_heightmap_grid_alignment(left, right)
    mismatch = plan_heightmap_grid_alignment(left, left, right_plane="xz")

    assert disjoint.supported is False
    assert disjoint.alignment == "refused"
    assert isinstance(disjoint.diagnostics[0], HeightmapProjectionRefusalDiagnostic)
    assert disjoint.diagnostics[0].code == "disjoint-domain"
    assert disjoint.diagnostics[0].no_mesh_fallback is True
    assert mismatch.supported is False
    assert mismatch.diagnostics[0].code == "projection-mismatch"
    assert "mesh fallback" in mismatch.diagnostics[0].message


def test_heightmap_composition_operators_preserve_native_heightmap_payloads():
    left = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float),
        alpha_mask=np.ones((2, 2), dtype=bool),
    )
    right = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.asarray([[1.0, 0.5], [1.5, 4.0]], dtype=float),
        alpha_mask=np.ones((2, 2), dtype=bool),
    )

    union = compose_heightmap_csg_result("union", left, right)
    intersection = compose_heightmap_csg_result("intersection", left, right)
    difference = compose_heightmap_csg_result("difference", left, right)

    assert isinstance(union, HeightmapCompositionResult)
    assert isinstance(union.operation_record, HeightmapCompositionRecord)
    assert union.supported is True
    assert union.patch is not None and union.body is not None
    assert union.patch.family == "heightmap"
    assert np.allclose(union.patch.height_samples, [[1.0, 1.0], [2.0, 4.0]])
    assert np.all(union.patch.alpha_mask)
    assert np.allclose(intersection.patch.height_samples, [[0.0, 0.5], [1.5, 3.0]])
    assert np.allclose(difference.patch.height_samples, [[0.0, 0.5], [0.5, 0.0]])
    assert union.operation_record.no_mesh_fallback is True
    assert union.operation_record.resample_kernel == "none"
    assert union.patch.metadata["kernel"]["heightmap_csg_composition"]["operation"] == "union"


def test_heightmap_composition_resamples_overlapping_misaligned_grids_without_mesh_fallback():
    left = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.asarray([[0.0, 0.0], [2.0, 2.0]], dtype=float),
        alpha_mask=np.ones((2, 2), dtype=bool),
        xy_scale=(1.0, 1.0),
    )
    right = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((3, 3), dtype=float),
        alpha_mask=np.ones((3, 3), dtype=bool),
        xy_scale=(0.5, 0.5),
    )

    result = compose_heightmap_csg_result("union", left, right)

    assert result.supported is True
    assert result.patch is not None
    assert result.operation_record.resample_kernel == "bilinear"
    assert result.patch.height_samples.shape == (3, 3)
    assert np.max(result.patch.height_samples) >= 1.0
    assert all(diagnostic.no_mesh_fallback for diagnostic in result.diagnostics)


def test_heightmap_composition_refuses_disjoint_and_empty_results_without_mesh_fallback():
    left = HeightmapSurfacePatch(family="heightmap", height_samples=np.ones((2, 2), dtype=float))
    disjoint = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((2, 2), dtype=float),
        center=(10.0, 0.0, 0.0),
    )
    masked = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((2, 2), dtype=float),
        alpha_mask=np.zeros((2, 2), dtype=bool),
    )

    disjoint_result = compose_heightmap_csg_result("union", left, disjoint)
    empty_intersection = compose_heightmap_csg_result("intersection", left, masked)

    assert disjoint_result.supported is False
    assert isinstance(disjoint_result.diagnostics[0], HeightmapCompositionDiagnostic)
    assert disjoint_result.diagnostics[0].code == "alignment-refusal"
    assert disjoint_result.diagnostics[0].no_mesh_fallback is True
    assert empty_intersection.supported is False
    assert empty_intersection.diagnostics[0].code == "representability-refusal"
    assert "mesh fallback" in empty_intersection.diagnostics[0].message


def test_heightmap_representability_reports_overhang_and_multivalue_projection_without_mesh_fallback():
    overhang_transform = np.eye(4, dtype=float)
    overhang_transform[0, 2] = 0.25
    collapsed_transform = np.eye(4, dtype=float)
    collapsed_transform[1, 1] = 0.0
    left = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((2, 2), dtype=float),
        transform_matrix=overhang_transform,
    )
    right = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((2, 2), dtype=float),
        transform_matrix=collapsed_transform,
    )

    report = heightmap_representability_report("union", left, right)
    result = compose_heightmap_csg_result("union", left, right)

    assert isinstance(report, HeightmapRepresentabilityReport)
    assert report.representable is False
    assert {diagnostic.code for diagnostic in report.diagnostics} == {"overhang", "multi-valued-projection"}
    assert all(isinstance(diagnostic, HeightmapOverhangDiagnostic) for diagnostic in report.diagnostics)
    assert all(diagnostic.no_mesh_fallback for diagnostic in report.diagnostics)
    assert result.supported is False
    assert result.diagnostics[0].code == "representability-refusal"
    assert "No mesh fallback" in result.diagnostics[0].message


def test_heightmap_representability_reports_invalid_projection_and_unsafe_grid_before_execution():
    left = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((2, 2), dtype=float),
        alpha_mask=np.zeros((2, 2), dtype=bool),
    )
    right = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((2, 2), dtype=float),
        center=(10.0, 0.0, 0.0),
    )

    report = heightmap_representability_report("intersection", left, right)

    assert report.representable is False
    assert {diagnostic.code for diagnostic in report.diagnostics} == {"invalid-projection", "unsafe-grid"}
    assert report.diagnostics[0].no_mesh_fallback is True
    assert "mesh fallback" in report.diagnostics[0].message


def test_heightmap_promotion_routes_overhangs_to_implicit_with_provenance():
    overhang_transform = np.eye(4, dtype=float)
    overhang_transform[0, 2] = 0.5
    left = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((2, 2), dtype=float),
        transform_matrix=overhang_transform,
    )
    right = HeightmapSurfacePatch(family="heightmap", height_samples=np.ones((2, 2), dtype=float))

    decision = plan_heightmap_promotion_route("union", left, right)

    assert isinstance(decision, HeightmapPromotionDecision)
    assert isinstance(decision.trigger, HeightmapPromotionTriggerRecord)
    assert decision.supported is True
    assert decision.target_family == "implicit"
    assert decision.lossiness == "volumetric-field"
    assert decision.source_families == ("heightmap", "heightmap")
    assert decision.trigger.trigger_codes == ("overhang",)
    assert decision.no_mesh_fallback is True
    assert decision.canonical_payload()["target_family"] == "implicit"


def test_heightmap_promotion_routes_multivalue_projection_to_subdivision_and_refuses_missing_target():
    collapsed_transform = np.eye(4, dtype=float)
    collapsed_transform[0, 0] = 0.0
    left = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((2, 2), dtype=float),
        transform_matrix=collapsed_transform,
    )
    right = HeightmapSurfacePatch(family="heightmap", height_samples=np.ones((2, 2), dtype=float))

    report = heightmap_representability_report("intersection", left, right)
    decision = select_heightmap_promotion_target(report)
    refused = select_heightmap_promotion_target(report, allowed_targets=("implicit",))

    assert decision.supported is True
    assert decision.target_family == "subdivision"
    assert decision.lossiness == "sampled-reconstruction"
    assert refused.supported is False
    assert refused.target_family == "subdivision"
    assert isinstance(refused.diagnostics[0], HeightmapPromotionDiagnostic)
    assert refused.diagnostics[0].code == "missing-route"
    assert refused.diagnostics[0].no_mesh_fallback is True


def test_heightmap_promotion_refuses_safe_representable_and_unsafe_source_routes():
    left = HeightmapSurfacePatch(family="heightmap", height_samples=np.ones((2, 2), dtype=float))
    right = HeightmapSurfacePatch(family="heightmap", height_samples=np.ones((2, 2), dtype=float))
    unsafe = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.ones((2, 2), dtype=float),
        alpha_mask=np.zeros((2, 2), dtype=bool),
    )

    not_needed = plan_heightmap_promotion_route("difference", left, right)
    unsafe_route = plan_heightmap_promotion_route("union", unsafe, right)

    assert not_needed.supported is False
    assert not_needed.target_family is None
    assert not_needed.diagnostics[0].code == "non-applicable"
    assert unsafe_route.supported is False
    assert unsafe_route.diagnostics[0].code == "unsafe-source"
    assert "mesh fallback" in unsafe_route.diagnostics[0].message


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
        )

    masked = displace_heightmap(
        surface,
        image,
        height=0.5,
        plane="xz",
        direction="z",
        bounds=(-0.5, 0.5, -0.5, 0.5),
        alpha_mode="mask",
    )
    ignored = displace_heightmap(
        surface,
        image,
        height=0.5,
        plane="xz",
        direction="z",
        bounds=(-0.5, 0.5, -0.5, 0.5),
        alpha_mode="ignore",
    )
    mesh = tessellate_surface_body(masked).mesh

    assert masked.iter_patches()[0].alpha_mode == "mask"
    assert mesh.n_faces < tessellate_surface_body(ignored).mesh.n_faces


def test_displace_heightmap_requires_surface_body_input() -> None:
    from impression.mesh import Mesh

    mesh = Mesh(vertices=np.zeros((0, 3), dtype=float), faces=np.zeros((0, 3), dtype=int))

    with pytest.raises(ValueError, match="Surface displacement requires a SurfaceBody"):
        displace_heightmap(mesh, np.ones((2, 2), dtype=float))
