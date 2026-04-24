from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from impression.mesh import combine_meshes
from impression.mesh import section_mesh_with_plane
from impression.modeling import (
    Loft,
    as_section,
    handoff_hinge_surface,
    make_bistable_hinge,
    make_box,
    make_living_hinge,
    make_traditional_hinge_pair,
    tessellate_surface_body,
)
from impression.modeling.drafting import make_arrow
from impression.modeling.text import make_text
from impression.modeling.heightmap import heightmap
from tests.loft_showcases import (
    build_anchor_shift_rectangle_profiles,
    build_branching_manifold_profiles,
    build_cylinder_correspondence_profiles,
    build_dual_cylinder_correspondence_profiles,
    build_perforated_cylinder_correspondence_profiles,
    build_phase_shift_cylinder_profiles,
    build_square_correspondence_profiles,
)
from tests.csg_reference_fixtures import (
    build_csg_difference_slot_fixture,
    build_csg_union_box_post_fixture,
    surface_body_section_loops,
)
from tests.text_font_fixtures import require_glyph_capable_font
from tests.reference_images import (
    CameraFramingContract,
    CanonicalObjectViewBundle,
    CvArtifactBundle,
    CvArtifactBundleContract,
    CvFixtureContract,
    DiagnosticPanelContract,
    assess_cv_result,
    canonical_object_view_camera_contracts,
    classify_handedness_from_silhouettes,
    clean_reference_path,
    clean_reference_stl_path,
    compare_canonical_object_view_bundles,
    compare_handedness_space_anchor_contracts,
    compare_planar_loop_silhouettes,
    compare_silhouette_masks,
    compare_text_loop_silhouettes,
    dirty_reference_path,
    dirty_reference_stl_path,
    ensure_complete_cv_artifact_bundle,
    ensure_reference_fixture_pair,
    ensure_reference_image_bundle,
    extract_panel_regions,
    image_signal_stats,
    invalidate_reference_fixture_pair,
    invalidate_reference_image_bundle,
    initial_text_cv_scope_support,
    HandednessSpaceAnchorContract,
    planar_loop_bounds,
    reference_artifact_state,
    reference_fixture_pair_state,
    required_reference_fixture_pair_failures,
    render_canonical_object_view_bundle,
    render_mesh_image_with_camera_contract,
    render_planar_section_bitmap,
    render_planar_section_diff_image,
    render_planar_section_fill_bitmap,
    render_surface_body_image,
    render_surface_body_triptych_image,
    render_surface_consumer_collection_image,
    stl_signal_stats,
    text_cv_actual_loops,
    text_cv_expected_loops,
    triptych_panel_layout,
    validate_camera_contract,
    validate_diagnostic_panel_contract,
    validate_handedness_space_anchor_contract,
    write_surface_body_stl,
    write_surface_consumer_collection_stl,
)

SYMBOL_FONT_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "fonts"
    / "NotoSansSymbols2-Regular.ttf"
)


def _require_text_cv_font() -> Path:
    try:
        return require_glyph_capable_font("SURFACETEST")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _load_docs_example_module(module_name: str, relative_path: str):
    module_path = Path(relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SURFACE_CSG_PROMOTION_MATRIX = (
    "surfacebody/csg_union_box_post",
    "surfacebody/csg_difference_slot",
    "surfacebody/csg_intersection_box_sphere",
)


def _ensure_model_reference_fixture(
    *,
    render_path: Path,
    stl_path: Path,
    reference_image_root: Path,
    reference_stl_root: Path,
    name: str,
    update_dirty_reference_images: bool,
) -> None:
    ensure_reference_fixture_pair(
        rendered_image_path=render_path,
        rendered_stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name=name,
        update_dirty_reference_images=update_dirty_reference_images,
    )
    assert dirty_reference_path(reference_image_root, name).exists()
    assert dirty_reference_stl_path(reference_stl_root, name).exists()


def _write_text_fixture(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_small_image(path: Path, *, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((12, 12, 3), np.asarray(color, dtype=np.uint8), dtype=np.uint8)
    from PIL import Image

    Image.fromarray(image, mode="RGB").save(path)
    return path


def _slice_cv_fixture_contract(*, fixture_id: str, orientation_required: bool) -> CvFixtureContract:
    passing = ("same_shape_same_orientation",) if orientation_required else (
        "same_shape_same_orientation",
        "same_shape_rotated",
    )
    return CvFixtureContract(
        fixture_id=fixture_id,
        lane="slice_silhouette",
        required_artifact_keys=("expected", "actual", "diff"),
        known_result_classes=(
            "same_shape_same_orientation",
            "same_shape_rotated",
            "different_shape",
        ),
        positive_result_classes=("same_shape_same_orientation",),
        transformed_result_classes=("same_shape_rotated",),
        different_result_classes=("different_shape",),
        passing_result_classes=passing,
    )


def _section_bundle_contract(*, fixture_id: str) -> CvArtifactBundleContract:
    return CvArtifactBundleContract(
        fixture_id=fixture_id,
        lane="slice_silhouette",
        required_keys=("expected", "actual", "diff"),
        authoritative_keys=("expected", "actual", "diff"),
        review_keys=("expected", "actual", "diff"),
    )


def _text_cv_fixture_contract(*, fixture_id: str, orientation_required: bool) -> CvFixtureContract:
    passing = ("same_text_same_orientation",) if orientation_required else (
        "same_text_same_orientation",
        "same_text_rotated",
        "same_text_mirrored",
    )
    return CvFixtureContract(
        fixture_id=fixture_id,
        lane="text_glyph_interpretation",
        required_artifact_keys=("expected", "actual", "diff"),
        known_result_classes=(
            "same_text_same_orientation",
            "same_text_rotated",
            "same_text_mirrored",
            "different_text",
            "unreadable",
        ),
        positive_result_classes=("same_text_same_orientation",),
        transformed_result_classes=("same_text_rotated", "same_text_mirrored"),
        different_result_classes=("different_text",),
        unknown_result_classes=("unreadable",),
        passing_result_classes=passing,
    )


def _surface_boolean_bounds_relation(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
    *,
    epsilon: float = 1e-9,
) -> str:
    overlap = (
        max(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        min(left[3], right[3]),
        max(left[4], right[4]),
        min(left[5], right[5]),
    )
    spans = (
        overlap[1] - overlap[0],
        overlap[3] - overlap[2],
        overlap[5] - overlap[4],
    )
    if any(span < -epsilon for span in spans):
        return "disjoint"
    if any(abs(span) <= epsilon for span in spans):
        return "touching"

    def _contains(container: tuple[float, ...], candidate: tuple[float, ...]) -> bool:
        return (
            container[0] <= candidate[0] + epsilon
            and container[1] >= candidate[1] - epsilon
            and container[2] <= candidate[2] + epsilon
            and container[3] >= candidate[3] - epsilon
            and container[4] <= candidate[4] + epsilon
            and container[5] >= candidate[5] - epsilon
        )

    if _contains(left, right) or _contains(right, left):
        return "containment"
    return "overlap"


def _planar_loop_signed_area(loop: np.ndarray) -> float:
    x = loop[:, 0]
    y = loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _planar_loop_total_area(loops: list[np.ndarray] | tuple[np.ndarray, ...]) -> float:
    return float(sum(abs(_planar_loop_signed_area(np.asarray(loop, dtype=float))) for loop in loops))


def _text_bundle_contract(*, fixture_id: str) -> CvArtifactBundleContract:
    return CvArtifactBundleContract(
        fixture_id=fixture_id,
        lane="text_glyph_interpretation",
        required_keys=("expected", "actual", "diff"),
        authoritative_keys=("expected", "actual"),
        review_keys=("expected", "actual", "diff"),
    )


def test_reference_fixture_pair_bootstraps_dirty_artifacts_for_new_fixture(tmp_path: Path) -> None:
    reference_image_root = tmp_path / "reference-images"
    reference_stl_root = tmp_path / "reference-stl"
    render_path = _write_small_image(tmp_path / "rendered.png", color=(10, 20, 30))
    stl_path = _write_text_fixture(tmp_path / "rendered.stl", "solid test\nendsolid test\n")

    state_before = reference_fixture_pair_state(
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="demo/fixture",
    )
    assert state_before.is_new_fixture

    image_reference, stl_reference = ensure_reference_fixture_pair(
        rendered_image_path=render_path,
        rendered_stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="demo/fixture",
        update_dirty_reference_images=False,
    )

    assert image_reference == dirty_reference_path(reference_image_root, "demo/fixture")
    assert stl_reference == dirty_reference_stl_path(reference_stl_root, "demo/fixture")

    state_after = reference_fixture_pair_state(
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="demo/fixture",
    )
    assert not state_after.is_new_fixture
    assert not state_after.has_partial_group
    assert state_after.image.selected_tier == "dirty"
    assert state_after.stl.selected_tier == "dirty"


def test_reference_fixture_pair_prefers_clean_baselines(tmp_path: Path) -> None:
    reference_image_root = tmp_path / "reference-images"
    reference_stl_root = tmp_path / "reference-stl"
    name = "demo/fixture"
    render_path = _write_small_image(tmp_path / "rendered.png", color=(40, 50, 60))
    stl_path = _write_text_fixture(tmp_path / "rendered.stl", "solid test\nendsolid test\n")

    dirty_image = dirty_reference_path(reference_image_root, name)
    dirty_stl = dirty_reference_stl_path(reference_stl_root, name)
    clean_image = clean_reference_path(reference_image_root, name)
    clean_stl = clean_reference_stl_path(reference_stl_root, name)
    _write_small_image(dirty_image, color=(255, 0, 0))
    _write_text_fixture(dirty_stl, "solid dirty\nendsolid dirty\n")
    _write_small_image(clean_image, color=(40, 50, 60))
    _write_text_fixture(clean_stl, "solid test\nendsolid test\n")

    image_state = reference_artifact_state(reference_image_root, name, kind="image")
    stl_state = reference_artifact_state(reference_stl_root, name, kind="stl")
    assert image_state.selected_tier == "clean"
    assert stl_state.selected_tier == "clean"

    image_reference, stl_reference = ensure_reference_fixture_pair(
        rendered_image_path=render_path,
        rendered_stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name=name,
        update_dirty_reference_images=False,
    )

    assert image_reference == clean_image
    assert stl_reference == clean_stl


def test_reference_fixture_pair_fails_for_partial_existing_group(tmp_path: Path) -> None:
    reference_image_root = tmp_path / "reference-images"
    reference_stl_root = tmp_path / "reference-stl"
    name = "demo/fixture"
    render_path = _write_small_image(tmp_path / "rendered.png", color=(70, 80, 90))
    stl_path = _write_text_fixture(tmp_path / "rendered.stl", "solid test\nendsolid test\n")
    _write_small_image(dirty_reference_path(reference_image_root, name), color=(70, 80, 90))

    state = reference_fixture_pair_state(
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name=name,
    )
    assert state.has_partial_group
    assert state.image.exists
    assert not state.stl.exists

    with pytest.raises(AssertionError, match="missing STL counterpart"):
        ensure_reference_fixture_pair(
            rendered_image_path=render_path,
            rendered_stl_path=stl_path,
            reference_image_root=reference_image_root,
            reference_stl_root=reference_stl_root,
            name=name,
            update_dirty_reference_images=False,
        )


def test_invalidate_reference_fixture_pair_removes_dirty_and_clean_baselines(tmp_path: Path) -> None:
    reference_image_root = tmp_path / "reference-images"
    reference_stl_root = tmp_path / "reference-stl"
    name = "demo/fixture"
    _write_small_image(dirty_reference_path(reference_image_root, name), color=(1, 2, 3))
    _write_small_image(clean_reference_path(reference_image_root, name), color=(4, 5, 6))
    _write_text_fixture(dirty_reference_stl_path(reference_stl_root, name), "solid dirty\nendsolid dirty\n")
    _write_text_fixture(clean_reference_stl_path(reference_stl_root, name), "solid clean\nendsolid clean\n")

    removed = invalidate_reference_fixture_pair(
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name=name,
    )

    assert len(removed) == 4
    assert reference_fixture_pair_state(
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name=name,
    ).is_new_fixture


def test_surface_csg_promotion_matrix_names_are_explicit() -> None:
    assert SURFACE_CSG_PROMOTION_MATRIX == (
        "surfacebody/csg_union_box_post",
        "surfacebody/csg_difference_slot",
        "surfacebody/csg_intersection_box_sphere",
    )


def test_surface_csg_promotion_gate_reports_missing_named_references(
    reference_image_root: Path,
    reference_stl_root: Path,
) -> None:
    failures = required_reference_fixture_pair_failures(
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        names=SURFACE_CSG_PROMOTION_MATRIX,
    )

    assert not any("csg_union_box_post" in failure for failure in failures)
    assert not any("csg_difference_slot" in failure for failure in failures)
    assert "surfacebody/csg_intersection_box_sphere is missing a reference image baseline." in failures
    assert "surfacebody/csg_intersection_box_sphere is missing a reference STL baseline." in failures


def test_cv_fixture_contract_requires_shared_fields() -> None:
    with pytest.raises(ValueError, match="fixture_id"):
        CvFixtureContract(
            fixture_id="",
            lane="slice_silhouette",
            required_artifact_keys=("expected",),
            known_result_classes=("same",),
            positive_result_classes=("same",),
            passing_result_classes=("same",),
        )
    with pytest.raises(ValueError, match="required_artifact_keys"):
        CvFixtureContract(
            fixture_id="demo",
            lane="slice_silhouette",
            required_artifact_keys=(),
            known_result_classes=("same",),
            positive_result_classes=("same",),
            passing_result_classes=("same",),
        )


def test_cv_fixture_contract_maps_transformed_outcomes_explicitly() -> None:
    contract = _slice_cv_fixture_contract(fixture_id="demo", orientation_required=False)

    positive = assess_cv_result(contract, "same_shape_same_orientation")
    transformed = assess_cv_result(contract, "same_shape_rotated")
    different = assess_cv_result(contract, "different_shape")

    assert positive.shared_pattern == "positive"
    assert positive.passes_contract
    assert transformed.shared_pattern == "transformed"
    assert transformed.passes_contract
    assert different.shared_pattern == "different"
    assert not different.passes_contract


def test_cv_artifact_bundle_bootstraps_all_required_images(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference-images"
    bundle = CvArtifactBundle(
        contract=_section_bundle_contract(fixture_id="demo"),
        stage="review",
        artifacts={
            "expected": _write_small_image(tmp_path / "expected.png", color=(10, 10, 10)),
            "actual": _write_small_image(tmp_path / "actual.png", color=(20, 20, 20)),
            "diff": _write_small_image(tmp_path / "diff.png", color=(30, 30, 30)),
        },
    )

    references = ensure_reference_image_bundle(
        bundle=bundle,
        reference_root=reference_root,
        bundle_name="bundles/demo",
        update_dirty_reference_images=False,
    )

    assert set(references) == {"expected", "actual", "diff"}
    for key in bundle.contract.required_keys:
        assert dirty_reference_path(reference_root, f"bundles/demo_{key}").exists()


def test_cv_artifact_bundle_fails_for_partial_existing_bundle(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference-images"
    bundle = CvArtifactBundle(
        contract=_section_bundle_contract(fixture_id="demo"),
        stage="review",
        artifacts={
            "expected": _write_small_image(tmp_path / "expected.png", color=(10, 10, 10)),
            "actual": _write_small_image(tmp_path / "actual.png", color=(20, 20, 20)),
            "diff": _write_small_image(tmp_path / "diff.png", color=(30, 30, 30)),
        },
    )
    _write_small_image(dirty_reference_path(reference_root, "bundles/demo_expected"), color=(10, 10, 10))

    with pytest.raises(AssertionError, match="missing existing references"):
        ensure_reference_image_bundle(
            bundle=bundle,
            reference_root=reference_root,
            bundle_name="bundles/demo",
            update_dirty_reference_images=False,
        )


def test_invalidate_reference_image_bundle_removes_all_bundle_members(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference-images"
    for key, color in (("expected", (1, 1, 1)), ("actual", (2, 2, 2)), ("diff", (3, 3, 3))):
        _write_small_image(dirty_reference_path(reference_root, f"bundles/demo_{key}"), color=color)
        _write_small_image(clean_reference_path(reference_root, f"bundles/demo_{key}"), color=color)

    removed = invalidate_reference_image_bundle(
        reference_root,
        "bundles/demo",
        keys=("expected", "actual", "diff"),
    )

    assert len(removed) == 6


def test_text_cv_scope_is_bounded_to_single_line_uppercase_ascii() -> None:
    font_path = str(_require_text_cv_font())

    in_scope, missing = initial_text_cv_scope_support("HELLO", font_path=font_path)
    assert in_scope
    assert missing == ()

    multiline_scope, multiline_missing = initial_text_cv_scope_support("HI\nTHERE", font_path=font_path)
    assert not multiline_scope
    assert multiline_missing == ()

    lowercase_scope, lowercase_missing = initial_text_cv_scope_support("Hello", font_path=font_path)
    assert not lowercase_scope
    assert lowercase_missing == ()


def test_text_cv_classifies_same_text_same_orientation(tmp_path: Path) -> None:
    font_path = str(_require_text_cv_font())
    body = make_text("TEST", depth=0.2, font_size=1.0, font_path=font_path, backend="surface")
    expected_loops = text_cv_expected_loops(content="TEST", font_path=font_path)
    actual_loops = text_cv_actual_loops(body, slice_z=0.1)
    shared_bounds = planar_loop_bounds(expected_loops, actual_loops)

    expected_path = tmp_path / "text-expected.png"
    actual_path = tmp_path / "text-actual.png"
    diff_path = tmp_path / "text-diff.png"
    render_planar_section_fill_bitmap(expected_loops, expected_path, bounds=shared_bounds, image_size=(1200, 400))
    render_planar_section_fill_bitmap(actual_loops, actual_path, bounds=shared_bounds, image_size=(1200, 400))
    render_planar_section_diff_image(expected_loops, actual_loops, diff_path, bounds=shared_bounds)

    bundle = ensure_complete_cv_artifact_bundle(
        CvArtifactBundle(
            contract=_text_bundle_contract(fixture_id="text/demo_same"),
            stage="review",
            artifacts={"expected": expected_path, "actual": actual_path, "diff": diff_path},
        )
    )
    assert set(bundle.artifacts) == {"expected", "actual", "diff"}

    comparison = compare_text_loop_silhouettes(
        content="TEST",
        font_path=font_path,
        expected_loops=expected_loops,
        actual_loops=actual_loops,
        bounds=shared_bounds,
    )
    assessment = assess_cv_result(
        _text_cv_fixture_contract(fixture_id="text/demo_same", orientation_required=True),
        comparison.result_class,
    )
    assert comparison.result_class == "same_text_same_orientation"
    assert comparison.same_orientation_iou >= 0.9
    assert not comparison.fallback_detected
    assert assessment.passes_contract


def test_text_cv_classifies_rotated_and_mirrored_text() -> None:
    font_path = str(_require_text_cv_font())
    expected_loops = text_cv_expected_loops(content="TEST", font_path=font_path)
    rotated_loops = _rotate_planar_loops_90(expected_loops)
    mirrored_loops = [np.column_stack((-loop[:, 0], loop[:, 1])) for loop in expected_loops]

    rotated = compare_text_loop_silhouettes(
        content="TEST",
        font_path=font_path,
        expected_loops=expected_loops,
        actual_loops=rotated_loops,
    )
    mirrored = compare_text_loop_silhouettes(
        content="TEST",
        font_path=font_path,
        expected_loops=expected_loops,
        actual_loops=mirrored_loops,
    )

    assert rotated.result_class == "same_text_rotated"
    assert rotated.best_rotation_iou >= 0.9
    assert mirrored.result_class == "same_text_mirrored"
    assert mirrored.best_mirror_iou >= 0.9


def test_text_cv_classifies_different_text() -> None:
    font_path = str(_require_text_cv_font())
    expected_loops = text_cv_expected_loops(content="TEST", font_path=font_path)
    actual_loops = text_cv_expected_loops(content="BEST", font_path=font_path)

    comparison = compare_text_loop_silhouettes(
        content="TEST",
        font_path=font_path,
        expected_loops=expected_loops,
        actual_loops=actual_loops,
    )

    assert comparison.result_class == "different_text"
    assert comparison.best_rotation_iou < 0.9
    assert comparison.best_mirror_iou < 0.9


def test_text_cv_treats_fallback_glyph_output_as_semantic_failure() -> None:
    comparison = compare_text_loop_silhouettes(
        content="AB",
        font_path=str(SYMBOL_FONT_PATH),
        actual_loops=text_cv_expected_loops(content="AB", font_path=str(SYMBOL_FONT_PATH)),
    )
    assessment = assess_cv_result(
        _text_cv_fixture_contract(fixture_id="text/fallback", orientation_required=True),
        comparison.result_class,
    )

    assert comparison.result_class == "different_text"
    assert comparison.fallback_detected
    assert comparison.missing_glyphs == ("A", "B")
    assert not assessment.passes_contract


def test_text_cv_marks_out_of_scope_text_as_unreadable() -> None:
    font_path = str(_require_text_cv_font())
    comparison = compare_text_loop_silhouettes(
        content="Hello",
        font_path=font_path,
        actual_loops=text_cv_expected_loops(content="HELLO", font_path=font_path),
    )

    assert comparison.result_class == "unreadable"
    assert not comparison.in_scope


def test_camera_contract_detects_declared_drift_categories() -> None:
    contract = CameraFramingContract(
        position=(1.0, 2.0, 3.0),
        target=(0.0, 0.0, 0.0),
        up_vector=(0.0, 0.0, 1.0),
        projection_mode="parallel",
        window_size=(512, 512),
        parallel_scale=2.0,
    )
    observed = CameraFramingContract(
        position=(1.5, 2.0, 3.0),
        target=(0.0, 0.5, 0.0),
        up_vector=(0.0, 1.0, 0.0),
        projection_mode="perspective",
        window_size=(640, 512),
        parallel_scale=2.5,
    )

    violations = validate_camera_contract(contract, observed)

    assert {violation.kind for violation in violations} == {
        "pose_drift",
        "target_drift",
        "up_vector_drift",
        "projection_drift",
        "crop_drift",
        "framing_drift",
    }


def test_render_mesh_image_with_camera_contract_emits_contract_bound_render(tmp_path: Path) -> None:
    body = make_arrow((0.0, 0.0, 0.0), (1.25, 0.25, 0.35), color="#ffb703", backend="surface")
    mesh = tessellate_surface_body(body).mesh
    contract = canonical_object_view_camera_contracts(mesh.bounds)["front"]
    output_path = tmp_path / "front.png"

    render_mesh_image_with_camera_contract(mesh, output_path, camera_contract=contract, mesh_color="black")

    stats = image_signal_stats(output_path)
    assert stats["occupancy"] > 0.005


def test_canonical_object_view_bundle_emits_stable_view_set_and_diagnostic_beauty(tmp_path: Path) -> None:
    body = make_arrow((0.0, 0.0, 0.0), (1.25, 0.25, 0.35), color="#ffb703", backend="surface")
    mesh = tessellate_surface_body(body).mesh

    bundle = render_canonical_object_view_bundle(
        mesh,
        tmp_path,
        stem="arrow",
        include_diagnostic_beauty=True,
    )

    assert bundle.view_order == ("front", "side", "top", "isometric")
    assert tuple(bundle.silhouettes) == bundle.view_order
    assert all(path.exists() for path in bundle.silhouettes.values())
    assert bundle.diagnostic_beauty is not None
    assert tuple(bundle.diagnostic_beauty) == bundle.view_order
    assert all(path.exists() for path in bundle.diagnostic_beauty.values())


def test_canonical_object_view_comparison_uses_declared_silhouettes_and_catches_mismatch(tmp_path: Path) -> None:
    fixture = build_csg_union_box_post_fixture()
    mesh = tessellate_surface_body(fixture["result_body"]).mesh
    expected = render_canonical_object_view_bundle(mesh, tmp_path / "expected", stem="arrow")
    from PIL import Image

    rotated_front = tmp_path / "rotated-front.png"
    with Image.open(expected.silhouettes["front"]) as image:
        image.rotate(90, expand=True).save(rotated_front)
    actual = CanonicalObjectViewBundle(
        view_order=expected.view_order,
        silhouettes={
            "front": rotated_front,
            "side": expected.silhouettes["side"],
            "top": expected.silhouettes["top"],
            "isometric": expected.silhouettes["isometric"],
        },
        diagnostic_beauty={
            "front": expected.silhouettes["front"],
            "side": expected.silhouettes["side"],
            "top": expected.silhouettes["top"],
            "isometric": expected.silhouettes["isometric"],
        },
    )

    comparison = compare_canonical_object_view_bundles(expected, actual)

    assert "front" in comparison.mismatched_views
    assert "top" not in comparison.mismatched_views


@pytest.mark.parametrize(
    ("builder", "fixture_name"),
    [
        (build_csg_union_box_post_fixture, "surfacebody/csg_union_box_post"),
        (build_csg_difference_slot_fixture, "surfacebody/csg_difference_slot"),
    ],
)
def test_surface_csg_overlap_fixtures_require_triptych_overlap_and_edge_protrusion(
    builder,
    fixture_name: str,
) -> None:
    fixture = builder()
    assert fixture["fixture_name"] == fixture_name
    assert fixture["presentation"] == "triptych"
    assert fixture["evidence_scope"] == "overlap"
    assert fixture["orientation_cue"] == "edge_protrusion"
    assert fixture["orientation_required"] is True

    left_operand = fixture["left_operand"]
    right_operand = fixture["right_operand"]
    result_body = fixture["result_body"]
    slice_z = float(fixture["slice_z"])
    expected_slice_loops = fixture["expected_slice_loops"]

    assert _surface_boolean_bounds_relation(
        left_operand.bounds_estimate(),
        right_operand.bounds_estimate(),
    ) == "overlap"

    actual_slice_loops = surface_body_section_loops(result_body, slice_z)
    left_slice_loops = surface_body_section_loops(left_operand, slice_z)
    right_slice_loops = surface_body_section_loops(right_operand, slice_z)
    result_area = _planar_loop_total_area(actual_slice_loops)
    left_area = _planar_loop_total_area(left_slice_loops)
    right_area = _planar_loop_total_area(right_slice_loops)

    assert compare_planar_loop_silhouettes(expected_slice_loops, actual_slice_loops).relationship == (
        "same_shape_same_orientation"
    )
    left_distinct = compare_planar_loop_silhouettes(expected_slice_loops, left_slice_loops).relationship == "different_shape"
    right_distinct = compare_planar_loop_silhouettes(expected_slice_loops, right_slice_loops).relationship == "different_shape"
    assert left_distinct or abs(result_area - left_area) >= (0.10 * result_area)
    assert right_distinct or abs(result_area - right_area) >= (0.10 * result_area)

    rotated = compare_planar_loop_silhouettes(expected_slice_loops, _rotate_planar_loops_90(expected_slice_loops))
    assert rotated.relationship == "same_shape_rotated"


def test_handedness_anchor_contract_requires_explicit_camera_dependency() -> None:
    contract = HandednessSpaceAnchorContract(
        modeling_basis=("x", "y", "z"),
        export_basis=("x", "y", "z"),
        viewer_basis=("x", "y", "z"),
        canonical_view="front",
        camera_contract=None,
    )

    issues = validate_handedness_space_anchor_contract(contract)

    assert "camera_contract dependency is required for handedness verification." in issues


def test_handedness_anchor_contract_detects_space_drift() -> None:
    camera_contract = CameraFramingContract(
        position=(0.0, -5.0, 0.0),
        target=(0.0, 0.0, 0.0),
        up_vector=(0.0, 0.0, 1.0),
        projection_mode="parallel",
        window_size=(512, 512),
        parallel_scale=2.0,
    )
    expected = HandednessSpaceAnchorContract(
        modeling_basis=("x", "y", "z"),
        export_basis=("x", "y", "z"),
        viewer_basis=("x", "y", "z"),
        canonical_view="front",
        camera_contract=camera_contract,
    )
    observed = HandednessSpaceAnchorContract(
        modeling_basis=("x", "y", "z"),
        export_basis=("x", "-y", "z"),
        viewer_basis=("x", "y", "z"),
        canonical_view="front",
        camera_contract=camera_contract,
    )

    issues = compare_handedness_space_anchor_contracts(expected, observed)

    assert "export_basis drifted from the declared anchor contract." in issues


def test_handedness_classifier_detects_same_and_mirrored_witness(tmp_path: Path) -> None:
    mesh = combine_meshes(
        [
            make_box(size=(1.2, 0.4, 1.0), center=(0.0, 0.0, 0.0)),
            make_box(size=(0.3, 0.4, 0.3), center=(0.75, 0.0, 0.55)),
        ]
    )
    bundle = render_canonical_object_view_bundle(mesh, tmp_path / "expected", stem="arrow")
    from PIL import Image

    mirrored_front = tmp_path / "mirrored-front.png"
    with Image.open(bundle.silhouettes["front"]) as image:
        image.transpose(Image.FLIP_LEFT_RIGHT).save(mirrored_front)

    same = classify_handedness_from_silhouettes(bundle.silhouettes["front"], bundle.silhouettes["front"])
    mirrored = classify_handedness_from_silhouettes(bundle.silhouettes["front"], mirrored_front)

    assert same.witness_adequate
    assert same.result_class == "same_handedness"
    assert mirrored.result_class == "mirrored"


def test_handedness_classifier_returns_unknown_for_symmetric_witness(tmp_path: Path) -> None:
    body = make_box(size=(2.0, 2.0, 1.0), backend="surface")
    mesh = tessellate_surface_body(body).mesh
    bundle = render_canonical_object_view_bundle(mesh, tmp_path / "expected", stem="box")

    classification = classify_handedness_from_silhouettes(
        bundle.silhouettes["front"],
        bundle.silhouettes["front"],
    )

    assert not classification.witness_adequate
    assert classification.result_class == "orientation_unknown"


@pytest.mark.parametrize("builder", [build_csg_union_box_post_fixture, build_csg_difference_slot_fixture])
def test_triptych_panel_layout_and_extraction_are_deterministic(tmp_path: Path, builder) -> None:
    fixture = builder()
    output_path = tmp_path / "triptych.png"
    render_surface_body_triptych_image(
        fixture["left_operand"],
        fixture["result_body"],
        fixture["right_operand"],
        output_path,
    )

    layout = triptych_panel_layout(output_path)
    regions = extract_panel_regions(output_path, layout)

    assert layout.contract.panel_order == ("left", "result", "right")
    assert tuple(regions) == layout.contract.panel_order
    assert all(region.size[0] > 0 and region.size[1] > 0 for region in regions.values())


def test_diagnostic_panel_contract_defaults_to_honest_diagnostic_only_posture() -> None:
    contract = DiagnosticPanelContract(panel_order=("left", "result", "right"))
    issues = validate_diagnostic_panel_contract(contract)

    assert issues == ()


def test_diagnostic_panel_contract_rejects_implicit_proof_claims() -> None:
    contract = DiagnosticPanelContract(
        panel_order=("left", "result", "right"),
        diagnostic_only=False,
        delegated_proof_lane=None,
        shared_scene=False,
    )
    issues = validate_diagnostic_panel_contract(contract)

    assert "Diagnostic panels must default to diagnostic_only when no proof lane is delegated." in issues


def test_diagnostic_panel_contract_requires_shared_scene_for_proof_delegation() -> None:
    contract = DiagnosticPanelContract(
        panel_order=("expected", "actual", "diff"),
        diagnostic_only=False,
        delegated_proof_lane="slice_silhouette",
        shared_scene=False,
    )
    issues = validate_diagnostic_panel_contract(contract)

    assert "Proof delegation requires shared_scene=True for panel honesty." in issues


def _loft_fixture_mesh(profiles: list[object], path: np.ndarray):
    sections = [as_section(profile) for profile in profiles]
    return Loft(
        progression=np.linspace(0.0, 1.0, len(sections)),
        stations=path,
        topology=sections,
        cap_ends=True,
        split_merge_mode="resolve",
    )


def _expected_section_loops(profile: object, station_origin: np.ndarray) -> list[np.ndarray]:
    section = as_section(profile)
    offset = np.asarray(station_origin, dtype=float)[:2]
    loops: list[np.ndarray] = []
    for region in section.regions:
        loops.append(np.asarray(region.outer.points, dtype=float) + offset)
        for hole in region.holes:
            loops.append(np.asarray(hole.points, dtype=float) + offset)
    return loops


def _actual_section_loops(body, station_origin: np.ndarray) -> list[np.ndarray]:
    mesh = tessellate_surface_body(body).mesh
    result = section_mesh_with_plane(
        mesh,
        origin=(0.0, 0.0, float(station_origin[2])),
        normal=(0.0, 0.0, 1.0),
        stitch_epsilon=1e-5,
    )
    return [polyline.points[:, :2] for polyline in result.polylines if polyline.closed]


def _notched_rectangle_mask(
    *,
    height: int = 120,
    width: int = 120,
    top: int = 24,
    left: int = 24,
    rect_height: int = 56,
    rect_width: int = 48,
    notch_height: int = 14,
    notch_width: int = 10,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    bottom = top + rect_height
    right = left + rect_width
    mask[top:bottom, left:right] = True
    notch_top = top + ((rect_height - notch_height) // 2)
    notch_bottom = notch_top + notch_height
    mask[notch_top:notch_bottom, right : right + notch_width] = True
    return mask


def _rotate_planar_loops_90(loops: list[np.ndarray] | tuple[np.ndarray, ...]) -> list[np.ndarray]:
    return [np.column_stack((-loop[:, 1], loop[:, 0])) for loop in loops]


@pytest.mark.surfacebody
@pytest.mark.reference_image
def test_surfacebody_box_reference_image(
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    body = make_box(size=(2.0, 3.0, 1.5), center=(0.0, 0.0, 0.0), backend="surface")
    render_path = tmp_path / "surfacebody-box.png"
    stl_path = tmp_path / "surfacebody-box.stl"
    render_surface_body_image(body, render_path)
    write_surface_body_stl(body, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.03
    assert stats["std_luma"] > 10.0
    assert stl_stats["facet_count"] >= 12
    assert stl_stats["vertex_count"] >= 36
    assert stl_stats["file_size"] > 256
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="surfacebody/box",
        update_dirty_reference_images=update_dirty_reference_images,
    )


@pytest.mark.loft
@pytest.mark.reference_image
def test_loft_branching_manifold_reference_image(
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    profiles, path = build_branching_manifold_profiles()
    body = Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=True,
        split_merge_mode="resolve",
    )
    render_path = tmp_path / "loft-branching-manifold.png"
    stl_path = tmp_path / "loft-branching-manifold.stl"
    render_surface_body_image(body, render_path)
    write_surface_body_stl(body, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.02
    assert stats["std_luma"] > 10.0
    assert stl_stats["facet_count"] > 100
    assert stl_stats["vertex_count"] > 300
    assert stl_stats["file_size"] > 4096
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="loft/branching_manifold",
        update_dirty_reference_images=update_dirty_reference_images,
    )


@pytest.mark.loft
@pytest.mark.reference_image
def test_loft_hourglass_vessel_reference_image(
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    module = _load_docs_example_module(
        "loft_hourglass_vessel_example",
        "docs/examples/loft/real_world/loft_hourglass_vessel_example.py",
    )
    body = module.build_surface_body(module.TEST_PARAMETERS, module.TEST_QUALITY)
    render_path = tmp_path / "loft-hourglass-vessel.png"
    stl_path = tmp_path / "loft-hourglass-vessel.stl"
    render_surface_body_image(body, render_path)
    write_surface_body_stl(body, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.04
    assert stats["std_luma"] > 10.0
    assert stl_stats["facet_count"] > 2000
    assert stl_stats["vertex_count"] > 6000
    assert stl_stats["file_size"] > 250000
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="loft/hourglass_vessel",
        update_dirty_reference_images=update_dirty_reference_images,
    )


@pytest.mark.loft
@pytest.mark.reference_image
def test_loft_square_correspondence_reference_image(
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    profiles, path = build_square_correspondence_profiles()
    body = Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=True,
    )
    render_path = tmp_path / "loft-square-correspondence.png"
    stl_path = tmp_path / "loft-square-correspondence.stl"
    render_surface_body_image(body, render_path)
    write_surface_body_stl(body, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.01
    assert stl_stats["facet_count"] > 100
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="loft/square_correspondence",
        update_dirty_reference_images=update_dirty_reference_images,
    )


@pytest.mark.loft
@pytest.mark.reference_image
def test_loft_cylinder_correspondence_reference_image(
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    profiles, path = build_cylinder_correspondence_profiles()
    body = Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=True,
    )
    render_path = tmp_path / "loft-cylinder-correspondence.png"
    stl_path = tmp_path / "loft-cylinder-correspondence.stl"
    render_surface_body_image(body, render_path)
    write_surface_body_stl(body, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.01
    assert stl_stats["facet_count"] > 100
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="loft/cylinder_correspondence",
        update_dirty_reference_images=update_dirty_reference_images,
    )


@pytest.mark.loft
@pytest.mark.reference_image
def test_loft_anchor_shift_rectangle_reference_image(
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    profiles, path = build_anchor_shift_rectangle_profiles()
    body = Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=True,
    )
    render_path = tmp_path / "loft-anchor-shift-rectangle.png"
    stl_path = tmp_path / "loft-anchor-shift-rectangle.stl"
    render_surface_body_image(body, render_path)
    write_surface_body_stl(body, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.01
    assert stl_stats["facet_count"] > 100
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="loft/anchor_shift_rectangle",
        update_dirty_reference_images=update_dirty_reference_images,
    )


@pytest.mark.loft
@pytest.mark.reference_image
def test_loft_phase_shift_cylinder_reference_image(
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    profiles, path = build_phase_shift_cylinder_profiles()
    body = Loft(
        progression=np.linspace(0.0, 1.0, len(profiles)),
        stations=path,
        topology=profiles,
        cap_ends=True,
    )
    render_path = tmp_path / "loft-phase-shift-cylinder.png"
    stl_path = tmp_path / "loft-phase-shift-cylinder.stl"
    render_surface_body_image(body, render_path)
    write_surface_body_stl(body, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.01
    assert stl_stats["facet_count"] > 100
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="loft/phase_shift_cylinder",
        update_dirty_reference_images=update_dirty_reference_images,
    )


@pytest.mark.loft
@pytest.mark.reference_image
@pytest.mark.parametrize(
    ("fixture_name", "builder", "station_index", "orientation_required"),
    [
        ("square_station_1", build_square_correspondence_profiles, 1, False),
        ("cylinder_station_1", build_cylinder_correspondence_profiles, 1, False),
        ("anchor_shift_rectangle_station_2", build_anchor_shift_rectangle_profiles, 2, False),
        ("phase_shift_cylinder_station_2", build_phase_shift_cylinder_profiles, 2, False),
        ("dual_cylinder_station_1", build_dual_cylinder_correspondence_profiles, 1, False),
        ("perforated_cylinder_station_1", build_perforated_cylinder_correspondence_profiles, 1, False),
    ],
)
def test_loft_section_comparison_reference_images(
    fixture_name: str,
    builder,
    station_index: int,
    orientation_required: bool,
    reference_image_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    profiles, path = builder()
    body = _loft_fixture_mesh(profiles, path)
    expected_loops = _expected_section_loops(profiles[station_index], path[station_index])
    actual_loops = _actual_section_loops(body, path[station_index])
    shared_bounds = planar_loop_bounds(expected_loops, actual_loops)

    expected_path = tmp_path / f"{fixture_name}-expected.png"
    actual_path = tmp_path / f"{fixture_name}-actual.png"
    diff_path = tmp_path / f"{fixture_name}-diff.png"

    render_planar_section_bitmap(expected_loops, expected_path, bounds=shared_bounds)
    render_planar_section_bitmap(actual_loops, actual_path, bounds=shared_bounds)
    render_planar_section_diff_image(expected_loops, actual_loops, diff_path, bounds=shared_bounds)
    bundle = ensure_complete_cv_artifact_bundle(
        CvArtifactBundle(
            contract=_section_bundle_contract(fixture_id=f"loft_sections/{fixture_name}"),
            stage="review",
            artifacts={
                "expected": expected_path,
                "actual": actual_path,
                "diff": diff_path,
            },
        )
    )

    expected_stats = image_signal_stats(expected_path, background_threshold=5)
    actual_stats = image_signal_stats(actual_path, background_threshold=5)
    diff_stats = image_signal_stats(diff_path, background_threshold=5)

    assert expected_stats["occupancy"] > 0.002
    assert actual_stats["occupancy"] > 0.002
    assert diff_stats["occupancy"] > 0.002
    contract = _slice_cv_fixture_contract(
        fixture_id=f"loft_sections/{fixture_name}",
        orientation_required=orientation_required,
    )
    silhouette_comparison = compare_planar_loop_silhouettes(
        expected_loops,
        actual_loops,
        bounds=shared_bounds,
    )
    assessment = assess_cv_result(contract, silhouette_comparison.relationship)
    assert assessment.shared_pattern != "different"
    assert assessment.passes_contract
    if orientation_required:
        assert silhouette_comparison.same_orientation_iou >= 0.9
    else:
        assert silhouette_comparison.best_rotation_iou >= 0.9

    ensure_reference_image_bundle(
        bundle=bundle,
        reference_root=reference_image_root,
        bundle_name=f"loft_sections/{fixture_name}",
        update_dirty_reference_images=update_dirty_reference_images,
    )


def test_silhouette_classifier_treats_scaled_and_translated_masks_as_same_shape() -> None:
    expected_mask = _notched_rectangle_mask()
    actual_mask = _notched_rectangle_mask(
        top=34,
        left=14,
        rect_height=44,
        rect_width=38,
        notch_height=11,
        notch_width=8,
    )

    comparison = compare_silhouette_masks(expected_mask, actual_mask)

    assert comparison.relationship == "same_shape_same_orientation"
    assert comparison.same_orientation_iou >= 0.9


def test_silhouette_classifier_detects_rotated_same_shape() -> None:
    expected_mask = _notched_rectangle_mask()
    actual_mask = np.rot90(expected_mask)

    comparison = compare_silhouette_masks(expected_mask, actual_mask)

    assert comparison.relationship == "same_shape_rotated"
    assert comparison.best_rotation_deg in {90, 270}
    assert comparison.best_rotation_iou >= 0.9


def test_silhouette_classifier_detects_different_shape() -> None:
    expected_mask = _notched_rectangle_mask()
    actual_mask = _notched_rectangle_mask(notch_height=26, notch_width=20)

    comparison = compare_silhouette_masks(expected_mask, actual_mask)

    assert comparison.relationship == "different_shape"
    assert comparison.best_rotation_iou < 0.9


@pytest.mark.surfacebody
@pytest.mark.reference_image
def test_surface_arrow_reference_image(
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    body = make_arrow((0.0, 0.0, 0.0), (1.25, 0.25, 0.35), color="#ffb703", backend="surface")
    render_path = tmp_path / "surface-arrow.png"
    stl_path = tmp_path / "surface-arrow.stl"
    render_surface_body_image(body, render_path)
    write_surface_body_stl(body, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.01
    assert stl_stats["facet_count"] > 50
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="surfacebody/drafting_arrow",
        update_dirty_reference_images=update_dirty_reference_images,
    )


@pytest.mark.surfacebody
@pytest.mark.reference_image
def test_surface_text_reference_image(
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    font_path = _require_text_cv_font()
    body = make_text(
        "SURFACE",
        depth=0.08,
        font_size=0.3,
        font_path=str(font_path),
        color="#5b84b1",
        backend="surface",
    )
    render_path = tmp_path / "surface-text.png"
    stl_path = tmp_path / "surface-text.stl"
    render_surface_body_image(body, render_path)
    write_surface_body_stl(body, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.01
    assert stl_stats["facet_count"] > 100
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="surfacebody/text_surface",
        update_dirty_reference_images=update_dirty_reference_images,
    )


@pytest.mark.surfacebody
@pytest.mark.reference_image
def test_surface_heightmap_reference_image(
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    body = heightmap(
        np.asarray(
            [
                [0.0, 0.2, 0.6, 0.9],
                [0.1, 0.5, 0.8, 0.7],
                [0.2, 0.7, 1.0, 0.4],
                [0.0, 0.3, 0.5, 0.2],
            ],
            dtype=float,
        ),
        height=0.6,
        xy_scale=0.3,
        alpha_mode="ignore",
        backend="surface",
    )
    render_path = tmp_path / "surface-heightmap.png"
    stl_path = tmp_path / "surface-heightmap.stl"
    render_surface_body_image(body, render_path)
    write_surface_body_stl(body, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.01
    assert stl_stats["facet_count"] >= 18
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="surfacebody/heightmap_surface",
        update_dirty_reference_images=update_dirty_reference_images,
    )


@pytest.mark.surfacebody
@pytest.mark.reference_image
@pytest.mark.parametrize(
    ("fixture_name", "builder"),
    [
        ("surfacebody/csg_union_box_post", build_csg_union_box_post_fixture),
        ("surfacebody/csg_difference_slot", build_csg_difference_slot_fixture),
    ],
)
def test_surface_csg_reference_images(
    fixture_name: str,
    builder,
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    fixture = builder()
    assert fixture["presentation"] == "triptych"
    assert fixture["evidence_scope"] == "overlap"
    assert fixture["orientation_cue"] == "edge_protrusion"
    left_operand = fixture["left_operand"]
    result_body = fixture["result_body"]
    right_operand = fixture["right_operand"]
    assert _surface_boolean_bounds_relation(left_operand.bounds_estimate(), right_operand.bounds_estimate()) == "overlap"
    render_path = tmp_path / f"{fixture_name.replace('/', '-')}.png"
    stl_path = tmp_path / f"{fixture_name.replace('/', '-')}.stl"
    render_surface_body_triptych_image(left_operand, result_body, right_operand, render_path)
    write_surface_body_stl(result_body, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.01
    assert stats["std_luma"] > 8.0
    assert stl_stats["facet_count"] >= 12
    assert stl_stats["file_size"] > 256
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name=fixture_name,
        update_dirty_reference_images=update_dirty_reference_images,
    )

    slice_z = fixture["slice_z"]
    expected_slice_loops = fixture["expected_slice_loops"]
    orientation_required = bool(fixture["orientation_required"])
    if slice_z is None or expected_slice_loops is None:
        return

    actual_slice_loops = surface_body_section_loops(result_body, float(slice_z))
    left_slice_loops = surface_body_section_loops(left_operand, float(slice_z))
    right_slice_loops = surface_body_section_loops(right_operand, float(slice_z))
    result_area = _planar_loop_total_area(actual_slice_loops)
    left_area = _planar_loop_total_area(left_slice_loops)
    right_area = _planar_loop_total_area(right_slice_loops)
    shared_bounds = planar_loop_bounds(expected_slice_loops, actual_slice_loops)
    slice_basename = fixture_name.split("/", 1)[1]
    expected_path = tmp_path / f"{slice_basename}-expected.png"
    actual_path = tmp_path / f"{slice_basename}-actual.png"
    diff_path = tmp_path / f"{slice_basename}-diff.png"

    render_planar_section_bitmap(expected_slice_loops, expected_path, bounds=shared_bounds)
    render_planar_section_bitmap(actual_slice_loops, actual_path, bounds=shared_bounds)
    render_planar_section_diff_image(expected_slice_loops, actual_slice_loops, diff_path, bounds=shared_bounds)
    bundle = ensure_complete_cv_artifact_bundle(
        CvArtifactBundle(
            contract=_section_bundle_contract(fixture_id=f"surfacebody_sections/{slice_basename}"),
            stage="review",
            artifacts={
                "expected": expected_path,
                "actual": actual_path,
                "diff": diff_path,
            },
        )
    )

    contract = _slice_cv_fixture_contract(
        fixture_id=f"surfacebody_sections/{slice_basename}",
        orientation_required=orientation_required,
    )
    silhouette_comparison = compare_planar_loop_silhouettes(
        expected_slice_loops,
        actual_slice_loops,
        bounds=shared_bounds,
    )
    left_comparison = compare_planar_loop_silhouettes(expected_slice_loops, left_slice_loops)
    right_comparison = compare_planar_loop_silhouettes(expected_slice_loops, right_slice_loops)
    assessment = assess_cv_result(contract, silhouette_comparison.relationship)
    assert assessment.shared_pattern != "different"
    assert assessment.passes_contract
    assert left_comparison.relationship == "different_shape" or abs(result_area - left_area) >= (0.10 * result_area)
    assert right_comparison.relationship == "different_shape" or abs(result_area - right_area) >= (0.10 * result_area)
    if orientation_required:
        rotated_comparison = compare_planar_loop_silhouettes(
            expected_slice_loops,
            _rotate_planar_loops_90(expected_slice_loops),
        )
        rotated_assessment = assess_cv_result(contract, rotated_comparison.relationship)
        assert rotated_assessment.shared_pattern == "transformed"
        assert not rotated_assessment.passes_contract
        assert rotated_comparison.same_orientation_iou < 0.9
        assert silhouette_comparison.same_orientation_iou >= 0.9
    else:
        assert silhouette_comparison.best_rotation_iou >= 0.9

    ensure_reference_image_bundle(
        bundle=bundle,
        reference_root=reference_image_root,
        bundle_name=f"surfacebody_sections/{slice_basename}",
        update_dirty_reference_images=update_dirty_reference_images,
    )


@pytest.mark.surfacebody
@pytest.mark.reference_image
def test_surface_traditional_hinge_reference_image(
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    collection = handoff_hinge_surface(
        make_traditional_hinge_pair(width=24.0, knuckle_count=5, opened_angle_deg=32.0, backend="surface")
    )
    render_path = tmp_path / "surface-hinge-traditional.png"
    stl_path = tmp_path / "surface-hinge-traditional.stl"
    render_surface_consumer_collection_image(collection, render_path)
    write_surface_consumer_collection_stl(collection, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.01
    assert stl_stats["facet_count"] > 100
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="surfacebody/hinge_traditional_pair",
        update_dirty_reference_images=update_dirty_reference_images,
    )


@pytest.mark.surfacebody
@pytest.mark.reference_image
def test_surface_living_hinge_reference_image(
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    collection = handoff_hinge_surface(
        make_living_hinge(width=48.0, height=20.0, hinge_band_width=12.0, slit_pitch=1.8, backend="surface")
    )
    render_path = tmp_path / "surface-hinge-living.png"
    stl_path = tmp_path / "surface-hinge-living.stl"
    render_surface_consumer_collection_image(collection, render_path)
    write_surface_consumer_collection_stl(collection, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.01
    assert stl_stats["facet_count"] > 100
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="surfacebody/hinge_living_panel",
        update_dirty_reference_images=update_dirty_reference_images,
    )


@pytest.mark.surfacebody
@pytest.mark.reference_image
def test_surface_bistable_hinge_reference_image(
    reference_image_root: Path,
    reference_stl_root: Path,
    tmp_path: Path,
    update_dirty_reference_images: bool,
) -> None:
    collection = handoff_hinge_surface(
        make_bistable_hinge(width=40.0, preload_offset=2.0, backend="surface")
    )
    render_path = tmp_path / "surface-hinge-bistable.png"
    stl_path = tmp_path / "surface-hinge-bistable.stl"
    render_surface_consumer_collection_image(collection, render_path)
    write_surface_consumer_collection_stl(collection, stl_path)
    stats = image_signal_stats(render_path)
    stl_stats = stl_signal_stats(stl_path)
    assert stats["occupancy"] > 0.01
    assert stl_stats["facet_count"] > 80
    _ensure_model_reference_fixture(
        render_path=render_path,
        stl_path=stl_path,
        reference_image_root=reference_image_root,
        reference_stl_root=reference_stl_root,
        name="surfacebody/hinge_bistable_blank",
        update_dirty_reference_images=update_dirty_reference_images,
    )
