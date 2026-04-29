from __future__ import annotations

from dataclasses import replace
import impression.modeling.csg as csg_module
import numpy as np
import pytest
import warnings

from tests.csg_reference_fixtures import (
    build_csg_difference_slot_fixture,
    build_csg_union_box_post_fixture,
    surface_body_section_loops,
)
from tests.reference_images import compare_planar_loop_silhouettes
from impression.modeling import (
    BooleanOperationError,
    PlanarSurfacePatch,
    SurfaceBody,
    SurfaceBooleanIntersectionStage,
    SurfaceBooleanOperands,
    SurfaceBooleanResult,
    SurfaceBooleanSplitRecord,
    SurfaceBooleanTrimmedPatchFragment,
    SurfaceShell,
    boolean_difference,
    boolean_intersection,
    boolean_union,
    make_box,
    make_plane,
    make_sphere,
    make_surface_body,
    make_surface_shell,
    prepare_surface_boolean_difference_operands,
    prepare_surface_boolean_operands,
    surface_boolean_overlap_fragments,
    surface_boolean_intersection_stage,
    surface_boolean_result,
)


def _translated(body: SurfaceBody, offset: tuple[float, float, float]) -> SurfaceBody:
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = np.asarray(offset, dtype=float)
    return body.with_transform(matrix)


def _planar_loop_signed_area(loop: np.ndarray) -> float:
    x = loop[:, 0]
    y = loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _planar_loop_total_area(loops: list[np.ndarray] | tuple[np.ndarray, ...]) -> float:
    return float(sum(abs(_planar_loop_signed_area(np.asarray(loop, dtype=float))) for loop in loops))


def test_prepare_surface_boolean_union_operands_accepts_closed_surface_bodies() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.5, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0))

    prepared = prepare_surface_boolean_operands("union", [left, right])

    assert isinstance(prepared, SurfaceBooleanOperands)
    assert prepared.operation == "union"
    assert prepared.operand_count == 2
    assert all(np.allclose(body.transform_matrix, np.eye(4)) for body in prepared.bodies)
    first_bounds = prepared.bodies[0].bounds_estimate()
    second_bounds = prepared.bodies[1].bounds_estimate()
    assert first_bounds[1] <= second_bounds[1]
    assert first_bounds[0] < 0.0
    assert second_bounds[1] > 0.0


def test_prepare_surface_boolean_difference_canonicalizes_base_and_cutters() -> None:
    base = _translated(make_box(size=(2.0, 2.0, 2.0), backend="surface"), (1.0, 0.0, 0.0))
    cutter = _translated(make_box(size=(1.0, 1.0, 3.0), backend="surface"), (1.0, 0.0, 0.0))

    prepared = prepare_surface_boolean_difference_operands(base, [cutter])

    assert prepared.operation == "difference"
    assert prepared.operand_count == 2
    assert all(np.allclose(body.transform_matrix, np.eye(4)) for body in prepared.bodies)
    xmin, xmax, _ymin, _ymax, _zmin, _zmax = prepared.bodies[0].bounds_estimate()
    assert xmin == pytest.approx(0.0)
    assert xmax == pytest.approx(2.0)


def test_prepare_surface_boolean_operands_rejects_open_multi_shell_and_disconnected_inputs() -> None:
    open_shell = make_surface_shell(make_plane(size=(2.0, 2.0), backend="surface").shells[0].patches, connected=True)
    open_body = make_surface_body([open_shell])
    with pytest.raises(BooleanOperationError, match="closed-valid"):
        prepare_surface_boolean_operands("union", [make_box(backend="surface"), open_body])

    shell_a = make_surface_shell([PlanarSurfacePatch(family="planar")])
    shell_b = make_surface_shell([PlanarSurfacePatch(family="planar", origin=(2.0, 0.0, 0.0))])
    multi_shell = make_surface_body([shell_a, shell_b])
    with pytest.raises(BooleanOperationError, match="exactly one shell"):
        prepare_surface_boolean_operands("union", [make_box(backend="surface"), multi_shell])

    disconnected_shell = SurfaceShell(
        patches=(make_box(backend="surface").shells[0].patches[0],),
        connected=False,
    )
    disconnected = make_surface_body([disconnected_shell])
    with pytest.raises(BooleanOperationError, match="connected"):
        prepare_surface_boolean_operands("union", [make_box(backend="surface"), disconnected])


def test_prepare_surface_boolean_operands_rejects_invalid_counts_and_types() -> None:
    with pytest.raises(ValueError, match="at least two"):
        prepare_surface_boolean_operands("union", [make_box(backend="surface")])

    with pytest.raises(ValueError, match="at least one cutter"):
        prepare_surface_boolean_difference_operands(make_box(backend="surface"), [])

    with pytest.raises(TypeError, match="SurfaceBody"):
        prepare_surface_boolean_operands("intersection", [make_box(backend="surface"), object()])  # type: ignore[list-item]

def test_public_surface_boolean_backend_returns_structured_result_without_deprecation() -> None:
    left = make_box(size=(2.0, 2.0, 2.0), backend="surface")
    inside = make_box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), backend="surface")
    far = make_box(size=(1.0, 1.0, 1.0), center=(5.0, 0.0, 0.0), backend="surface")
    overlap = make_box(size=(2.0, 2.0, 2.0), center=(0.5, 0.0, 0.0), backend="surface")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        union_result = boolean_union([left, inside], backend="surface")
        difference_result = boolean_difference(left, [far], backend="surface")
        intersection_result = boolean_intersection([left, overlap], backend="surface")

    assert isinstance(union_result, SurfaceBooleanResult)
    assert isinstance(difference_result, SurfaceBooleanResult)
    assert isinstance(intersection_result, SurfaceBooleanResult)
    assert union_result.operation == "union"
    assert difference_result.operation == "difference"
    assert intersection_result.operation == "intersection"
    assert union_result.status == "succeeded"
    assert difference_result.status == "succeeded"
    assert intersection_result.status == "succeeded"
    assert union_result.classification == "closed"
    assert difference_result.classification == "closed"
    assert intersection_result.classification == "closed"
    assert union_result.body is not None
    assert difference_result.body is not None
    assert intersection_result.body is not None
    assert len(union_result.operands.body_ids) == 2
    assert not [item for item in caught if issubclass(item.category, DeprecationWarning)]


def test_surface_boolean_result_contract_is_structured_and_unsupported_for_now() -> None:
    left = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    right = make_sphere(radius=0.75, backend="surface")
    operands = prepare_surface_boolean_operands("union", [left, right])

    result = surface_boolean_result("union", operands)

    assert isinstance(result, SurfaceBooleanResult)
    assert result.operation == "union"
    assert result.operands is operands
    assert result.status == "unsupported"
    assert result.body is None
    assert result.classification is None
    assert "not implemented yet" in str(result.failure_reason)
    assert result.body_id is None


def test_surface_boolean_result_supports_empty_success_for_disjoint_intersection() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-2.0, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (2.0, 0.0, 0.0))
    operands = prepare_surface_boolean_operands("intersection", [left, right])

    result = surface_boolean_result("intersection", operands)

    assert result.status == "succeeded"
    assert result.classification == "empty"
    assert result.body is None
    assert result.body_id is None


def test_surface_boolean_result_supports_empty_success_for_touching_intersection() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-1.0, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.0, 0.0, 0.0))
    operands = prepare_surface_boolean_operands("intersection", [left, right])

    result = surface_boolean_result("intersection", operands)
    stage = surface_boolean_intersection_stage(operands)

    assert stage.body_relation == "touching"
    assert result.status == "succeeded"
    assert result.classification == "empty"
    assert result.body is None


def test_surface_boolean_result_supports_multi_shell_union_for_disjoint_boxes() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-2.0, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (2.0, 0.0, 0.0))
    operands = prepare_surface_boolean_operands("union", [left, right])

    result = surface_boolean_result("union", operands)

    assert result.status == "succeeded"
    assert result.classification == "closed"
    assert result.body is not None
    assert result.body.shell_count == 2


def test_surface_boolean_result_supports_exact_reuse_for_equal_operands() -> None:
    body = make_box(size=(1.0, 2.0, 3.0), backend="surface")

    union_result = surface_boolean_result("union", prepare_surface_boolean_operands("union", [body, body]))
    intersection_result = surface_boolean_result(
        "intersection",
        prepare_surface_boolean_operands("intersection", [body, body]),
    )

    assert union_result.status == "succeeded"
    assert union_result.body is not None
    assert union_result.body.bounds_estimate() == pytest.approx(body.bounds_estimate())
    assert intersection_result.status == "succeeded"
    assert intersection_result.body is not None
    assert intersection_result.body.bounds_estimate() == pytest.approx(body.bounds_estimate())


def test_surface_boolean_result_supports_disjoint_mixed_family_union() -> None:
    box = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-2.0, 0.0, 0.0))
    sphere = _translated(make_sphere(radius=0.5, backend="surface"), (2.0, 0.0, 0.0))
    operands = prepare_surface_boolean_operands("union", [box, sphere])

    result = surface_boolean_result("union", operands)

    assert result.status == "succeeded"
    assert result.classification == "closed"
    assert result.body is not None
    assert result.body.shell_count == 2


def test_surface_boolean_result_supports_overlapping_box_union_with_visible_post() -> None:
    fixture = build_csg_union_box_post_fixture()
    base = fixture["left_operand"]
    post = fixture["right_operand"]
    expected_slice_loops = fixture["expected_slice_loops"]
    slice_z = float(fixture["slice_z"])

    result = surface_boolean_result("union", prepare_surface_boolean_operands("union", [base, post]))

    assert result.status == "succeeded"
    assert result.classification == "closed"
    assert result.body is not None
    assert result.body.shell_count == 1
    assert result.body.patch_count > 6
    assert result.body.bounds_estimate() == pytest.approx(fixture["result_body"].bounds_estimate())

    actual_slice_loops = surface_body_section_loops(result.body, slice_z)
    comparison = compare_planar_loop_silhouettes(expected_slice_loops, actual_slice_loops)
    assert comparison.relationship == "same_shape_same_orientation"
    assert comparison.same_orientation_iou >= 0.95
    assert _planar_loop_total_area(actual_slice_loops) == pytest.approx(
        _planar_loop_total_area(expected_slice_loops),
        rel=0.05,
    )


def test_surface_boolean_result_supports_box_sphere_containment_cases_without_cut_reconstruction() -> None:
    box = make_box(size=(4.0, 4.0, 4.0), backend="surface")
    sphere = make_sphere(radius=0.75, backend="surface")

    union_result = surface_boolean_result("union", prepare_surface_boolean_operands("union", [box, sphere]))
    intersection_result = surface_boolean_result(
        "intersection",
        prepare_surface_boolean_operands("intersection", [box, sphere]),
    )

    assert union_result.status == "succeeded"
    assert union_result.body is not None
    assert union_result.body.bounds_estimate() == pytest.approx(box.bounds_estimate())
    assert intersection_result.status == "succeeded"
    assert intersection_result.body is not None
    assert intersection_result.body.bounds_estimate() == pytest.approx(sphere.bounds_estimate())


def test_surface_boolean_result_supports_box_contained_by_sphere_without_cut_reconstruction() -> None:
    box = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    sphere = make_sphere(radius=2.0, backend="surface")

    result = surface_boolean_result(
        "intersection",
        prepare_surface_boolean_operands("intersection", [box, sphere]),
    )

    assert result.status == "succeeded"
    assert result.body is not None
    assert result.body.bounds_estimate() == pytest.approx(box.bounds_estimate())


def test_surface_boolean_result_supports_overlapping_box_difference_with_side_slot() -> None:
    fixture = build_csg_difference_slot_fixture()
    base = fixture["left_operand"]
    cutter = fixture["right_operand"]
    expected_slice_loops = fixture["expected_slice_loops"]
    slice_z = float(fixture["slice_z"])

    result = surface_boolean_result("difference", prepare_surface_boolean_difference_operands(base, [cutter]))

    assert result.status == "succeeded"
    assert result.classification == "closed"
    assert result.body is not None
    assert result.body.shell_count == 1
    assert result.body.patch_count > 6
    assert result.body.bounds_estimate() == pytest.approx(base.bounds_estimate())

    actual_slice_loops = surface_body_section_loops(result.body, slice_z)
    comparison = compare_planar_loop_silhouettes(expected_slice_loops, actual_slice_loops)
    assert comparison.relationship == "same_shape_same_orientation"
    assert comparison.same_orientation_iou >= 0.95
    assert _planar_loop_total_area(actual_slice_loops) == pytest.approx(
        _planar_loop_total_area(expected_slice_loops),
        rel=0.05,
    )


def test_surface_boolean_result_supports_difference_when_cutter_fully_contains_base() -> None:
    base = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    cutter = make_sphere(radius=2.0, backend="surface")

    result = surface_boolean_result(
        "difference",
        prepare_surface_boolean_difference_operands(base, [cutter]),
    )

    assert result.status == "succeeded"
    assert result.classification == "empty"
    assert result.body is None


def test_surface_boolean_result_remains_explicitly_unsupported_for_partial_box_sphere_overlap() -> None:
    box = make_box(size=(2.0, 2.0, 2.0), backend="surface")
    sphere = make_sphere(radius=1.0, center=(0.75, 0.0, 0.0), backend="surface")

    result = surface_boolean_result(
        "intersection",
        prepare_surface_boolean_operands("intersection", [box, sphere]),
    )

    assert result.status == "unsupported"
    assert result.body is None
    assert result.classification is None
    assert "not implemented yet" in str(result.failure_reason)


def test_surface_boolean_result_propagates_boolean_metadata_for_supported_results() -> None:
    left = replace(
        make_box(size=(2.0, 2.0, 2.0), backend="surface"),
        metadata={"kernel": {"source": "left"}, "consumer": {"color": "red"}},
    )
    right = replace(
        make_box(size=(1.0, 1.0, 1.0), backend="surface"),
        metadata={"kernel": {"source": "right"}, "consumer": {"label": "tool"}},
    )
    operands = prepare_surface_boolean_operands("union", [left, right])

    result = surface_boolean_result("union", operands)

    assert result.status == "succeeded"
    assert result.body is not None
    kernel_metadata = result.body.kernel_metadata()
    consumer_metadata = result.body.consumer_metadata()
    provenance = {"backend": "surface", "operation": "union", "operand_ids": operands.body_ids}
    assert kernel_metadata["boolean_backend"] == "surface"
    assert kernel_metadata["boolean_operation"] == "union"
    assert tuple(kernel_metadata["boolean_operand_ids"]) == operands.body_ids
    assert kernel_metadata["boolean_provenance"] == provenance
    assert consumer_metadata["boolean_backend"] == "surface"
    assert consumer_metadata["boolean_operation"] == "union"
    assert tuple(consumer_metadata["boolean_operand_ids"]) == operands.body_ids
    assert consumer_metadata["boolean_provenance"] == provenance
    assert consumer_metadata["color"] == "red"
    assert consumer_metadata["label"] == "tool"


def test_surface_boolean_result_applies_bounded_cleanup_to_supported_results(monkeypatch: pytest.MonkeyPatch) -> None:
    original_box_builder = csg_module._surface_box_body_from_bounds

    def _dirty_box_body(bounds: tuple[float, float, float, float, float, float], *, metadata: dict[str, object]) -> SurfaceBody:
        body = original_box_builder(bounds, metadata=metadata)
        shell = body.iter_shells(world=True)[0]
        duplicate_seam = replace(shell.seams[0], seam_id=f"{shell.seams[0].seam_id}-duplicate")
        dirty_shell = replace(shell, seams=shell.seams + (duplicate_seam,))
        return make_surface_body((dirty_shell,), metadata=body.metadata)

    monkeypatch.setattr(csg_module, "_surface_box_body_from_bounds", _dirty_box_body)

    result = surface_boolean_result(
        "intersection",
        prepare_surface_boolean_operands(
            "intersection",
            [
                _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.5, 0.0, 0.0)),
                _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0)),
            ],
        ),
    )

    assert result.status == "succeeded"
    assert result.body is not None
    cleaned_shell = result.body.iter_shells(world=True)[0]
    assert len(cleaned_shell.seams) == 12
    assert len(
        {
            tuple(sorted((boundary.patch_index, boundary.boundary_id) for boundary in seam.boundaries))
            for seam in cleaned_shell.seams
        }
    ) == 12


def test_surface_boolean_result_returns_explicit_invalid_status_when_validity_gate_rejects_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_box_builder = csg_module._surface_box_body_from_bounds

    def _invalid_box_body(bounds: tuple[float, float, float, float, float, float], *, metadata: dict[str, object]) -> SurfaceBody:
        body = original_box_builder(bounds, metadata=metadata)
        shell = body.iter_shells(world=True)[0]
        invalid_shell = replace(shell, seams=shell.seams[:-1])
        return make_surface_body((invalid_shell,), metadata=body.metadata)

    monkeypatch.setattr(csg_module, "_surface_box_body_from_bounds", _invalid_box_body)

    result = surface_boolean_result(
        "intersection",
        prepare_surface_boolean_operands(
            "intersection",
            [
                _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.5, 0.0, 0.0)),
                _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0)),
            ],
        ),
    )

    assert result.status == "invalid"
    assert result.body is None
    assert result.classification is None
    assert "validity gate rejected" in str(result.failure_reason)
    assert "missing seam coverage" in str(result.failure_reason)


def test_surface_boolean_intersection_stage_is_deterministic_for_overlapping_boxes() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.5, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0))
    operands = prepare_surface_boolean_operands("union", [left, right])

    stage = surface_boolean_intersection_stage(operands)

    assert isinstance(stage, SurfaceBooleanIntersectionStage)
    assert stage.supported is True
    assert stage.body_relation == "overlap"
    assert len(stage.cut_curves) == 16
    assert all(len(curve.points_3d) == 2 for curve in stage.cut_curves)
    assert all(len(curve.trim_fragments) == 2 for curve in stage.cut_curves)
    assert len(stage.split_records) == len(stage.patch_classifications)
    assert all(isinstance(record, SurfaceBooleanSplitRecord) for record in stage.split_records)
    assert len({curve.cut_curve_id for curve in stage.cut_curves}) == len(stage.cut_curves)
    cut_curve_ids = [curve.cut_curve_id for curve in stage.cut_curves]
    assert cut_curve_ids == sorted(cut_curve_ids)
    patch_lookup = {}
    for operand_index, body in enumerate(operands.bodies):
        shell = body.iter_shells(world=True)[0]
        for patch_index, patch in enumerate(shell.iter_patches(world=True)):
            patch_lookup[(operand_index, patch_index)] = patch
    for curve in stage.cut_curves:
        assert curve.trim_fragments[0].patch == curve.patches[0]
        assert curve.trim_fragments[1].patch == curve.patches[1]
        for point in curve.points_3d:
            assert all(np.isfinite(coord) for coord in point)
        for fragment in curve.trim_fragments:
            patch = patch_lookup[(fragment.patch.operand_index, fragment.patch.patch_index)]
            assert len(fragment.points_uv) == 2
            for u, v in fragment.points_uv:
                assert patch.domain.contains(u, v)
    patch_relations = {classification.patch.operand_index: [] for classification in stage.patch_classifications}
    for classification in stage.patch_classifications:
        patch_relations[classification.patch.operand_index].append(classification.relation)
    assert "inside" in patch_relations[0]
    assert "inside" in patch_relations[1]
    assert any(classification.cut_curve_ids for classification in stage.patch_classifications)
    split_roles = {record.role for record in stage.split_records}
    assert split_roles == {"survive", "discard"}
    assert any(record.role == "discard" and record.relation == "inside" for record in stage.split_records)
    assert any(record.role == "survive" and record.relation == "outside" for record in stage.split_records)


def test_surface_boolean_intersection_stage_supports_disjoint_boxes_with_no_cut_curves() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-2.0, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (2.0, 0.0, 0.0))
    operands = prepare_surface_boolean_operands("intersection", [left, right])

    stage = surface_boolean_intersection_stage(operands)

    assert stage.supported is True
    assert stage.body_relation == "disjoint"
    assert stage.cut_curves == ()
    assert {classification.relation for classification in stage.patch_classifications} == {"outside"}
    assert {record.role for record in stage.split_records} == {"discard"}


def test_surface_boolean_intersection_stage_emits_operation_aware_split_records() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.5, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0))

    union_stage = surface_boolean_intersection_stage(prepare_surface_boolean_operands("union", [left, right]))
    intersection_stage = surface_boolean_intersection_stage(
        prepare_surface_boolean_operands("intersection", [left, right])
    )
    difference_stage = surface_boolean_intersection_stage(
        prepare_surface_boolean_difference_operands(left, [right])
    )

    assert any(record.role == "discard" and record.relation == "inside" for record in union_stage.split_records)
    assert any(record.role == "survive" and record.relation == "outside" for record in union_stage.split_records)

    assert any(record.role == "survive" and record.relation == "inside" for record in intersection_stage.split_records)
    assert any(record.role == "discard" and record.relation == "outside" for record in intersection_stage.split_records)

    cutter_records = [record for record in difference_stage.split_records if record.patch.operand_index == 1]
    base_records = [record for record in difference_stage.split_records if record.patch.operand_index == 0]
    assert any(record.role == "survive" and record.relation == "outside" for record in base_records)
    assert any(record.role == "discard" and record.relation == "inside" for record in base_records)
    assert any(record.role == "cut_cap" and record.relation in {"inside", "on"} for record in cutter_records)


def test_surface_boolean_overlap_fragments_reconstruct_trimmed_box_faces_for_intersection() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.5, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0))
    operands = prepare_surface_boolean_operands("intersection", [left, right])

    fragments = surface_boolean_overlap_fragments(operands)

    assert len(fragments) == 6
    assert all(isinstance(fragment, SurfaceBooleanTrimmedPatchFragment) for fragment in fragments)
    assert len({(fragment.source_patch.operand_index, fragment.source_patch.patch_index) for fragment in fragments}) == 6
    for fragment in fragments:
        assert fragment.patch.outer_trim is not None
        assert fragment.patch.outer_trim.category == "outer"
        assert len(fragment.patch.outer_trim.points_uv) == 4
        assert fragment.patch.outer_trim.area > 0.0
        for u, v in fragment.patch.outer_trim.points_uv:
            assert fragment.patch.domain.contains(float(u), float(v))
    assert any(fragment.cut_curve_ids for fragment in fragments)


def test_surface_boolean_overlap_fragments_are_bounded_to_supported_intersection_slice() -> None:
    box = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    sphere = make_sphere(radius=0.75, backend="surface")

    unsupported = surface_boolean_overlap_fragments(
        prepare_surface_boolean_operands("intersection", [box, sphere])
    )
    wrong_operation = surface_boolean_overlap_fragments(
        prepare_surface_boolean_operands("union", [box, _translated(box, (0.25, 0.0, 0.0))])
    )

    assert unsupported == ()
    assert wrong_operation == ()


def test_surface_boolean_result_overlap_intersection_exposes_single_shell_and_explicit_seams() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.5, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0))

    result = surface_boolean_result(
        "intersection",
        prepare_surface_boolean_operands("intersection", [left, right]),
    )

    assert result.status == "succeeded"
    assert result.classification == "closed"
    assert result.body is not None
    assert result.body.shell_count == 1
    assert result.body.patch_count == 6
    shell = result.body.iter_shells(world=True)[0]
    assert shell.connected is True
    assert len(shell.seams) == 12
    assert all(len(seam.boundaries) == 2 for seam in shell.seams)
    assert all(not seam.is_open for seam in shell.seams)


def test_surface_boolean_result_classifies_supported_initial_slice_outcomes() -> None:
    overlap_left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.5, 0.0, 0.0))
    overlap_right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0))
    disjoint_left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-2.0, 0.0, 0.0))
    disjoint_right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (2.0, 0.0, 0.0))
    contained_box = make_box(size=(4.0, 4.0, 4.0), backend="surface")
    contained_sphere = make_sphere(radius=0.75, backend="surface")

    overlap_result = surface_boolean_result(
        "intersection",
        prepare_surface_boolean_operands("intersection", [overlap_left, overlap_right]),
    )
    disjoint_union = surface_boolean_result(
        "union",
        prepare_surface_boolean_operands("union", [disjoint_left, disjoint_right]),
    )
    no_cut_intersection = surface_boolean_result(
        "intersection",
        prepare_surface_boolean_operands("intersection", [contained_box, contained_sphere]),
    )

    assert overlap_result.classification == "closed"
    assert overlap_result.body is not None
    assert overlap_result.body.shell_count == 1
    assert disjoint_union.classification == "closed"
    assert disjoint_union.body is not None
    assert disjoint_union.body.shell_count == 2
    assert no_cut_intersection.classification == "closed"
    assert no_cut_intersection.body is not None
    assert no_cut_intersection.body.shell_count == 1


def test_surface_boolean_initial_executable_scope_matrix_is_explicit() -> None:
    union_fixture = build_csg_union_box_post_fixture()
    difference_fixture = build_csg_difference_slot_fixture()
    disjoint_union = boolean_union(
        [
            _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-2.0, 0.0, 0.0)),
            _translated(make_sphere(radius=0.5, backend="surface"), (2.0, 0.0, 0.0)),
        ],
        backend="surface",
    )
    overlap_union = boolean_union([union_fixture["left_operand"], union_fixture["right_operand"]], backend="surface")
    overlap_intersection = boolean_intersection(
        [
            _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.5, 0.0, 0.0)),
            _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0)),
        ],
        backend="surface",
    )
    overlap_difference = boolean_difference(
        difference_fixture["left_operand"],
        [difference_fixture["right_operand"]],
        backend="surface",
    )
    no_cut_difference = boolean_difference(
        make_box(size=(1.0, 1.0, 1.0), backend="surface"),
        [make_sphere(radius=2.0, backend="surface")],
        backend="surface",
    )
    unsupported_overlap = boolean_intersection(
        [
            make_box(size=(2.0, 2.0, 2.0), backend="surface"),
            make_sphere(radius=1.0, center=(0.75, 0.0, 0.0), backend="surface"),
        ],
        backend="surface",
    )

    assert isinstance(disjoint_union, SurfaceBooleanResult)
    assert isinstance(overlap_union, SurfaceBooleanResult)
    assert isinstance(overlap_intersection, SurfaceBooleanResult)
    assert isinstance(overlap_difference, SurfaceBooleanResult)
    assert isinstance(no_cut_difference, SurfaceBooleanResult)
    assert isinstance(unsupported_overlap, SurfaceBooleanResult)
    assert disjoint_union.status == "succeeded"
    assert overlap_union.status == "succeeded"
    assert overlap_intersection.status == "succeeded"
    assert overlap_difference.status == "succeeded"
    assert no_cut_difference.status == "succeeded"
    assert unsupported_overlap.status == "unsupported"


def test_surface_boolean_intersection_stage_is_explicitly_unsupported_for_non_box_operands() -> None:
    box = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    sphere = make_sphere(radius=0.75, backend="surface")
    operands = prepare_surface_boolean_operands("intersection", [box, sphere])

    stage = surface_boolean_intersection_stage(operands)

    assert stage.supported is False
    assert "axis-aligned planar box-style operands" in str(stage.support_reason)
