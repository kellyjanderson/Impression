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
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SurfaceBody,
    SurfaceBooleanIntersectionStage,
    SurfaceBooleanOperands,
    SurfaceBooleanPatchRef,
    SurfaceBooleanResult,
    SurfaceBooleanSplitRecord,
    SurfaceBooleanTrimmedPatchFragment,
    SurfaceCSGAnalyticIntersectionRecord,
    SurfaceCSGArrangementDiagnostic,
    SurfaceCSGBoundaryUseProvenanceRecord,
    SurfaceCSGCallerInventoryRecord,
    SurfaceCSGConicDiagnostic,
    SurfaceCSGCutCapRequirementRecord,
    SurfaceCSGCurvePrimitive,
    SurfaceCSGCurveMappingDiagnostic,
    SurfaceCSGFeatureGateDiagnostic,
    SurfaceCSGFragmentClassificationDiagnostic,
    SurfaceCSGFragmentClassificationRecord,
    SurfaceCSGFragmentProvenanceRecord,
    SurfaceCSGOperationSelectionRecord,
    SurfaceCSGPatchLocalCurve,
    SurfaceCSGPatchLocalArrangementGraph,
    SurfaceCSGPatchLocalCurveMappingResult,
    SurfaceCSGPlanarRelationDiagnostic,
    SurfaceCSGProvenanceMetadataRecord,
    SurfaceCSGRevolutionIntersectionRecord,
    SurfaceCSGShellAssemblyRecord,
    SurfaceCSGSeamRebuildRecord,
    SurfaceCSGSplitTrimLoopRecord,
    SurfaceCSGToleranceDiagnostic,
    SurfaceCSGTolerancePolicy,
    SurfaceCSGValidityDiagnostic,
    SurfaceCSGValidityGateRecord,
    SurfaceShell,
    TrimLoop,
    assert_no_hidden_surface_csg_mesh_fallback,
    boolean_difference,
    boolean_intersection,
    boolean_union,
    assemble_surface_csg_shells_from_fragments,
    build_surface_csg_patch_arrangement,
    classify_surface_csg_fragment_against_body,
    classify_surface_csg_point_against_bounds,
    intersect_axis_compatible_revolution_pair,
    intersect_planar_linear_patch_pair,
    intersect_planar_revolution_patch_pair,
    finalize_surface_csg_validity_gate,
    map_surface_csg_curve_to_patch_local,
    make_surface_csg_curve,
    make_surface_csg_line_curve,
    make_box,
    make_plane,
    make_sphere,
    make_surface_body,
    make_surface_shell,
    prepare_surface_boolean_difference_operands,
    prepare_surface_boolean_operands,
    rebuild_surface_csg_shell_seams,
    sort_surface_csg_curves,
    surface_csg_caller_inventory,
    surface_csg_curve_digest,
    surface_csg_curve_key,
    surface_csg_curves_equal,
    surface_csg_feature_gate,
    surface_boolean_overlap_fragments,
    surface_boolean_intersection_stage,
    surface_boolean_result,
    select_surface_csg_fragment_sample,
    select_surface_csg_operation_fragment,
    select_surface_csg_operation_fragments,
    surface_csg_selection_is_empty,
    validate_surface_csg_curve,
    validate_surface_csg_patch_local_curve_domain,
)


def test_surface_csg_curve_primitives_have_deterministic_payload_keys_and_digests() -> None:
    policy = SurfaceCSGTolerancePolicy(snap_tolerance=1e-6, equality_tolerance=1e-6)
    first = make_surface_csg_line_curve((0.0, 0.0, 0.0), (1.0000004, 0.0, 0.0), policy=policy)
    second = make_surface_csg_line_curve((0.0, 0.0, 0.0), (1.00000049, 0.0, 0.0), policy=policy)
    sampled = make_surface_csg_curve(
        "sampled",
        ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
        policy=policy,
    )

    assert isinstance(first, SurfaceCSGCurvePrimitive)
    assert first.canonical_payload(policy)["kind"] == "line"
    assert surface_csg_curve_key(first, policy=policy) == surface_csg_curve_key(second, policy=policy)
    assert surface_csg_curve_digest(first, policy=policy) == surface_csg_curve_digest(second, policy=policy)
    assert surface_csg_curves_equal(first, second, policy=policy)
    assert sort_surface_csg_curves((sampled, first), policy=policy) == tuple(
        sorted((sampled, first), key=lambda curve: surface_csg_curve_key(curve, policy=policy))
    )


def test_surface_csg_tolerance_policy_reports_degenerate_and_ambiguous_curves() -> None:
    policy = SurfaceCSGTolerancePolicy(degeneracy_tolerance=1e-3)
    degenerate = SurfaceCSGCurvePrimitive(
        kind="line",
        points_3d=((0.0, 0.0, 0.0), (0.0, 0.0, 5e-4)),
    )
    ambiguous_arc = SurfaceCSGCurvePrimitive(
        kind="arc",
        points_3d=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )

    degenerate_diagnostics = validate_surface_csg_curve(degenerate, policy=policy)
    ambiguous_diagnostics = validate_surface_csg_curve(ambiguous_arc, policy=policy)

    assert all(isinstance(diagnostic, SurfaceCSGToleranceDiagnostic) for diagnostic in degenerate_diagnostics)
    assert degenerate_diagnostics[0].code == "degenerate-curve"
    assert ambiguous_diagnostics[0].code == "ambiguous-curve"
    with pytest.raises(ValueError, match="positive finite"):
        SurfaceCSGTolerancePolicy(snap_tolerance=0.0)
    with pytest.raises(ValueError, match="degeneracy tolerance"):
        make_surface_csg_line_curve((0.0, 0.0, 0.0), (0.0, 0.0, 5e-4), policy=policy)


def test_surface_csg_curve_maps_to_planar_patch_local_domain() -> None:
    patch = PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 0.0), u_axis=(2.0, 0.0, 0.0), v_axis=(0.0, 3.0, 0.0))
    patch_ref = SurfaceBooleanPatchRef(operand_index=0, patch_index=2)
    curve = make_surface_csg_line_curve((0.5, 0.75, 0.0), (1.5, 2.25, 0.0))

    result = map_surface_csg_curve_to_patch_local(curve, patch_ref, patch)

    assert isinstance(result, SurfaceCSGPatchLocalCurveMappingResult)
    assert result.supported is True
    assert isinstance(result.curve, SurfaceCSGPatchLocalCurve)
    assert result.curve.patch == patch_ref
    assert np.allclose(result.curve.points_uv, ((0.25, 0.25), (0.75, 0.75)))
    assert result.curve.source_curve_digest == surface_csg_curve_digest(curve)
    assert result.diagnostics == ()


def test_surface_csg_curve_mapping_refuses_outside_domain_and_singular_revolution() -> None:
    patch_ref = SurfaceBooleanPatchRef(operand_index=1, patch_index=0)
    planar_patch = PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 0.0), u_axis=(1.0, 0.0, 0.0), v_axis=(0.0, 1.0, 0.0))
    outside_curve = make_surface_csg_line_curve((0.25, 0.25, 0.0), (2.0, 0.25, 0.0))
    revolution_patch = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=((1.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
    )
    singular_curve = make_surface_csg_line_curve((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    outside = map_surface_csg_curve_to_patch_local(outside_curve, patch_ref, planar_patch)
    singular = map_surface_csg_curve_to_patch_local(singular_curve, patch_ref, revolution_patch)

    assert outside.supported is False
    assert all(isinstance(diagnostic, SurfaceCSGCurveMappingDiagnostic) for diagnostic in outside.diagnostics)
    assert outside.diagnostics[0].code == "outside-domain"
    assert singular.supported is False
    assert singular.diagnostics[0].code == "ambiguous-curve"


def test_surface_csg_curve_maps_to_revolution_patch_local_domain() -> None:
    patch = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=((1.0, 0.0, 0.0), (1.0, 0.0, 2.0)),
    )
    patch_ref = SurfaceBooleanPatchRef(operand_index=0, patch_index=1)
    curve = make_surface_csg_line_curve((1.0, 0.0, 0.5), (0.0, 1.0, 1.5))

    result = map_surface_csg_curve_to_patch_local(curve, patch_ref, patch)

    assert result.supported is True
    assert result.curve is not None
    assert result.curve.points_uv[0] == pytest.approx((0.0, 0.25))
    assert result.curve.points_uv[1] == pytest.approx((0.25, 0.75))
    assert validate_surface_csg_patch_local_curve_domain(result.curve) == ()


def test_surface_csg_patch_arrangement_preserves_loop_category_orientation_and_cut_ids() -> None:
    patch = PlanarSurfacePatch(family="planar")
    patch_ref = SurfaceBooleanPatchRef(0, 0)
    curve = make_surface_csg_line_curve((0.5, 0.0, 0.0), (0.5, 1.0, 0.0))
    mapping = map_surface_csg_curve_to_patch_local(curve, patch_ref, patch)
    assert mapping.curve is not None
    loop = TrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), category="outer")

    graph = build_surface_csg_patch_arrangement(
        patch_ref,
        patch,
        patch_local_curves=(mapping.curve,),
        generated_loop=loop,
        cut_curve_ids=("cut-b", "cut-a"),
    )

    assert isinstance(graph, SurfaceCSGPatchLocalArrangementGraph)
    assert graph.supported is True
    assert len(graph.patch_local_curves) == 1
    assert len(graph.split_loops) == 1
    split = graph.split_loops[0]
    assert isinstance(split, SurfaceCSGSplitTrimLoopRecord)
    assert split.source_category == "outer"
    assert split.loop.category == "outer"
    assert split.loop.is_clockwise is False
    assert split.cut_curve_ids == ("cut-a", "cut-b")
    assert graph.diagnostics == ()


def test_surface_csg_patch_arrangement_reports_zero_length_fragments_and_outside_curves() -> None:
    patch = PlanarSurfacePatch(family="planar")
    patch_ref = SurfaceBooleanPatchRef(0, 0)
    outside_curve = SurfaceCSGPatchLocalCurve(
        source_curve_digest="outside",
        patch=patch_ref,
        points_uv=((0.5, 0.5), (1.5, 0.5)),
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
    )
    zero_loop = TrimLoop(((0.0, 0.0), (1e-12, 0.0), (1.0, 0.0)), category="outer")

    graph = build_surface_csg_patch_arrangement(
        patch_ref,
        patch,
        patch_local_curves=(outside_curve,),
        generated_loop=zero_loop,
        cut_curve_ids=("cut-0",),
    )

    assert graph.supported is False
    assert all(isinstance(diagnostic, SurfaceCSGArrangementDiagnostic) for diagnostic in graph.diagnostics)
    assert {diagnostic.code for diagnostic in graph.diagnostics} == {"outside-domain", "zero-length-fragment"}


def test_surface_csg_fragment_classification_distinguishes_inside_outside_and_on_boundary() -> None:
    opposing = make_box(size=(2.0, 2.0, 2.0), backend="surface")
    patch_ref = SurfaceBooleanPatchRef(0, 0)
    inside_patch = PlanarSurfacePatch(family="planar", origin=(-0.25, -0.25, 0.0), u_axis=(0.5, 0.0, 0.0), v_axis=(0.0, 0.5, 0.0))
    outside_patch = PlanarSurfacePatch(family="planar", origin=(2.0, 2.0, 2.0), u_axis=(0.5, 0.0, 0.0), v_axis=(0.0, 0.5, 0.0))
    boundary_patch = PlanarSurfacePatch(family="planar", origin=(-0.5, -0.5, 1.0), u_axis=(1.0, 0.0, 0.0), v_axis=(0.0, 1.0, 0.0))

    inside = classify_surface_csg_fragment_against_body(patch_ref, inside_patch, opposing)
    outside = classify_surface_csg_fragment_against_body(patch_ref, outside_patch, opposing)
    boundary = classify_surface_csg_fragment_against_body(
        patch_ref,
        boundary_patch,
        opposing,
        cut_curve_ids=("cut-on-boundary",),
    )

    assert isinstance(inside, SurfaceCSGFragmentClassificationRecord)
    assert inside.relation == "inside"
    assert outside.relation == "outside"
    assert boundary.relation == "on"
    assert boundary.cut_curve_ids == ("cut-on-boundary",)
    assert boundary.supported is True
    assert classify_surface_csg_point_against_bounds((0.0, 0.0, 1.0), opposing.bounds_estimate()) == "on"


def test_surface_csg_fragment_classification_reports_ambiguous_and_domain_failures() -> None:
    opposing = make_box(size=(2.0, 2.0, 2.0), backend="surface")
    patch_ref = SurfaceBooleanPatchRef(0, 0)
    boundary_patch = PlanarSurfacePatch(family="planar", origin=(-0.5, -0.5, 1.0), u_axis=(1.0, 0.0, 0.0), v_axis=(0.0, 1.0, 0.0))
    outside_loop = TrimLoop(((2.0, 2.0), (3.0, 2.0), (2.5, 3.0)), category="outer")

    ambiguous = classify_surface_csg_fragment_against_body(patch_ref, boundary_patch, opposing)
    outside_domain = classify_surface_csg_fragment_against_body(
        patch_ref,
        boundary_patch,
        opposing,
        trim_loop=outside_loop,
    )

    assert all(isinstance(diagnostic, SurfaceCSGFragmentClassificationDiagnostic) for diagnostic in ambiguous.diagnostics)
    assert ambiguous.diagnostics[0].code == "ambiguous-boundary"
    assert outside_domain.supported is False
    assert outside_domain.diagnostics[0].code == "outside-domain"
    assert select_surface_csg_fragment_sample(boundary_patch, trim_loop=outside_loop) == pytest.approx((2.5, 7.0 / 3.0))


def test_surface_csg_operation_selection_tables_cover_all_operations_and_relations() -> None:
    classifications = tuple(
        SurfaceCSGFragmentClassificationRecord(
            patch=SurfaceBooleanPatchRef(operand_index, index),
            relation=relation,
            sample_uv=(0.5, 0.5),
            sample_point=(0.0, 0.0, 0.0),
            cut_curve_ids=(f"cut-{relation}",),
        )
        for index, relation in enumerate(("inside", "outside", "on"))
        for operand_index in (0, 1)
    )

    union = select_surface_csg_operation_fragments("union", classifications)
    intersection = select_surface_csg_operation_fragments("intersection", classifications)
    difference = select_surface_csg_operation_fragments("difference", classifications)

    assert all(isinstance(record, SurfaceCSGOperationSelectionRecord) for record in union)
    assert {record.relation: record.role for record in union if record.patch.operand_index == 0} == {
        "inside": "discard",
        "outside": "survive",
        "on": "survive",
    }
    assert {record.relation: record.role for record in intersection if record.patch.operand_index == 0} == {
        "inside": "survive",
        "outside": "discard",
        "on": "survive",
    }
    assert {record.relation: record.role for record in difference if record.patch.operand_index == 0} == {
        "inside": "discard",
        "outside": "survive",
        "on": "survive",
    }
    assert {record.relation: record.role for record in difference if record.patch.operand_index == 1} == {
        "inside": "cut_cap",
        "outside": "discard",
        "on": "cut_cap",
    }
    assert all(isinstance(record.cut_cap, SurfaceCSGCutCapRequirementRecord) for record in difference)
    assert all(record.cut_cap.required for record in difference if record.role == "cut_cap")


def test_surface_csg_operation_selection_reports_empty_result() -> None:
    discarded = (
        select_surface_csg_operation_fragment(
            "intersection",
            SurfaceCSGFragmentClassificationRecord(
                patch=SurfaceBooleanPatchRef(0, 0),
                relation="outside",
                sample_uv=(0.5, 0.5),
                sample_point=(2.0, 2.0, 2.0),
            ),
        ),
    )

    assert surface_csg_selection_is_empty(discarded) is True
    assert discarded[0].survives is False


def test_surface_csg_caller_inventory_names_surface_and_explicit_mesh_routes() -> None:
    inventory = surface_csg_caller_inventory()
    ids = {record.caller_id for record in inventory}

    assert all(isinstance(record, SurfaceCSGCallerInventoryRecord) for record in inventory)
    assert {
        "csg.boolean_union",
        "csg.boolean_difference",
        "csg.boolean_intersection",
        "threading.lower_thread_surface_assembly",
        "hinges.make_traditional_hinge_pair",
        "primitive.boolean_dependent_surface_builders",
    }.issubset(ids)
    assert all(record.surface_route for record in inventory)
    assert all(record.explicit_mesh_route for record in inventory if record.mesh_route is not None)
    assert all("caller_id" in record.canonical_payload() for record in inventory)


def test_surface_csg_feature_gate_reports_supported_and_unsupported_without_mesh_fallback() -> None:
    left = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    right = make_box(size=(1.0, 1.0, 1.0), center=(1.25, 0.0, 0.0), backend="surface")

    supported = surface_csg_feature_gate("fixture.boolean_union", "union", (left, right))
    unsupported = surface_csg_feature_gate("fixture.boolean_union", "union", (left, object()))

    assert isinstance(supported, SurfaceCSGFeatureGateDiagnostic)
    assert supported.supported is True
    assert supported.operand_ids == (left.stable_identity, right.stable_identity)
    assert supported.canonical_payload()["boundary"] == "surface-boolean"
    assert unsupported.supported is False
    assert "no mesh fallback" in unsupported.reason


def test_surface_csg_no_hidden_mesh_fallback_assertion_rejects_mesh_results() -> None:
    mesh = make_box(size=(1.0, 1.0, 1.0), backend="mesh")

    with pytest.raises(BooleanOperationError, match="explicit mesh compatibility"):
        assert_no_hidden_surface_csg_mesh_fallback("fixture.boolean_union", mesh)


def test_surface_csg_shell_assembly_builds_single_shell_with_provenance() -> None:
    fragments = (
        SurfaceBooleanTrimmedPatchFragment(
            source_patch=SurfaceBooleanPatchRef(0, 2),
            patch=PlanarSurfacePatch(family="planar"),
            cut_curve_ids=("cut-a",),
        ),
        SurfaceBooleanTrimmedPatchFragment(
            source_patch=SurfaceBooleanPatchRef(1, 1),
            patch=PlanarSurfacePatch(family="planar", origin=(1.0, 0.0, 0.0)),
            cut_curve_ids=("cut-b",),
        ),
    )

    assembly = assemble_surface_csg_shells_from_fragments("union", fragments)

    assert isinstance(assembly, SurfaceCSGShellAssemblyRecord)
    assert assembly.supported is True
    assert assembly.classification == "closed"
    assert len(assembly.shells) == 1
    assert assembly.shells[0].patch_count == 2
    assert all(isinstance(record, SurfaceCSGFragmentProvenanceRecord) for record in assembly.provenance)
    assert tuple(record.source_patch for record in assembly.provenance) == (SurfaceBooleanPatchRef(0, 2), SurfaceBooleanPatchRef(1, 1))
    assert assembly.to_body() is not None


def test_surface_csg_shell_assembly_represents_empty_and_multi_shell_results() -> None:
    empty = assemble_surface_csg_shells_from_fragments("intersection", ())
    fragments = (
        SurfaceBooleanTrimmedPatchFragment(
            source_patch=SurfaceBooleanPatchRef(0, 0),
            patch=PlanarSurfacePatch(family="planar"),
        ),
        SurfaceBooleanTrimmedPatchFragment(
            source_patch=SurfaceBooleanPatchRef(0, 1),
            patch=PlanarSurfacePatch(family="planar", origin=(2.0, 0.0, 0.0)),
        ),
    )
    multi = assemble_surface_csg_shells_from_fragments("union", fragments, multi_shell=True)

    assert empty.classification == "empty"
    assert empty.to_body() is None
    assert multi.classification == "closed"
    assert len(multi.shells) == 2
    assert tuple(record.result_shell_index for record in multi.provenance) == (0, 1)


def test_surface_csg_seam_rebuild_deduplicates_seams_and_records_boundary_uses() -> None:
    body = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    shell = body.iter_shells(world=True)[0]
    duplicate = replace(shell.seams[0], seam_id=f"{shell.seams[0].seam_id}-copy")
    dirty_shell = replace(shell, seams=shell.seams + (duplicate,))

    record = rebuild_surface_csg_shell_seams(dirty_shell)

    assert isinstance(record, SurfaceCSGSeamRebuildRecord)
    assert record.supported is True
    assert len(record.shell.seams) == len(shell.seams)
    assert all(isinstance(boundary_use, SurfaceCSGBoundaryUseProvenanceRecord) for boundary_use in record.boundary_uses)
    assert all(boundary_use.use_count == 1 for boundary_use in record.boundary_uses)
    assert record.diagnostics == ()


def test_surface_csg_seam_rebuild_reports_open_boundaries() -> None:
    body = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    shell = body.iter_shells(world=True)[0]
    open_shell = replace(shell, seams=shell.seams[:-1])

    record = rebuild_surface_csg_shell_seams(open_shell)

    assert record.supported is False
    assert "missing seam coverage" in record.diagnostics[0]
    assert any(boundary_use.use_count == 0 for boundary_use in record.boundary_uses)


def test_surface_csg_validity_gate_accepts_closed_body_and_records_provenance() -> None:
    left = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    right = make_box(size=(1.0, 1.0, 1.0), center=(1.25, 0.0, 0.0), backend="surface")
    operands = prepare_surface_boolean_operands("union", [left, right])

    gate = finalize_surface_csg_validity_gate("union", operands, left)

    assert isinstance(gate, SurfaceCSGValidityGateRecord)
    assert gate.accepted is True
    assert gate.status == "succeeded"
    assert gate.body is not None
    assert gate.diagnostics == ()
    assert isinstance(gate.provenance, SurfaceCSGProvenanceMetadataRecord)
    assert gate.provenance.canonical_payload() == {
        "backend": "surface",
        "operand_ids": operands.body_ids,
        "operation": "union",
    }


def test_surface_csg_validity_gate_rejects_open_shell_with_diagnostics() -> None:
    left = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    right = make_box(size=(1.0, 1.0, 1.0), center=(1.25, 0.0, 0.0), backend="surface")
    operands = prepare_surface_boolean_operands("union", [left, right])
    shell = left.iter_shells(world=True)[0]
    invalid_body = make_surface_body((replace(shell, seams=shell.seams[:-1]),), metadata=left.metadata)

    gate = finalize_surface_csg_validity_gate("union", operands, invalid_body)

    assert gate.accepted is False
    assert gate.status == "invalid"
    assert gate.body is None
    assert all(isinstance(diagnostic, SurfaceCSGValidityDiagnostic) for diagnostic in gate.diagnostics)
    assert gate.diagnostics[0].code == "invalid-shell"
    assert "missing seam coverage" in gate.diagnostics[0].message


def test_planar_linear_analytic_intersection_emits_exact_line_record() -> None:
    first_ref = SurfaceBooleanPatchRef(0, 0)
    second_ref = SurfaceBooleanPatchRef(1, 0)
    first_patch = PlanarSurfacePatch(
        family="planar",
        origin=(0.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
    )
    second_patch = PlanarSurfacePatch(
        family="planar",
        origin=(0.5, 0.0, -0.5),
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
    )

    record = intersect_planar_linear_patch_pair(first_ref, first_patch, second_ref, second_patch)

    assert isinstance(record, SurfaceCSGAnalyticIntersectionRecord)
    assert record.supported is True
    assert record.relation == "crossing"
    assert record.curve is not None
    assert record.curve.kind == "line"
    assert np.allclose(record.curve.points_3d, ((0.5, 0.0, 0.0), (0.5, 1.0, 0.0)))
    assert len(record.patch_local_curves) == 2
    assert np.allclose(record.patch_local_curves[0].points_uv, ((0.5, 0.0), (0.5, 1.0)))
    assert np.allclose(record.patch_local_curves[1].points_uv, ((0.0, 0.5), (1.0, 0.5)))
    assert record.diagnostics == ()


def test_planar_linear_analytic_intersection_classifies_parallel_coincident_disjoint_and_touching() -> None:
    ref_a = SurfaceBooleanPatchRef(0, 0)
    ref_b = SurfaceBooleanPatchRef(1, 0)
    base = PlanarSurfacePatch(family="planar")
    parallel = PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 1.0))
    coincident = PlanarSurfacePatch(family="planar")
    disjoint = PlanarSurfacePatch(
        family="planar",
        origin=(2.0, 0.0, -0.5),
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
    )
    touching = PlanarSurfacePatch(
        family="planar",
        origin=(1.0, 1.0, -0.5),
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
    )

    records = (
        intersect_planar_linear_patch_pair(ref_a, base, ref_b, parallel),
        intersect_planar_linear_patch_pair(ref_a, base, ref_b, coincident),
        intersect_planar_linear_patch_pair(ref_a, base, ref_b, disjoint),
        intersect_planar_linear_patch_pair(ref_a, base, ref_b, touching),
    )

    assert tuple(record.relation for record in records) == ("parallel", "coincident", "disjoint", "touching")
    assert all(record.supported is False for record in records)
    assert all(isinstance(record.diagnostics[0], SurfaceCSGPlanarRelationDiagnostic) for record in records)


def test_planar_linear_analytic_intersection_gates_ruled_pairs() -> None:
    ref_a = SurfaceBooleanPatchRef(0, 0)
    ref_b = SurfaceBooleanPatchRef(1, 0)
    ruled = RuledSurfacePatch(
        family="ruled",
        start_curve=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        end_curve=((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
    )

    record = intersect_planar_linear_patch_pair(ref_a, PlanarSurfacePatch(family="planar"), ref_b, ruled)

    assert record.relation == "unsupported-linear"
    assert record.supported is False
    assert "Ruled patch analytic intersection is gated" in record.diagnostics[0].message


def test_plane_revolution_intersection_emits_circle_records_for_cylinder_cone_and_sphere() -> None:
    plane_ref = SurfaceBooleanPatchRef(0, 0)
    revolution_ref = SurfaceBooleanPatchRef(1, 0)
    plane = PlanarSurfacePatch(
        family="planar",
        origin=(-2.0, -2.0, 0.5),
        u_axis=(4.0, 0.0, 0.0),
        v_axis=(0.0, 4.0, 0.0),
    )
    cylinder = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=((1.0, 0.0, 0.0), (1.0, 0.0, 2.0)),
    )
    cone = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=((0.0, 0.0, 0.0), (2.0, 0.0, 2.0)),
    )
    sphere = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )

    records = (
        intersect_planar_revolution_patch_pair(plane_ref, plane, revolution_ref, cylinder),
        intersect_planar_revolution_patch_pair(plane_ref, plane, revolution_ref, cone),
        intersect_planar_revolution_patch_pair(
            SurfaceBooleanPatchRef(0, 1),
            PlanarSurfacePatch(
                family="planar",
                origin=(-2.0, -2.0, 0.0),
                u_axis=(4.0, 0.0, 0.0),
                v_axis=(0.0, 4.0, 0.0),
            ),
            revolution_ref,
            sphere,
        ),
    )

    assert all(isinstance(record, SurfaceCSGRevolutionIntersectionRecord) for record in records)
    assert all(record.supported for record in records)
    assert all(record.conic_kind == "circle" for record in records)
    assert all(record.curve is not None and record.curve.kind == "arc" for record in records)
    assert all(len(record.patch_local_curves) == 2 for record in records)
    assert records[0].curve is not None
    assert dict(records[0].curve.parameters)["radius"] == pytest.approx(1.0)
    assert records[1].curve is not None
    assert dict(records[1].curve.parameters)["radius"] == pytest.approx(0.5)
    assert records[2].curve is not None
    assert dict(records[2].curve.parameters)["radius"] == pytest.approx(1.0)


def test_plane_revolution_intersection_reports_degenerate_and_unsupported_cases() -> None:
    plane_ref = SurfaceBooleanPatchRef(0, 0)
    revolution_ref = SurfaceBooleanPatchRef(1, 0)
    oblique_plane = PlanarSurfacePatch(
        family="planar",
        origin=(0.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 1.0),
    )
    horizontal_plane = PlanarSurfacePatch(
        family="planar",
        origin=(-1.0, -1.0, 0.0),
        u_axis=(2.0, 0.0, 0.0),
        v_axis=(0.0, 2.0, 0.0),
    )
    cone = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=((0.0, 0.0, 0.0), (2.0, 0.0, 2.0)),
    )

    oblique = intersect_planar_revolution_patch_pair(plane_ref, oblique_plane, revolution_ref, cone)
    tangent = intersect_planar_revolution_patch_pair(plane_ref, horizontal_plane, revolution_ref, cone)

    assert oblique.supported is False
    assert isinstance(oblique.diagnostics[0], SurfaceCSGConicDiagnostic)
    assert oblique.diagnostics[0].code == "unsupported-oblique-plane"
    assert tangent.supported is False
    assert tangent.diagnostics[0].code == "tangent-or-singular-axis"


def test_axis_compatible_revolution_pair_is_explicitly_gated() -> None:
    first_ref = SurfaceBooleanPatchRef(0, 0)
    second_ref = SurfaceBooleanPatchRef(1, 0)
    cylinder = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=((1.0, 0.0, 0.0), (1.0, 0.0, 2.0)),
    )

    record = intersect_axis_compatible_revolution_pair(first_ref, cylinder, second_ref, cylinder)

    assert record.supported is False
    assert record.diagnostics[0].code == "axis-compatible-revolution-gate"
    assert "recognized" in record.diagnostics[0].message


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
    assert all(isinstance(curve.curve, SurfaceCSGCurvePrimitive) for curve in stage.cut_curves)
    assert {curve.curve.kind for curve in stage.cut_curves if curve.curve is not None} == {"line"}
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
