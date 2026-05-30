from __future__ import annotations

from dataclasses import replace
import impression.modeling.csg as csg_module
import numpy as np
import pytest
import warnings

from impression.mesh import Mesh
from tests.csg_reference_fixtures import (
    build_csg_difference_slot_fixture,
    build_csg_union_box_post_fixture,
    surface_body_section_loops,
)
from tests.reference_images import (
    ExpectedDiagnosticKeyRecord,
    NegativeDiagnosticFixtureRecord,
    compare_planar_loop_silhouettes,
    evaluate_negative_diagnostic_fixture_matrix,
    normalize_diagnostic_snapshot,
)
from impression.modeling import (
    BooleanOperationError,
    BSplineSurfacePatch,
    ImplicitCompositionDiagnostic,
    ImplicitCompositionOperandSignPolicy,
    ImplicitCompositionResult,
    ImplicitOperandFieldAdapterRecord,
    NURBSSurfacePatch,
    Path3D,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SurfaceBody,
    SurfaceBooleanFamilyPairSupport,
    SurfaceBooleanIntersectionStage,
    SurfaceBooleanOperands,
    SurfaceBooleanPatchRef,
    SurfaceBooleanResult,
    SurfaceBooleanSplitRecord,
    SurfaceBooleanTrimmedPatchFragment,
    SurfacePatch,
    SurfaceCSGAnalyticBSplineIntersectionRecord,
    SurfaceCSGAnalyticIntersectionRecord,
    SurfaceCSGAnalyticNURBSIntersectionRecord,
    SurfaceCSGArrangementDiagnostic,
    SurfaceCSGBoundaryExposureDiagnostic,
    SurfaceCSGBoundaryUseProvenanceRecord,
    SurfaceCSGCapConstructionRecord,
    SurfaceCSGCallerInventoryRecord,
    SurfaceCSGConicDiagnostic,
    SurfaceCSGContinuityHandoffDiagnostic,
    SurfaceCSGContinuityHandoffRecord,
    SurfaceCSGCutCapRequirementRecord,
    SurfaceCSGCutBoundaryRecord,
    SurfaceCSGCurvePrimitive,
    SurfaceCSGCurveMappingDiagnostic,
    SurfaceCSGFeatureGateDiagnostic,
    SurfaceCSGCapEligibilityRecord,
    SurfaceCSGClassifiedFragmentSet,
    SurfaceCSGCoincidentOwnershipDiagnostic,
    SurfaceCSGCoincidentOwnershipRecord,
    SurfaceCSGFragmentBuildDiagnostic,
    SurfaceCSGFragmentBuildResult,
    SurfaceCSGFragmentClassificationDiagnostic,
    SurfaceCSGFragmentClassificationEdgeRecord,
    SurfaceCSGFragmentGraphDiagnostic,
    SurfaceCSGFragmentGraphRecord,
    SurfaceCSGFragmentClassificationRecord,
    SurfaceCSGSurfaceFragment,
    SurfaceCSGFragmentProvenanceRecord,
    SurfaceCSGGeneratedCapPatchPayloadRecord,
    SurfaceCSGArrangementVertex,
    SurfaceCSGArrangementEdge,
    SurfaceCSGArrangementFaceCandidate,
    SurfaceCSGOperationFragmentSelectionSet,
    SurfaceCSGOperationPlan,
    SurfaceCSGOperationSelectionDiagnostic,
    SurfaceCSGOrientedFragmentRecord,
    SurfaceCSGOperandOrderingNormalizationRecord,
    SurfaceCSGSolverRegistryDiagnostic,
    SurfaceCSGSolverRegistryRecord,
    SurfaceCSGOperationSelectionRecord,
    SurfaceCSGPairDispatchRecord,
    SurfaceCSGPairFixtureEvidenceReport,
    SurfaceCSGPairFixtureRow,
    SurfaceHeightmapCSGEvidenceReport,
    SurfaceHeightmapCSGFixtureRow,
    SurfaceImplicitCSGEvidenceReport,
    SurfaceImplicitCSGFixtureRow,
    SurfaceDisplacementCSGEvidenceReport,
    SurfaceDisplacementCSGFixtureRow,
    SurfaceSampledImplicitPromotionDecision,
    SurfaceSampledImplicitPromotionEvidenceReport,
    SurfaceSampledImplicitPromotionFixtureRow,
    SurfaceSampledImplicitPromotionLossinessRecord,
    SurfaceSampledImplicitPromotionMatrixReport,
    SurfaceSampledImplicitPromotionPolicyRow,
    SurfaceSampledImplicitPromotionProvenanceRecord,
    SurfaceSampledImplicitReconstructionFeasibilityReport,
    SurfaceSampledImplicitCSGUnsupportedRow,
    SurfaceSampledImplicitCSGUnsupportedRowReport,
    SurfaceCSGPatchLocalCurve,
    SurfaceCSGPatchLocalArrangementGraph,
    SurfaceCSGPatchLocalCurveMappingResult,
    SurfaceCSGPatchLocalRegionLoop,
    SurfaceCSGPatchLocalRegionMappingResult,
    SurfaceCSGIntersectionMappingResult,
    SurfaceCSGPlanarRelationDiagnostic,
    SurfaceCSGPlanDiagnostic,
    SurfaceCSGPostReconstructionValidityDiagnostic,
    SurfaceCSGPersistenceEvidenceRecord,
    SurfaceCSGProvenanceMetadataRecord,
    SurfaceCSGProvenanceDiagnostic,
    SurfaceCSGReconstructionDiagnostic,
    SurfaceCSGReferencePromotionReport,
    SurfaceCSGResultPatchProvenanceRecord,
    SurfaceCSGResultProvenanceMap,
    SurfaceCSGRuntimeValidityReport,
    SurfaceCSGRevolutionIntersectionRecord,
    SurfaceCSGShellAssemblyRecord,
    SurfaceCSGShellOrderingRecord,
    SurfaceCSGSeamRebuildRecord,
    SurfaceCSGSplineCoincidentRegionRecord,
    SurfaceCSGSplinePairIntersectionRecord,
    SurfaceCSGSplitTrimLoopRecord,
    SurfaceCSGSubdivisionPairIntersectionRecord,
    SurfaceCSGSweepPairIntersectionRecord,
    SurfaceCSGTessellationBoundaryEvidenceRecord,
    SurfaceCSGToleranceDiagnostic,
    SurfaceCSGTolerancePolicy,
    SurfaceCSGTrimAttachmentRecord,
    SurfaceCSGUnsupportedCapDiagnostic,
    SurfaceCSGValidityDiagnostic,
    SurfaceCSGValidityGateRecord,
    SurfaceCSGValidityHandoffRecord,
    SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX,
    SURFACE_BOOLEAN_OPERATIONS,
    SURFACE_CSG_SOLVER_REGISTRY,
    HIGHER_ORDER_CSG_FIXTURE_PAIR_CLASSES,
    SurfaceShell,
    SubdivisionSurfacePatch,
    SweepSurfacePatch,
    TrimLoop,
    assert_surface_csg_solver_registry_complete,
    assert_no_hidden_surface_csg_mesh_fallback,
    boolean_difference,
    boolean_intersection,
    boolean_union,
    build_surface_csg_solver_registry,
    assemble_surface_csg_result_shells,
    assemble_surface_csg_shells_from_fragments,
    build_surface_csg_cap_patches,
    build_surface_csg_cut_boundary_trims,
    build_surface_csg_fragments_from_arrangement,
    build_surface_csg_fragment_graph,
    build_surface_csg_patch_arrangement,
    build_surface_csg_result_provenance_map,
    check_surface_csg_runtime_result_validity,
    classify_surface_csg_cap_eligibility,
    classify_surface_csg_fragments_against_body,
    classify_surface_csg_fragment_against_body,
    classify_surface_csg_point_against_bounds,
    detect_spline_nurbs_coincident_regions,
    enumerate_higher_order_csg_pair_fixture_rows,
    enumerate_heightmap_csg_fixture_rows,
    enumerate_sampled_implicit_csg_unsupported_rows,
    build_sampled_implicit_promotion_matrix,
    build_sampled_implicit_promotion_provenance_record,
    build_sampled_implicit_reconstruction_refusal,
    evaluate_sampled_implicit_reconstruction_feasibility,
    enumerate_sampled_implicit_promotion_fixture_rows,
    sampled_implicit_reconstruction_criteria,
    sampled_implicit_promotion_metadata_payload,
    select_sampled_implicit_promotion_target,
    verify_sampled_implicit_promotion_fixture_evidence_matrix,
    intersect_analytic_bspline_patch_pair,
    intersect_analytic_nurbs_patch_pair,
    intersect_axis_compatible_revolution_pair,
    intersect_planar_linear_patch_pair,
    intersect_planar_revolution_patch_pair,
    intersect_spline_nurbs_patch_pair,
    intersect_subdivision_csg_patch_pair,
    intersect_sweep_csg_patch_pair,
    finalize_surface_csg_validity_gate,
    detect_surface_csg_dangling_trims,
    map_surface_csg_coincident_region_loop,
    map_surface_csg_curve_to_affected_patches,
    map_surface_csg_curve_to_patch_local,
    make_surface_csg_curve,
    make_surface_csg_line_curve,
    make_box,
    make_plane,
    make_sphere,
    make_surface_body,
    make_surface_shell,
    normalize_surface_csg_operand_ordering,
    orient_surface_csg_selected_fragment,
    orient_surface_csg_selected_fragments,
    plan_prepared_surface_csg_operation,
    plan_surface_csg_operation,
    prepare_surface_boolean_difference_operands,
    prepare_surface_boolean_operands,
    resolve_surface_csg_coincident_fragment_ownership,
    rebuild_surface_csg_shell_seams,
    record_surface_csg_continuity_handoff,
    sort_surface_csg_curves,
    surface_csg_caller_inventory,
    surface_csg_curve_digest,
    surface_csg_curve_key,
    surface_csg_curves_equal,
    surface_csg_feature_gate,
    surface_csg_solver_support_state,
    surface_boolean_overlap_fragments,
    surface_boolean_intersection_stage,
    surface_boolean_result,
    select_surface_csg_fragment_sample,
    select_surface_csg_operation_fragment,
    select_surface_csg_operation_fragment_set,
    select_surface_csg_operation_fragments,
    surface_csg_selection_is_empty,
    validate_surface_csg_curve,
    validate_surface_csg_patch_local_curve_domain,
    validate_surface_csg_result_handoff,
    verify_surface_csg_persistence_tessellation_evidence,
    verify_higher_order_csg_pair_fixture_matrix,
    verify_heightmap_csg_fixture_evidence_matrix,
    enumerate_displacement_csg_fixture_rows,
    verify_displacement_csg_fixture_evidence_matrix,
    enumerate_implicit_csg_fixture_rows,
    verify_implicit_csg_fixture_evidence_matrix,
    verify_sampled_implicit_csg_unsupported_row_tracker,
    verify_sampled_implicit_promotion_matrix,
    adapt_surface_patch_to_implicit_field,
    compose_implicit_field_csg_result,
    implicit_composition_operand_sign_policies,
)
from impression.modeling.surface import PATCH_FAMILY_CAPABILITY_MATRIX


def _csg_negative_fixture(
    fixture_id: str,
    diagnostic: object,
    *,
    expected_keys: tuple[ExpectedDiagnosticKeyRecord, ...],
) -> NegativeDiagnosticFixtureRecord:
    return NegativeDiagnosticFixtureRecord(
        fixture_id=fixture_id,
        domain="csg",
        expected_keys=expected_keys,
        expected_snapshot=normalize_diagnostic_snapshot(diagnostic, fixture_id=fixture_id),
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


def test_surface_csg_curve_maps_to_all_affected_patch_local_domains() -> None:
    first_patch = PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 0.0), u_axis=(2.0, 0.0, 0.0), v_axis=(0.0, 2.0, 0.0))
    second_patch = PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 0.0), u_axis=(0.0, 2.0, 0.0), v_axis=(2.0, 0.0, 0.0))
    curve = make_surface_csg_line_curve((0.5, 0.25, 0.0), (1.5, 1.75, 0.0))

    result = map_surface_csg_curve_to_affected_patches(
        curve,
        (
            (SurfaceBooleanPatchRef(0, 0), first_patch),
            (SurfaceBooleanPatchRef(1, 3), second_patch),
        ),
    )

    assert isinstance(result, SurfaceCSGIntersectionMappingResult)
    assert result.supported is True
    assert result.diagnostics == ()
    assert len(result.curve_mappings) == 2
    assert all(isinstance(mapping.curve, SurfaceCSGPatchLocalCurve) for mapping in result.curve_mappings)
    assert result.curve_mappings[0].curve is not None
    assert result.curve_mappings[1].curve is not None
    assert np.allclose(result.curve_mappings[0].curve.points_uv, ((0.25, 0.125), (0.75, 0.875)))
    assert np.allclose(result.curve_mappings[1].curve.points_uv, ((0.125, 0.25), (0.875, 0.75)))


def test_surface_csg_curve_mapping_requires_both_affected_patches() -> None:
    patch = PlanarSurfacePatch(family="planar")
    curve = make_surface_csg_line_curve((0.25, 0.25, 0.0), (0.75, 0.75, 0.0))

    result = map_surface_csg_curve_to_affected_patches(curve, ((SurfaceBooleanPatchRef(0, 0), patch),))

    assert result.supported is False
    assert result.diagnostics
    assert result.diagnostics[-1].code == "ambiguous-curve"
    assert "both affected patches" in result.diagnostics[-1].message


def test_analytic_bspline_csg_intersection_emits_patch_local_curves() -> None:
    plane = PlanarSurfacePatch(
        family="planar",
        origin=(0.5, 0.0, 0.0),
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
    )
    spline = BSplineSurfacePatch(family="bspline")

    result = intersect_analytic_bspline_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        plane,
        SurfaceBooleanPatchRef(1, 0),
        spline,
        sample_count=5,
    )

    assert isinstance(result, SurfaceCSGAnalyticBSplineIntersectionRecord)
    assert result.supported is True
    assert result.intersection.quality == "within-tolerance"
    assert result.residual_report.converged is True
    assert result.curves[0].kind == "sampled"
    assert len(result.patch_local_curves) == 2
    assert {curve.patch for curve in result.patch_local_curves} == {
        SurfaceBooleanPatchRef(0, 0),
        SurfaceBooleanPatchRef(1, 0),
    }
    spline_curve = next(curve for curve in result.patch_local_curves if curve.patch == SurfaceBooleanPatchRef(1, 0))
    assert spline_curve.points_uv
    assert result.canonical_payload()["supported"] is True


def test_analytic_bspline_csg_intersection_covers_ruled_and_revolution_pairs() -> None:
    ruled = RuledSurfacePatch(family="ruled")
    revolution = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=((1.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
    )
    vertical_spline = BSplineSurfacePatch(
        family="bspline",
        control_net=[
            [(1.0, 0.0, 0.0), (1.0, 0.0, 1.0)],
            [(1.0, 0.0, 0.0), (1.0, 0.0, 1.0)],
        ],
    )

    ruled_result = intersect_analytic_bspline_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        ruled,
        SurfaceBooleanPatchRef(1, 0),
        BSplineSurfacePatch(family="bspline"),
        sample_count=5,
    )
    revolution_result = intersect_analytic_bspline_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        revolution,
        SurfaceBooleanPatchRef(1, 0),
        vertical_spline,
        sample_count=5,
    )

    assert ruled_result.supported is True
    assert revolution_result.supported is True
    assert len(ruled_result.patch_local_curves) == 2
    assert len(revolution_result.patch_local_curves) == 2


def test_analytic_bspline_csg_intersection_reports_non_convergence_without_mesh() -> None:
    plane = PlanarSurfacePatch(
        family="planar",
        origin=(10.0, 0.0, 0.0),
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
    )
    spline = BSplineSurfacePatch(family="bspline")

    result = intersect_analytic_bspline_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        plane,
        SurfaceBooleanPatchRef(1, 0),
        spline,
        sample_count=5,
    )

    assert result.supported is False
    assert result.diagnostics
    assert result.diagnostics[0].code == "unsupported-family-pair"
    assert "mesh" not in result.diagnostics[0].message.lower()
    assert result.residual_report.converged is False


def test_analytic_nurbs_csg_intersection_emits_rational_patch_local_curves() -> None:
    plane = PlanarSurfacePatch(
        family="planar",
        origin=(0.5, 0.0, 0.0),
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
    )
    nurbs = NURBSSurfacePatch(
        family="nurbs",
        weights=((1.0, 2.0), (1.0, 2.0)),
    )

    result = intersect_analytic_nurbs_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        plane,
        SurfaceBooleanPatchRef(1, 0),
        nurbs,
        sample_count=5,
    )

    assert isinstance(result, SurfaceCSGAnalyticNURBSIntersectionRecord)
    assert result.supported is True
    assert result.weight_diagnostics == ()
    assert result.residual_report.converged is True
    assert result.curves[0].kind == "sampled"
    assert len(result.patch_local_curves) == 2
    nurbs_curve = next(curve for curve in result.patch_local_curves if curve.patch == SurfaceBooleanPatchRef(1, 0))
    assert nurbs_curve.points_uv
    assert result.canonical_payload()["weight_diagnostics"] == []


def test_analytic_nurbs_csg_intersection_covers_ruled_and_revolution_pairs() -> None:
    ruled = RuledSurfacePatch(family="ruled")
    revolution = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=((1.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
    )
    vertical_nurbs = NURBSSurfacePatch(
        family="nurbs",
        control_net=[
            [(1.0, 0.0, 0.0), (1.0, 0.0, 1.0)],
            [(1.0, 0.0, 0.0), (1.0, 0.0, 1.0)],
        ],
    )

    ruled_result = intersect_analytic_nurbs_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        ruled,
        SurfaceBooleanPatchRef(1, 0),
        NURBSSurfacePatch(family="nurbs"),
        sample_count=5,
    )
    revolution_result = intersect_analytic_nurbs_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        revolution,
        SurfaceBooleanPatchRef(1, 0),
        vertical_nurbs,
        sample_count=5,
    )

    assert ruled_result.supported is True
    assert revolution_result.supported is True
    assert len(ruled_result.patch_local_curves) == 2
    assert len(revolution_result.patch_local_curves) == 2


def test_analytic_nurbs_csg_intersection_reports_non_convergence_without_mesh() -> None:
    plane = PlanarSurfacePatch(
        family="planar",
        origin=(10.0, 0.0, 0.0),
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
    )
    nurbs = NURBSSurfacePatch(family="nurbs")

    result = intersect_analytic_nurbs_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        plane,
        SurfaceBooleanPatchRef(1, 0),
        nurbs,
        sample_count=5,
    )

    assert result.supported is False
    assert result.diagnostics
    assert result.diagnostics[0].code == "unsupported-family-pair"
    assert "mesh" not in result.diagnostics[0].message.lower()
    assert result.residual_report.converged is False


@pytest.mark.parametrize(
    ("first", "second"),
    (
        (BSplineSurfacePatch(family="bspline"), BSplineSurfacePatch(family="bspline")),
        (BSplineSurfacePatch(family="bspline"), NURBSSurfacePatch(family="nurbs")),
        (NURBSSurfacePatch(family="nurbs"), NURBSSurfacePatch(family="nurbs")),
    ),
)
def test_spline_nurbs_csg_intersection_emits_patch_local_curve_pairs(
    first: BSplineSurfacePatch | NURBSSurfacePatch,
    second: BSplineSurfacePatch | NURBSSurfacePatch,
) -> None:
    result = intersect_spline_nurbs_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        first,
        SurfaceBooleanPatchRef(1, 0),
        second,
        sample_count=5,
    )

    assert isinstance(result, SurfaceCSGSplinePairIntersectionRecord)
    assert result.supported is True
    assert result.residual_report.converged is True
    assert result.curves
    assert len(result.patch_local_curves) == 2
    assert {curve.patch for curve in result.patch_local_curves} == {
        SurfaceBooleanPatchRef(0, 0),
        SurfaceBooleanPatchRef(1, 0),
    }
    assert result.canonical_payload()["supported"] is True


def test_spline_nurbs_csg_intersection_reports_non_convergence_without_mesh() -> None:
    first = BSplineSurfacePatch(family="bspline")
    second = NURBSSurfacePatch(
        family="nurbs",
        control_net=[
            [(10.0, 0.0, 0.0), (10.0, 1.0, 0.0)],
            [(11.0, 0.0, 0.0), (11.0, 1.0, 0.0)],
        ],
    )

    result = intersect_spline_nurbs_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        first,
        SurfaceBooleanPatchRef(1, 0),
        second,
        sample_count=5,
    )

    assert result.supported is False
    assert result.diagnostics
    assert result.diagnostics[0].code == "unsupported-family-pair"
    assert "mesh" not in result.diagnostics[0].message.lower()
    assert result.residual_report.converged is False


def test_spline_nurbs_coincident_region_detector_maps_overlap_loops() -> None:
    first = BSplineSurfacePatch(family="bspline")
    second = NURBSSurfacePatch(family="nurbs")

    result = detect_spline_nurbs_coincident_regions(
        SurfaceBooleanPatchRef(0, 0),
        first,
        SurfaceBooleanPatchRef(1, 0),
        second,
    )

    assert isinstance(result, SurfaceCSGSplineCoincidentRegionRecord)
    assert result.supported is True
    assert result.intersection.classification == "overlap"
    assert result.intersection.overlap_regions[0].region_id == "spline-coincident-region-0"
    assert len(result.region_mappings) == 2
    assert all(mapping.supported for mapping in result.region_mappings)
    assert result.canonical_payload()["supported"] is True


def test_spline_nurbs_coincident_region_detector_reports_ambiguous_overlap_without_mesh() -> None:
    first = BSplineSurfacePatch(family="bspline")
    second = NURBSSurfacePatch(
        family="nurbs",
        control_net=[
            [(0.1, 0.0, 0.0), (0.1, 1.0, 0.0)],
            [(1.1, 0.0, 0.0), (1.1, 1.0, 0.0)],
        ],
    )

    result = detect_spline_nurbs_coincident_regions(
        SurfaceBooleanPatchRef(0, 0),
        first,
        SurfaceBooleanPatchRef(1, 0),
        second,
    )

    assert result.supported is False
    assert result.diagnostics
    assert result.diagnostics[0].code == "ambiguous-overlap"
    assert "mesh" not in result.diagnostics[0].message.lower()
    assert result.intersection.classification == "unsupported"


@pytest.mark.parametrize(
    ("first", "second"),
    (
        (PlanarSurfacePatch(family="planar"), SweepSurfacePatch(family="sweep")),
        (BSplineSurfacePatch(family="bspline"), SweepSurfacePatch(family="sweep")),
        (NURBSSurfacePatch(family="nurbs"), SweepSurfacePatch(family="sweep")),
        (SweepSurfacePatch(family="sweep"), SweepSurfacePatch(family="sweep")),
        (
            SubdivisionSurfacePatch(
                family="subdivision",
                control_points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            ),
            SweepSurfacePatch(family="sweep"),
        ),
    ),
)
def test_sweep_csg_patch_pair_intersection_emits_patch_local_curves(
    first: SurfacePatch,
    second: SurfacePatch,
) -> None:
    result = intersect_sweep_csg_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        first,
        SurfaceBooleanPatchRef(1, 0),
        second,
        sample_count=5,
    )

    assert isinstance(result, SurfaceCSGSweepPairIntersectionRecord)
    assert result.supported is True
    assert result.residual_report.converged is True
    assert result.curves
    assert len(result.patch_local_curves) == 2
    assert result.canonical_payload()["supported"] is True


def test_sweep_csg_patch_pair_intersection_reports_ambiguity_without_mesh() -> None:
    sweep = SweepSurfacePatch(
        family="sweep",
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]),
    )

    result = intersect_sweep_csg_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        PlanarSurfacePatch(family="planar"),
        SurfaceBooleanPatchRef(1, 0),
        sweep,
        sample_count=5,
    )

    assert result.supported is False
    assert result.ambiguity_diagnostics
    assert result.ambiguity_diagnostics[0].blocking is True
    assert "mesh" not in result.ambiguity_diagnostics[0].message.lower()


@pytest.mark.parametrize(
    ("first", "second"),
    (
        (PlanarSurfacePatch(family="planar"), SubdivisionSurfacePatch(family="subdivision")),
        (BSplineSurfacePatch(family="bspline"), SubdivisionSurfacePatch(family="subdivision")),
        (NURBSSurfacePatch(family="nurbs"), SubdivisionSurfacePatch(family="subdivision")),
        (SubdivisionSurfacePatch(family="subdivision"), SubdivisionSurfacePatch(family="subdivision")),
    ),
)
def test_subdivision_csg_patch_pair_intersection_emits_patch_local_curves(
    first: SurfacePatch,
    second: SurfacePatch,
) -> None:
    result = intersect_subdivision_csg_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        first,
        SurfaceBooleanPatchRef(1, 0),
        second,
        sample_count=5,
    )

    assert isinstance(result, SurfaceCSGSubdivisionPairIntersectionRecord)
    assert result.supported is True
    assert result.adapter_report.converged is True
    assert result.curves
    assert len(result.patch_local_curves) == 2
    assert result.canonical_payload()["supported"] is True


def test_subdivision_csg_patch_pair_intersection_reports_budget_refusal_without_mesh() -> None:
    subdivision = SubdivisionSurfacePatch(family="subdivision", subdivision_level=3)

    result = intersect_subdivision_csg_patch_pair(
        SurfaceBooleanPatchRef(0, 0),
        subdivision,
        SurfaceBooleanPatchRef(1, 0),
        PlanarSurfacePatch(family="planar", origin=(10.0, 0.0, 0.0)),
        sample_count=5,
    )

    assert result.supported is False
    assert result.diagnostics
    assert result.diagnostics[0].code == "budget-exhausted"
    assert "mesh" not in result.diagnostics[0].message.lower()


def test_higher_order_csg_pair_fixture_matrix_covers_promoted_pair_classes_without_mesh() -> None:
    rows = enumerate_higher_order_csg_pair_fixture_rows()
    report = verify_higher_order_csg_pair_fixture_matrix()

    assert rows
    assert isinstance(report, SurfaceCSGPairFixtureEvidenceReport)
    assert report.passed is True
    assert set(report.required_pair_classes) == set(HIGHER_ORDER_CSG_FIXTURE_PAIR_CLASSES)
    assert all(isinstance(row, SurfaceCSGPairFixtureRow) for row in rows)
    assert all(row.mesh_fallback_attempted is False for row in rows)
    assert all(row.executable for row in rows)
    assert set(HIGHER_ORDER_CSG_FIXTURE_PAIR_CLASSES) <= {row.pair_class for row in rows}
    assert {row.expected_category for row in rows} >= {"crossing", "coincident", "boundary"}
    assert report.canonical_payload()["diagnostics"] == []


def test_surface_csg_coincident_region_loop_maps_to_patch_local_trim_space() -> None:
    patch = PlanarSurfacePatch(family="planar")
    patch_ref = SurfaceBooleanPatchRef(0, 1)

    result = map_surface_csg_coincident_region_loop(
        "overlap-1",
        patch_ref,
        patch,
        ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)),
        source_curve_digests=("cut-a", "cut-b"),
    )
    outside = map_surface_csg_coincident_region_loop(
        "overlap-outside",
        patch_ref,
        patch,
        ((1.25, 0.25), (1.75, 0.25), (1.75, 0.75), (1.25, 0.75)),
    )

    assert isinstance(result, SurfaceCSGPatchLocalRegionMappingResult)
    assert result.supported is True
    assert isinstance(result.region_loop, SurfaceCSGPatchLocalRegionLoop)
    assert result.region_loop.source_curve_digests == ("cut-a", "cut-b")
    assert result.region_loop.loop.category == "outer"
    assert result.canonical_payload()["supported"] is True
    assert outside.supported is False
    assert outside.diagnostics[0].code == "outside-domain"


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
    assert all(isinstance(vertex, SurfaceCSGArrangementVertex) for vertex in graph.vertices)
    assert all(isinstance(edge, SurfaceCSGArrangementEdge) for edge in graph.edges)
    assert all(isinstance(face, SurfaceCSGArrangementFaceCandidate) for face in graph.face_candidates)
    assert len(graph.face_candidates) == 1
    assert graph.face_candidates[0].source_category == "outer"
    assert graph.canonical_payload()["supported"] is True
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


def test_surface_csg_fragment_builder_promotes_arrangement_faces_to_surface_fragments() -> None:
    patch = PlanarSurfacePatch(family="planar")
    patch_ref = SurfaceBooleanPatchRef(0, 0)
    loop = TrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), category="outer")
    graph = build_surface_csg_patch_arrangement(
        patch_ref,
        patch,
        generated_loop=loop,
        cut_curve_ids=("cut-b", "cut-a"),
    )

    result = build_surface_csg_fragments_from_arrangement(graph, patch)

    assert isinstance(result, SurfaceCSGFragmentBuildResult)
    assert result.supported is True
    assert result.diagnostics == ()
    assert len(result.fragments) == 1
    fragment = result.fragments[0]
    assert isinstance(fragment, SurfaceCSGSurfaceFragment)
    assert fragment.source_patch == patch_ref
    assert fragment.cut_curve_ids == ("cut-a", "cut-b")
    assert fragment.sample_uv == pytest.approx((0.5, 0.5))
    assert len(fragment.patch.trim_loops) == 1
    assert np.allclose(fragment.patch.trim_loops[0].points_uv, loop.points_uv)
    assert result.canonical_payload()["supported"] is True


def test_surface_csg_fragment_builder_reports_invalid_arrangement_and_missing_faces() -> None:
    patch = PlanarSurfacePatch(family="planar")
    patch_ref = SurfaceBooleanPatchRef(0, 0)
    outside_curve = SurfaceCSGPatchLocalCurve(
        source_curve_digest="outside",
        patch=patch_ref,
        points_uv=((0.5, 0.5), (1.5, 0.5)),
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
    )
    graph = build_surface_csg_patch_arrangement(
        patch_ref,
        patch,
        patch_local_curves=(outside_curve,),
    )

    result = build_surface_csg_fragments_from_arrangement(graph, patch)

    assert result.supported is False
    assert all(isinstance(diagnostic, SurfaceCSGFragmentBuildDiagnostic) for diagnostic in result.diagnostics)
    assert {diagnostic.code for diagnostic in result.diagnostics} == {
        "invalid-arrangement",
        "missing-face-candidate",
    }


def test_surface_csg_fragment_classifier_collects_classifications_and_coincident_ownership() -> None:
    opposing = make_box(size=(2.0, 2.0, 2.0), backend="surface")
    patch = PlanarSurfacePatch(
        family="planar",
        origin=(-0.5, -0.5, 1.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
    )
    patch_ref = SurfaceBooleanPatchRef(0, 0)
    loop = TrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), category="outer")
    graph = build_surface_csg_patch_arrangement(
        patch_ref,
        patch,
        generated_loop=loop,
        cut_curve_ids=("cut-on-boundary",),
    )
    fragments = build_surface_csg_fragments_from_arrangement(graph, patch).fragments

    classified = classify_surface_csg_fragments_against_body(fragments, opposing)

    assert isinstance(classified, SurfaceCSGClassifiedFragmentSet)
    assert classified.supported is True
    assert classified.classifications[0].relation == "on"
    assert classified.classifications[0].cut_curve_ids == ("cut-on-boundary",)
    assert isinstance(classified.coincident_ownership[0], SurfaceCSGCoincidentOwnershipRecord)
    assert classified.coincident_ownership[0].owner_patch == patch_ref
    assert classified.canonical_payload()["supported"] is True


def test_surface_csg_coincident_ownership_refuses_boundary_fragments_without_cut_provenance() -> None:
    opposing = make_box(size=(2.0, 2.0, 2.0), backend="surface")
    patch = PlanarSurfacePatch(
        family="planar",
        origin=(-0.5, -0.5, 1.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
    )
    patch_ref = SurfaceBooleanPatchRef(0, 0)
    loop = TrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), category="outer")
    graph = build_surface_csg_patch_arrangement(patch_ref, patch, generated_loop=loop)
    fragment = build_surface_csg_fragments_from_arrangement(graph, patch).fragments[0]
    classification = classify_surface_csg_fragment_against_body(patch_ref, patch, opposing, trim_loop=loop)

    ownership = resolve_surface_csg_coincident_fragment_ownership(fragment, classification)

    assert ownership.supported is False
    assert isinstance(ownership.diagnostics[0], SurfaceCSGCoincidentOwnershipDiagnostic)
    assert ownership.diagnostics[0].code == "missing-cut-provenance"


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


def test_surface_csg_operation_selection_set_reports_classification_and_ownership_blockers() -> None:
    patch_ref = SurfaceBooleanPatchRef(0, 0)
    fragment = SurfaceCSGSurfaceFragment(
        fragment_id="face-0:fragment0",
        source_patch=patch_ref,
        patch=PlanarSurfacePatch(family="planar", origin=(-0.5, -0.5, 1.0)),
        loop=TrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), category="outer"),
        source_face_id="face-0",
    )
    classification = SurfaceCSGFragmentClassificationRecord(
        patch=patch_ref,
        relation="on",
        sample_uv=(0.5, 0.5),
        sample_point=(0.0, 0.0, 1.0),
        diagnostics=(
            SurfaceCSGFragmentClassificationDiagnostic(
                code="ambiguous-boundary",
                message="ambiguous",
                patch=patch_ref,
                sample_point=(0.0, 0.0, 1.0),
            ),
        ),
    )
    ownership = resolve_surface_csg_coincident_fragment_ownership(fragment, classification)
    classified = SurfaceCSGClassifiedFragmentSet(
        fragments=(fragment,),
        classifications=(classification,),
        coincident_ownership=(ownership,),
    )

    selection_set = select_surface_csg_operation_fragment_set("difference", classified)

    assert isinstance(selection_set, SurfaceCSGOperationFragmentSelectionSet)
    assert selection_set.supported is False
    assert all(isinstance(diagnostic, SurfaceCSGOperationSelectionDiagnostic) for diagnostic in selection_set.diagnostics)
    assert {diagnostic.code for diagnostic in selection_set.diagnostics} == {
        "unsupported-classification",
        "ambiguous-coincident-ownership",
    }


def test_surface_csg_cap_eligibility_distinguishes_planar_missing_and_non_planar_caps() -> None:
    patch_ref = SurfaceBooleanPatchRef(1, 0)
    selection = select_surface_csg_operation_fragment(
        "difference",
        SurfaceCSGFragmentClassificationRecord(
            patch=patch_ref,
            relation="inside",
            sample_uv=(0.5, 0.5),
            sample_point=(0.0, 0.0, 0.0),
            cut_curve_ids=("cut-0",),
        ),
    )

    missing = classify_surface_csg_cap_eligibility(selection)
    planar = classify_surface_csg_cap_eligibility(selection, PlanarSurfacePatch(family="planar"))
    non_planar = classify_surface_csg_cap_eligibility(
        selection,
        RevolutionSurfacePatch(
            family="revolution",
            profile_curve=((1.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
        ),
    )

    assert isinstance(planar, SurfaceCSGCapEligibilityRecord)
    assert missing.supported is False
    assert missing.diagnostics[0].code == "missing-source-patch"
    assert planar.supported is True
    assert planar.cap_family == "planar"
    assert non_planar.supported is False
    assert non_planar.diagnostics[0].code == "unsupported-cap-family"


def test_surface_csg_operation_selection_set_classifies_required_caps() -> None:
    patch_ref = SurfaceBooleanPatchRef(1, 0)
    fragment = SurfaceCSGSurfaceFragment(
        fragment_id="face-0:fragment0",
        source_patch=patch_ref,
        patch=PlanarSurfacePatch(family="planar"),
        loop=TrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), category="outer"),
        source_face_id="face-0",
        cut_curve_ids=("cut-0",),
    )
    classified = SurfaceCSGClassifiedFragmentSet(
        fragments=(fragment,),
        classifications=(
            SurfaceCSGFragmentClassificationRecord(
                patch=patch_ref,
                relation="inside",
                sample_uv=(0.5, 0.5),
                sample_point=(0.0, 0.0, 0.0),
                cut_curve_ids=("cut-0",),
            ),
        ),
    )

    selection_set = select_surface_csg_operation_fragment_set(
        "difference",
        classified,
        source_patches={patch_ref: fragment.patch},
    )

    assert selection_set.supported is True
    assert selection_set.selections[0].role == "cut_cap"
    assert selection_set.cap_eligibility[0].required is True
    assert selection_set.cap_eligibility[0].eligible is True
    assert selection_set.canonical_payload()["supported"] is True


def test_surface_csg_caller_inventory_names_surface_and_explicit_mesh_routes() -> None:
    inventory = surface_csg_caller_inventory()
    ids = {record.caller_id for record in inventory}

    assert all(isinstance(record, SurfaceCSGCallerInventoryRecord) for record in inventory)
    assert {
        "csg.boolean_union",
        "csg.boolean_difference",
        "csg.boolean_intersection",
        "hinges.make_traditional_hinge_pair",
        "primitive.boolean_dependent_surface_builders",
    }.issubset(ids)
    assert all(record.surface_route for record in inventory)
    assert all(record.explicit_mesh_route for record in inventory if record.mesh_route is not None)
    assert all("caller_id" in record.canonical_payload() for record in inventory)


def test_surface_csg_solver_registry_covers_all_promoted_family_pairs() -> None:
    registry = build_surface_csg_solver_registry()

    assert isinstance(registry, SurfaceCSGSolverRegistryRecord)
    assert registry.passed is True
    assert registry.diagnostics == ()
    assert registry is not SURFACE_CSG_SOLVER_REGISTRY
    assert assert_surface_csg_solver_registry_complete(registry) is registry
    assert len(registry.support_records) == (
        len(SURFACE_BOOLEAN_OPERATIONS)
        * len(PATCH_FAMILY_CAPABILITY_MATRIX)
        * len(PATCH_FAMILY_CAPABILITY_MATRIX)
    )
    assert registry.support_for("union", "planar", "revolution").support_state == "exact"
    assert surface_csg_solver_support_state("union", "revolution", "planar", registry=registry) == "exact"
    assert registry.canonical_payload()["passed"] is True


def test_surface_csg_solver_registry_reports_missing_and_unknown_pairs() -> None:
    missing_key = ("union", "planar", "planar")
    extra_key = ("union", "mystery", "planar")
    broken_matrix = {
        key: value
        for key, value in SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX.items()
        if key != missing_key
    }
    broken_matrix[extra_key] = SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX[missing_key]

    registry = build_surface_csg_solver_registry(broken_matrix)

    assert registry.passed is False
    assert all(isinstance(diagnostic, SurfaceCSGSolverRegistryDiagnostic) for diagnostic in registry.diagnostics)
    assert {diagnostic.code for diagnostic in registry.diagnostics} == {"missing-pair", "unknown-pair"}
    with pytest.raises(ValueError, match="missing-pair:union:planar/planar"):
        assert_surface_csg_solver_registry_complete(registry)


def test_sampled_implicit_csg_unsupported_row_tracker_covers_153_in_progress_rows() -> None:
    rows = enumerate_sampled_implicit_csg_unsupported_rows()
    report = verify_sampled_implicit_csg_unsupported_row_tracker()

    assert len(rows) == 153
    assert isinstance(report, SurfaceSampledImplicitCSGUnsupportedRowReport)
    assert report.passed is True
    assert report.diagnostics == ()
    assert report.expected_row_count == 153
    assert report.expected_rows_per_operation == 51
    assert {len(report.rows_for_operation(operation)) for operation in SURFACE_BOOLEAN_OPERATIONS} == {51}
    assert all(isinstance(row, SurfaceSampledImplicitCSGUnsupportedRow) for row in rows)
    assert {row.route_status for row in rows} == {"in-progress"}
    assert {row.support_state for row in rows} == {"unsupported"}
    assert all(row.required_future_capability for row in rows)
    assert all(not row.mesh_fallback_attempted for row in rows)
    assert all(
        row.left_family in {"implicit", "heightmap", "displacement"}
        or row.right_family in {"implicit", "heightmap", "displacement"}
        for row in rows
    )
    assert report.canonical_payload()["passed"] is True


def test_sampled_implicit_csg_unsupported_row_tracker_reports_missing_route_classification() -> None:
    broken_matrix = dict(SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX)
    key = ("union", "implicit", "planar")
    broken_matrix[key] = replace(broken_matrix[key], required_future_capability=None)
    registry = build_surface_csg_solver_registry(broken_matrix)

    report = verify_sampled_implicit_csg_unsupported_row_tracker(registry=registry)

    assert report.passed is False
    assert any(
        diagnostic.left_family == "implicit"
        and diagnostic.right_family == "planar"
        and "future capability or route classification" in diagnostic.message
        for diagnostic in report.diagnostics
    )


def test_sampled_implicit_promotion_matrix_covers_153_rows_without_in_progress_or_mesh_fallback() -> None:
    report = verify_sampled_implicit_promotion_matrix()

    assert isinstance(report, SurfaceSampledImplicitPromotionMatrixReport)
    assert report.passed is True
    assert report.diagnostics == ()
    assert len(report.rows) == 153
    assert report.expected_row_count == 153
    assert report.expected_rows_per_operation == 51
    assert {len(report.rows_for_operation(operation)) for operation in SURFACE_BOOLEAN_OPERATIONS} == {51}
    assert all(isinstance(row, SurfaceSampledImplicitPromotionPolicyRow) for row in report.rows)
    assert {row.route_status for row in report.rows} == {"promotion-route"}
    assert all(row.source_support_state == "unsupported" for row in report.rows)
    assert all(row.target_family is not None for row in report.rows)
    assert all(row.complete for row in report.rows)
    assert all(not row.mesh_fallback_attempted for row in report.rows)
    assert {"implicit", "subdivision", "nurbs", "bspline"} <= {row.target_family for row in report.rows}
    assert report.canonical_payload()["passed"] is True


def test_sampled_implicit_promotion_target_selector_reports_missing_target_without_mesh_fallback() -> None:
    source_row = next(
        row
        for row in enumerate_sampled_implicit_csg_unsupported_rows(operations=("union",))
        if row.left_family == "implicit" and row.right_family == "planar"
    )

    decision = select_sampled_implicit_promotion_target(source_row, allowed_targets=("subdivision",))

    assert isinstance(decision, SurfaceSampledImplicitPromotionDecision)
    assert decision.supported is False
    assert decision.row.route_status == "in-progress"
    assert decision.row.target_family is None
    assert decision.diagnostics[0].code == "missing-target"
    assert decision.diagnostics[0].no_mesh_fallback is True
    assert "no mesh fallback" in decision.diagnostics[0].message.lower()


def test_sampled_implicit_promotion_matrix_detects_missing_target_policy() -> None:
    report = build_sampled_implicit_promotion_matrix(
        operations=("union",),
        allowed_targets=("subdivision",),
    )

    assert report.passed is False
    assert any(diagnostic.code == "missing-target" for diagnostic in report.diagnostics)
    assert any(row.route_status == "in-progress" for row in report.rows)


def test_sampled_implicit_promotion_provenance_records_sources_lossiness_and_tolerance() -> None:
    report = verify_sampled_implicit_promotion_matrix(operations=("union",))
    row = next(
        row
        for row in report.rows
        if row.left_family == "heightmap" and row.right_family == "bspline"
    )

    provenance = build_sampled_implicit_promotion_provenance_record(
        row,
        operand_ids=("heightmap-a", "bspline-b"),
        tolerance=2.5e-6,
    )
    payload = sampled_implicit_promotion_metadata_payload(row, operand_ids=("heightmap-a", "bspline-b"), tolerance=2.5e-6)

    assert isinstance(provenance, SurfaceSampledImplicitPromotionProvenanceRecord)
    assert provenance.supported is True
    assert provenance.diagnostics == ()
    assert provenance.source_families == ("heightmap", "bspline")
    assert provenance.source_operand_ids == ("heightmap-a", "bspline-b")
    assert provenance.target_family == "bspline"
    assert isinstance(provenance.lossiness, SurfaceSampledImplicitPromotionLossinessRecord)
    assert provenance.lossiness.lossiness == "exact-reconstruction"
    assert provenance.lossiness.tolerance == pytest.approx(2.5e-6)
    assert provenance.lossiness.reconstruction_kind == "bspline-fit"
    assert provenance.lossiness.no_mesh_fallback is True
    assert payload["supported"] is True
    assert payload["lossiness"]["reconstruction_kind"] == "bspline-fit"


def test_sampled_implicit_promotion_provenance_reports_invalid_tolerance_and_operands() -> None:
    row = next(
        row
        for row in build_sampled_implicit_promotion_matrix(operations=("union",)).rows
        if row.left_family == "implicit" and row.right_family == "planar"
    )

    provenance = build_sampled_implicit_promotion_provenance_record(
        row,
        operand_ids=("only-one",),
        tolerance=-1.0,
    )

    assert provenance.supported is False
    assert {diagnostic.code for diagnostic in provenance.diagnostics} == {"invalid-operands", "invalid-tolerance"}
    assert all(diagnostic.no_mesh_fallback for diagnostic in provenance.diagnostics)
    assert "no mesh fallback" in provenance.diagnostics[-1].message.lower()


def test_sampled_implicit_reconstruction_criteria_accepts_supported_targets() -> None:
    rows = verify_sampled_implicit_promotion_matrix(operations=("union",)).rows
    targets = {
        "implicit": next(row for row in rows if row.target_family == "implicit"),
        "subdivision": next(row for row in rows if row.target_family == "subdivision"),
        "nurbs": next(row for row in rows if row.target_family == "nurbs"),
        "bspline": next(row for row in rows if row.target_family == "bspline"),
    }

    reports = {}
    for target, row in targets.items():
        provenance = build_sampled_implicit_promotion_provenance_record(row, operand_ids=("left", "right"))
        reports[target] = evaluate_sampled_implicit_reconstruction_feasibility(
            provenance,
            estimated_sample_count=16,
            residual=0.0,
        )

    assert all(isinstance(report, SurfaceSampledImplicitReconstructionFeasibilityReport) for report in reports.values())
    assert all(report.supported for report in reports.values())
    assert reports["implicit"].criteria.target_family == "implicit"
    assert sampled_implicit_reconstruction_criteria("implicit").max_residual is None
    assert sampled_implicit_reconstruction_criteria("subdivision").max_residual == pytest.approx(1e-3)
    assert sampled_implicit_reconstruction_criteria("nurbs").requires_exact_reconstruction is True
    assert sampled_implicit_reconstruction_criteria("bspline").requires_exact_reconstruction is True


def test_sampled_implicit_reconstruction_criteria_refuses_budget_residual_and_incomplete_provenance() -> None:
    row = next(
        row
        for row in verify_sampled_implicit_promotion_matrix(operations=("union",)).rows
        if row.target_family == "nurbs"
    )
    provenance = build_sampled_implicit_promotion_provenance_record(row, operand_ids=("left", "right"))
    report = evaluate_sampled_implicit_reconstruction_feasibility(
        provenance,
        estimated_sample_count=999_999,
        residual=1e-3,
    )

    assert report.supported is False
    assert {diagnostic.code for diagnostic in report.diagnostics} == {"sample-budget-exceeded", "residual-exceeded"}
    assert all(diagnostic.no_mesh_fallback for diagnostic in report.diagnostics)
    assert build_sampled_implicit_reconstruction_refusal(report) == report.diagnostics

    incomplete = next(
        row
        for row in build_sampled_implicit_promotion_matrix(operations=("union",), allowed_targets=("subdivision",)).rows
        if row.route_status == "in-progress"
    )
    incomplete_provenance = build_sampled_implicit_promotion_provenance_record(incomplete)
    incomplete_report = evaluate_sampled_implicit_reconstruction_feasibility(incomplete_provenance)

    assert incomplete_report.supported is False
    assert any(diagnostic.code == "incomplete-provenance" for diagnostic in incomplete_report.diagnostics)


def test_sampled_implicit_promotion_fixture_evidence_matrix_covers_targets_persistence_refusal_and_no_mesh() -> None:
    rows = enumerate_sampled_implicit_promotion_fixture_rows()
    report = verify_sampled_implicit_promotion_fixture_evidence_matrix()

    assert isinstance(report, SurfaceSampledImplicitPromotionEvidenceReport)
    assert report.passed is True
    assert report.diagnostics == ()
    assert set(report.required_route_kinds) <= {row.route_kind for row in rows}
    assert all(isinstance(row, SurfaceSampledImplicitPromotionFixtureRow) for row in rows)
    assert all(row.reference_state == "clean" for row in rows)
    assert all(not row.mesh_fallback_attempted for row in rows)
    assert all(row.passed for row in rows)
    target_rows = [row for row in rows if row.route_kind == "promotion-target"]
    assert {row.target_family for row in target_rows} == {"implicit", "subdivision", "nurbs", "bspline"}
    assert any(row.route_kind == "refusal" and row.target_family is None for row in rows)
    assert any(row.route_kind == "persistence" and row.message.endswith("without mesh truth.") for row in rows)
    assert "no mesh fallback" in next(row.message for row in rows if row.route_kind == "no-mesh-fallback").lower()
    assert report.canonical_payload()["passed"] is True


def test_implicit_composition_operation_sign_policies_are_deterministic() -> None:
    policies = implicit_composition_operand_sign_policies("difference", 3)

    assert all(isinstance(policy, ImplicitCompositionOperandSignPolicy) for policy in policies)
    assert [policy.role for policy in policies] == ["base", "cutter", "cutter"]
    assert [policy.sign for policy in policies] == ["preserve", "negate", "negate"]
    assert implicit_composition_operand_sign_policies("union", 2)[1].role == "member"
    with pytest.raises(ValueError, match="at least two operands"):
        implicit_composition_operand_sign_policies("intersection", 1)


def test_implicit_composition_result_builds_surface_native_union_body() -> None:
    left = adapt_surface_patch_to_implicit_field(
        PlanarSurfacePatch(family="planar"),
        bounds=(-1.0, 1.0, -1.0, 1.0, -0.1, 0.1),
    )
    right = adapt_surface_patch_to_implicit_field(
        PlanarSurfacePatch(family="planar", origin=np.array([0.0, 0.0, 0.5], dtype=float)),
        bounds=(-1.0, 1.0, -1.0, 1.0, 0.4, 0.6),
    )

    result = compose_implicit_field_csg_result("union", (left, right), samples=(3, 3, 3), max_sample_count=27)

    assert isinstance(result, ImplicitCompositionResult)
    assert result.supported is True
    assert result.body is not None
    assert result.patch is not None
    assert result.patch.family == "implicit"
    assert result.patch.field.kind == "union"
    assert result.safety is not None and result.safety.accepted is True
    assert result.operation_record.result_graph is not None
    assert result.operation_record.result_graph.root.kind == "union"
    assert result.patch.metadata["kernel"]["no_mesh_fallback"] is True


def test_implicit_composition_result_preserves_difference_operand_order_and_sign_policy() -> None:
    base = adapt_surface_patch_to_implicit_field(PlanarSurfacePatch(family="planar"))
    cutter = adapt_surface_patch_to_implicit_field(
        PlanarSurfacePatch(family="planar", origin=np.array([0.0, 0.0, 1.0], dtype=float))
    )

    result = compose_implicit_field_csg_result("difference", (base, cutter))

    assert result.supported is True
    assert result.patch is not None
    assert result.patch.field.kind == "difference"
    assert result.operation_record.sign_policies[0].role == "base"
    assert result.operation_record.sign_policies[1].role == "cutter"
    assert result.operation_record.sign_policies[1].sign == "negate"


def test_implicit_composition_refuses_unsupported_adapter_without_mesh_fallback() -> None:
    adapter = ImplicitOperandFieldAdapterRecord(
        family="unsupported-field",
        patch_id="bad",
        adapter_kind="refused",
        supported=False,
        diagnostics=(
            {
                "code": "unsupported-family",
                "message": "No field adapter exists; no mesh fallback was attempted.",
                "family": "unsupported-field",
                "patch_id": "bad",
            },
        ),
    )
    supported = adapt_surface_patch_to_implicit_field(
        PlanarSurfacePatch(family="planar"),
        bounds=(-1.0, 1.0, -1.0, 1.0, -0.1, 0.1),
    )

    result = compose_implicit_field_csg_result("intersection", (supported, adapter))

    assert result.supported is False
    assert isinstance(result.diagnostics[0], ImplicitCompositionDiagnostic)
    assert result.diagnostics[0].code == "unsupported-adapter"
    assert result.diagnostics[0].no_mesh_fallback is True
    assert "mesh fallback" in result.diagnostics[0].message


def test_implicit_composition_refuses_unsafe_result_budget_without_mesh_fallback() -> None:
    left = adapt_surface_patch_to_implicit_field(PlanarSurfacePatch(family="planar"))
    right = adapt_surface_patch_to_implicit_field(
        PlanarSurfacePatch(family="planar", origin=np.array([0.0, 0.0, 1.0], dtype=float))
    )

    result = compose_implicit_field_csg_result("union", (left, right), samples=(4, 4, 4), max_sample_count=4)

    assert result.supported is False
    assert result.safety is not None
    assert result.safety.accepted is False
    assert result.diagnostics[0].code == "unsafe-result"
    assert result.diagnostics[0].no_mesh_fallback is True


def test_implicit_csg_fixture_evidence_matrix_covers_success_refusals_persistence_and_no_mesh() -> None:
    rows = enumerate_implicit_csg_fixture_rows()
    report = verify_implicit_csg_fixture_evidence_matrix()

    assert isinstance(report, SurfaceImplicitCSGEvidenceReport)
    assert report.passed is True
    assert report.diagnostics == ()
    assert {row.route_kind for row in rows} == set(report.required_route_kinds)
    assert all(isinstance(row, SurfaceImplicitCSGFixtureRow) for row in rows)
    assert all(row.reference_state == "clean" for row in rows)
    assert all(not row.mesh_fallback_attempted for row in rows)
    assert all(row.passed for row in rows)
    by_kind = {row.route_kind: row for row in rows}
    assert by_kind["success"].left_family == "planar"
    assert by_kind["adapter-refusal"].right_family == "unsupported-field"
    assert by_kind["persistence"].message.endswith("without mesh truth.")
    assert "no mesh fallback" in by_kind["no-mesh-fallback"].message.lower()
    assert report.canonical_payload()["passed"] is True


def test_heightmap_csg_fixture_evidence_matrix_covers_success_promotion_persistence_and_no_mesh() -> None:
    rows = enumerate_heightmap_csg_fixture_rows()
    report = verify_heightmap_csg_fixture_evidence_matrix()

    assert isinstance(report, SurfaceHeightmapCSGEvidenceReport)
    assert report.passed is True
    assert report.diagnostics == ()
    assert {row.route_kind for row in rows} == set(report.required_route_kinds)
    assert all(isinstance(row, SurfaceHeightmapCSGFixtureRow) for row in rows)
    assert all(row.reference_state == "clean" for row in rows)
    assert all(not row.mesh_fallback_attempted for row in rows)
    assert all(row.passed for row in rows)
    by_kind = {row.route_kind: row for row in rows}
    assert by_kind["success"].operation == "union"
    assert by_kind["representability-refusal"].operation == "intersection"
    assert by_kind["promotion"].target_family == "implicit"
    assert by_kind["persistence"].message.endswith("without mesh truth.")
    assert "no mesh fallback" in by_kind["no-mesh-fallback"].message.lower()
    assert report.canonical_payload()["passed"] is True


def test_displacement_csg_fixture_evidence_matrix_covers_success_promotion_persistence_and_no_mesh() -> None:
    rows = enumerate_displacement_csg_fixture_rows()
    report = verify_displacement_csg_fixture_evidence_matrix()

    assert isinstance(report, SurfaceDisplacementCSGEvidenceReport)
    assert report.passed is True
    assert report.diagnostics == ()
    assert {row.route_kind for row in rows} == set(report.required_route_kinds)
    assert all(isinstance(row, SurfaceDisplacementCSGFixtureRow) for row in rows)
    assert all(row.reference_state == "clean" for row in rows)
    assert all(not row.mesh_fallback_attempted for row in rows)
    assert all(row.passed for row in rows)
    by_kind = {row.route_kind: row for row in rows}
    assert by_kind["success"].operation == "union"
    assert by_kind["source-refusal"].operation == "union"
    assert by_kind["promotion"].target_family == "implicit"
    assert by_kind["persistence"].message.endswith("without mesh truth.")
    assert "no mesh fallback" in by_kind["no-mesh-fallback"].message.lower()
    assert report.canonical_payload()["passed"] is True


def test_surface_csg_operation_plan_accumulates_invalid_operand_diagnostics() -> None:
    plan = plan_surface_csg_operation("union", (object(), object()))

    assert isinstance(plan, SurfaceCSGOperationPlan)
    assert plan.executable is False
    assert plan.operands is None
    assert all(isinstance(diagnostic, SurfaceCSGPlanDiagnostic) for diagnostic in plan.diagnostics)
    assert [diagnostic.code for diagnostic in plan.diagnostics] == ["invalid-operand", "invalid-operand"]
    assert "no mesh fallback" in plan.diagnostics[0].message
    with pytest.raises(BooleanOperationError, match="operand 0.*operand 1"):
        plan.assert_executable()


def test_surface_csg_operation_plan_dispatches_family_pairs_and_refuses_unsupported_registry_entries() -> None:
    left = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    right = make_sphere(radius=0.5, backend="surface")
    operands = prepare_surface_boolean_operands("union", (left, right))
    supported = plan_prepared_surface_csg_operation(operands)

    assert supported.executable is True
    assert all(isinstance(record, SurfaceCSGPairDispatchRecord) for record in supported.pair_dispatch)
    assert {record.support_state for record in supported.pair_dispatch} == {"exact"}

    broken_matrix = dict(SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX)
    broken_matrix[("union", "planar", "revolution")] = SurfaceBooleanFamilyPairSupport(
        operation="union",
        left_family="planar",
        right_family="revolution",
        supported=False,
        phase="intersection-kernel",
        support_state="not-yet-implemented",
        required_future_capability="fixture exact planar/revolution solver",
    )
    registry = assert_surface_csg_solver_registry_complete(build_surface_csg_solver_registry(broken_matrix))
    unsupported = plan_prepared_surface_csg_operation(operands, registry=registry)

    assert unsupported.executable is False
    assert unsupported.diagnostics[0].code == "unsupported-family-pair"
    assert "planar/revolution" in unsupported.diagnostics[0].message
    assert unsupported.canonical_payload()["executable"] is False


def test_surface_csg_negative_diagnostic_fixtures_feed_matrix() -> None:
    invalid_plan = plan_surface_csg_operation("union", (object(), object()))
    left = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    right = make_sphere(radius=0.5, backend="surface")
    operands = prepare_surface_boolean_operands("union", (left, right))
    broken_matrix = dict(SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX)
    broken_matrix[("union", "planar", "revolution")] = SurfaceBooleanFamilyPairSupport(
        operation="union",
        left_family="planar",
        right_family="revolution",
        supported=False,
        phase="intersection-kernel",
        support_state="not-yet-implemented",
        required_future_capability="fixture exact planar/revolution solver",
    )
    unsupported_plan = plan_prepared_surface_csg_operation(
        operands,
        registry=assert_surface_csg_solver_registry_complete(build_surface_csg_solver_registry(broken_matrix)),
    )
    non_executable_graph = build_surface_csg_fragment_graph(invalid_plan)
    cap_plan = SurfaceCSGOperationPlan(
        operation="difference",
        operands=SurfaceBooleanOperands(operation="difference", bodies=(left, right)),
    )
    cap_graph = SurfaceCSGFragmentGraphRecord(
        operation="difference",
        plan=cap_plan,
        classification_edges=(
            SurfaceCSGFragmentClassificationEdgeRecord(
                patch=SurfaceBooleanPatchRef(1, 0),
                relation="inside",
                role="cut_cap",
                cut_curve_ids=("cut-revolution",),
            ),
        ),
    )
    unsupported_caps = build_surface_csg_cap_patches(cap_graph)
    fixtures = (
        _csg_negative_fixture(
            "csg/invalid-operands",
            invalid_plan.diagnostics[0],
            expected_keys=(
                ExpectedDiagnosticKeyRecord(("code",), "invalid-operand"),
                ExpectedDiagnosticKeyRecord(("operation",), "union"),
                ExpectedDiagnosticKeyRecord(("message",)),
            ),
        ),
        _csg_negative_fixture(
            "csg/unsupported-family-pair",
            unsupported_plan.diagnostics[0],
            expected_keys=(
                ExpectedDiagnosticKeyRecord(("code",), "unsupported-family-pair"),
                ExpectedDiagnosticKeyRecord(("operation",), "union"),
                ExpectedDiagnosticKeyRecord(("message",)),
            ),
        ),
        _csg_negative_fixture(
            "csg/non-executable-fragment-graph",
            non_executable_graph.diagnostics[0],
            expected_keys=(
                ExpectedDiagnosticKeyRecord(("code",), "non-executable-plan"),
                ExpectedDiagnosticKeyRecord(("message",)),
            ),
        ),
        _csg_negative_fixture(
            "csg/unsupported-cap-family",
            unsupported_caps.diagnostics[0],
            expected_keys=(
                ExpectedDiagnosticKeyRecord(("code",), "unsupported-cap-family"),
                ExpectedDiagnosticKeyRecord(("cap_family",), "revolution"),
                ExpectedDiagnosticKeyRecord(("message",)),
            ),
        ),
    )

    report = evaluate_negative_diagnostic_fixture_matrix(fixtures, required_domains=("csg",))

    assert report.passed is True
    assert report.domain_coverage[0].fixture_count == 4
    assert all(fixture.expected_snapshot is not None for fixture in fixtures)
    assert "planar/revolution" in unsupported_plan.diagnostics[0].message


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


def test_surface_csg_orients_selected_difference_cut_cap_fragments_without_mutating_source() -> None:
    source_patch = PlanarSurfacePatch(family="planar")
    fragment = SurfaceBooleanTrimmedPatchFragment(
        source_patch=SurfaceBooleanPatchRef(1, 0),
        patch=source_patch,
        cut_curve_ids=("cut-0",),
    )
    selection = select_surface_csg_operation_fragment(
        "difference",
        SurfaceCSGFragmentClassificationRecord(
            patch=SurfaceBooleanPatchRef(1, 0),
            relation="inside",
            sample_uv=(0.5, 0.5),
            sample_point=(0.0, 0.0, 0.0),
            cut_curve_ids=("cut-0",),
        ),
    )

    oriented = orient_surface_csg_selected_fragment("difference", fragment, selection)

    assert isinstance(oriented, SurfaceCSGOrientedFragmentRecord)
    assert oriented.orientation == "reverse"
    assert oriented.included is True
    assert oriented.patch.kernel_metadata()["csg_fragment_orientation"] == "reverse"
    assert source_patch.kernel_metadata().get("csg_fragment_orientation") is None
    assert oriented.to_trimmed_fragment() is not None
    assert oriented.canonical_payload()["orientation"] == "reverse"


def test_surface_csg_orients_selected_fragment_collection_in_patch_order() -> None:
    fragments = (
        SurfaceBooleanTrimmedPatchFragment(
            source_patch=SurfaceBooleanPatchRef(1, 1),
            patch=PlanarSurfacePatch(family="planar"),
        ),
        SurfaceBooleanTrimmedPatchFragment(
            source_patch=SurfaceBooleanPatchRef(0, 0),
            patch=PlanarSurfacePatch(family="planar"),
        ),
    )
    selections = (
        SurfaceCSGOperationSelectionRecord(
            operation="union",
            patch=SurfaceBooleanPatchRef(1, 1),
            relation="outside",
            role="survive",
            cut_cap=SurfaceCSGCutCapRequirementRecord(
                patch=SurfaceBooleanPatchRef(1, 1),
                required=False,
                reason="test",
            ),
        ),
        SurfaceCSGOperationSelectionRecord(
            operation="union",
            patch=SurfaceBooleanPatchRef(0, 0),
            relation="outside",
            role="survive",
            cut_cap=SurfaceCSGCutCapRequirementRecord(
                patch=SurfaceBooleanPatchRef(0, 0),
                required=False,
                reason="test",
            ),
        ),
    )

    oriented = orient_surface_csg_selected_fragments("union", fragments, selections)

    assert tuple(record.source_patch for record in oriented) == (
        SurfaceBooleanPatchRef(0, 0),
        SurfaceBooleanPatchRef(1, 1),
    )
    assert all(record.orientation == "preserve" for record in oriented)


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


def test_surface_csg_continuity_handoff_records_enforceable_c0_g0_only() -> None:
    shell = make_box(size=(1.0, 1.0, 1.0), backend="surface").iter_shells(world=True)[0]
    seam_rebuild = rebuild_surface_csg_shell_seams(shell)

    handoff = record_surface_csg_continuity_handoff(
        seam_rebuild,
        requested_continuity=("C0", "G0", "G1", "C2"),
    )

    assert isinstance(handoff, SurfaceCSGContinuityHandoffRecord)
    assert handoff.supported is False
    assert handoff.enforceable_continuity == ("C0", "G0")
    assert all(isinstance(diagnostic, SurfaceCSGContinuityHandoffDiagnostic) for diagnostic in handoff.diagnostics)
    assert {diagnostic.continuity for diagnostic in handoff.diagnostics} == {"G1", "C2"}
    assert handoff.canonical_payload()["supported"] is False


def test_surface_csg_continuity_handoff_reports_invalid_seam_rebuild() -> None:
    shell = make_box(size=(1.0, 1.0, 1.0), backend="surface").iter_shells(world=True)[0]
    open_shell = replace(shell, seams=shell.seams[:-1])
    seam_rebuild = rebuild_surface_csg_shell_seams(open_shell)

    handoff = record_surface_csg_continuity_handoff(seam_rebuild)

    assert handoff.supported is False
    assert handoff.diagnostics[0].code == "invalid-seam-rebuild"


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


def test_surface_csg_runtime_validity_report_rejects_mesh_results_and_unresolved_diagnostics() -> None:
    operands = prepare_surface_boolean_operands(
        "union",
        [
            make_box(size=(1.0, 1.0, 1.0), backend="surface"),
            make_box(size=(1.0, 1.0, 1.0), center=(2.0, 0.0, 0.0), backend="surface"),
        ],
    )
    mesh = Mesh(
        vertices=np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        faces=np.array(((0, 1, 2),)),
    )

    report = check_surface_csg_runtime_result_validity(
        "union",
        operands,
        mesh,
        unresolved_diagnostics=("ambiguous fragment",),
    )

    assert isinstance(report, SurfaceCSGRuntimeValidityReport)
    assert report.accepted is False
    assert report.status == "invalid"
    assert {diagnostic.code for diagnostic in report.diagnostics} == {
        "mesh-backed-fragment",
        "unresolved-diagnostic",
    }


def test_surface_csg_runtime_validity_report_detects_dangling_trims() -> None:
    operands = prepare_surface_boolean_operands(
        "union",
        [
            make_box(size=(1.0, 1.0, 1.0), backend="surface"),
            make_box(size=(1.0, 1.0, 1.0), center=(2.0, 0.0, 0.0), backend="surface"),
        ],
    )
    dangling_patch = PlanarSurfacePatch(family="planar")
    object.__setattr__(
        dangling_patch,
        "trim_loops",
        (TrimLoop(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)), category="outer"),),
    )
    body = make_surface_body((make_surface_shell((dangling_patch,)),))

    trim_diagnostics = detect_surface_csg_dangling_trims(body)
    report = check_surface_csg_runtime_result_validity("union", operands, body)

    assert trim_diagnostics[0].code == "dangling-trim"
    assert report.accepted is False
    assert any(diagnostic.code == "dangling-trim" for diagnostic in report.diagnostics)
    assert report.canonical_payload()["accepted"] is False


def test_surface_csg_persistence_tessellation_evidence_promotes_clean_reference() -> None:
    body = make_box(size=(1.0, 1.0, 1.0), backend="surface")

    report = verify_surface_csg_persistence_tessellation_evidence(
        body,
        fixture_id="csg/clean-box",
        reference_state="clean",
    )

    assert isinstance(report, SurfaceCSGReferencePromotionReport)
    assert report.promoted is True
    assert isinstance(report.persistence, SurfaceCSGPersistenceEvidenceRecord)
    assert report.persistence.passed is True
    assert report.persistence.loaded_body_id is not None
    assert isinstance(report.tessellation, SurfaceCSGTessellationBoundaryEvidenceRecord)
    assert report.tessellation.passed is True
    assert report.tessellation.face_count > 0
    assert report.diagnostics == ()
    assert report.canonical_payload()["promoted"] is True


def test_surface_csg_persistence_tessellation_evidence_rejects_dirty_reference() -> None:
    body = make_box(size=(1.0, 1.0, 1.0), backend="surface")

    report = verify_surface_csg_persistence_tessellation_evidence(
        body,
        fixture_id="csg/dirty-box",
        reference_state="dirty",
    )

    assert report.promoted is False
    assert report.persistence.passed is True
    assert report.tessellation.passed is True
    assert "dirty" in report.diagnostics[0]


def test_surface_csg_validity_handoff_accepts_assembled_closed_shell_candidate() -> None:
    left = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    right = make_box(size=(1.0, 1.0, 1.0), center=(2.0, 0.0, 0.0), backend="surface")
    operands = prepare_surface_boolean_operands("union", [left, right])
    assembly = SurfaceCSGShellAssemblyRecord(
        operation="union",
        classification="closed",
        shells=(left.iter_shells(world=True)[0],),
    )

    handoff = validate_surface_csg_result_handoff(assembly, operands)

    assert isinstance(handoff, SurfaceCSGValidityHandoffRecord)
    assert handoff.accepted is True
    assert handoff.status == "succeeded"
    assert handoff.body is not None
    assert handoff.validity_gate is not None
    assert handoff.validity_gate.accepted is True
    assert all(isinstance(record, SurfaceCSGSeamRebuildRecord) for record in handoff.seam_rebuilds)
    assert handoff.diagnostics == ()
    assert handoff.canonical_payload()["accepted"] is True


def test_surface_csg_validity_handoff_preserves_empty_success_without_body() -> None:
    left = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    right = make_box(size=(1.0, 1.0, 1.0), center=(2.0, 0.0, 0.0), backend="surface")
    operands = prepare_surface_boolean_operands("intersection", [left, right])
    assembly = SurfaceCSGShellAssemblyRecord(operation="intersection", classification="empty")

    handoff = validate_surface_csg_result_handoff(assembly, operands)

    assert handoff.accepted is True
    assert handoff.classification == "empty"
    assert handoff.body is None
    assert handoff.validity_gate is None


def test_surface_csg_validity_handoff_reports_invalid_reconstructed_shells() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.25, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0))
    operands = prepare_surface_boolean_difference_operands(left, [right])
    graph = build_surface_csg_fragment_graph(plan_surface_csg_operation("difference", (left, right)))
    caps = build_surface_csg_cap_patches(graph)
    boundaries = build_surface_csg_cut_boundary_trims(graph, caps)
    assembly = assemble_surface_csg_result_shells(graph, caps, boundaries)

    handoff = validate_surface_csg_result_handoff(assembly, operands)

    assert handoff.accepted is False
    assert handoff.status == "invalid"
    assert handoff.body is None
    assert handoff.validity_gate is not None
    assert all(isinstance(diagnostic, SurfaceCSGPostReconstructionValidityDiagnostic) for diagnostic in handoff.diagnostics)
    assert {diagnostic.code for diagnostic in handoff.diagnostics} >= {
        "seam-rebuild-failed",
        "validity-gate-rejected",
    }


def test_surface_csg_operand_ordering_normalizer_sorts_commutative_operations_only() -> None:
    left = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    right = make_box(size=(1.0, 1.0, 1.0), center=(2.0, 0.0, 0.0), backend="surface")
    union_operands = prepare_surface_boolean_operands("union", [right, left])
    difference_operands = prepare_surface_boolean_difference_operands(right, [left])

    union_order = normalize_surface_csg_operand_ordering("union", union_operands)
    difference_order = normalize_surface_csg_operand_ordering("difference", difference_operands)

    assert isinstance(union_order, SurfaceCSGOperandOrderingNormalizationRecord)
    assert union_order.normalized_operand_ids == tuple(sorted(union_operands.body_ids))
    assert difference_order.normalized_operand_ids == difference_operands.body_ids
    assert difference_order.normalized_to_original_indices == (0, 1)


def test_surface_csg_result_provenance_map_tracks_fragments_caps_and_boundaries() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.25, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0))
    operands = prepare_surface_boolean_difference_operands(left, [right])
    graph = build_surface_csg_fragment_graph(plan_surface_csg_operation("difference", (left, right)))
    caps = build_surface_csg_cap_patches(graph)
    boundaries = build_surface_csg_cut_boundary_trims(graph, caps)
    assembly = assemble_surface_csg_result_shells(graph, caps, boundaries)

    provenance = build_surface_csg_result_provenance_map(
        assembly,
        operands,
        graph=graph,
        cap_construction=caps,
        cut_boundary=boundaries,
    )

    assert isinstance(provenance, SurfaceCSGResultProvenanceMap)
    assert provenance.supported is True
    assert provenance.diagnostics == ()
    assert all(isinstance(record, SurfaceCSGResultPatchProvenanceRecord) for record in provenance.result_patches)
    assert {record.source_role for record in provenance.result_patches} == {"surviving-fragment", "generated-cap"}
    cap_records = [record for record in provenance.result_patches if record.source_role == "generated-cap"]
    assert cap_records
    assert all(record.cap_payload_index is not None for record in cap_records)
    assert all(record.boundary_attachment_index is not None for record in cap_records)
    assert provenance.canonical_payload()["supported"] is True


def test_surface_csg_result_provenance_map_reports_missing_generated_boundary_attachment() -> None:
    source_patch = SurfaceBooleanPatchRef(1, 0)
    assembly = SurfaceCSGShellAssemblyRecord(
        operation="difference",
        classification="closed",
        shells=(
            make_surface_shell(
                (
                    replace(
                        PlanarSurfacePatch(family="planar"),
                        metadata={"kernel": {"generated_role": "csg_generated_cap"}},
                    ),
                )
            ),
        ),
        provenance=(
            SurfaceCSGFragmentProvenanceRecord(
                source_patch=source_patch,
                result_shell_index=0,
                result_patch_index=0,
                cut_curve_ids=("cut-a",),
            ),
        ),
    )
    operands = prepare_surface_boolean_difference_operands(make_box(backend="surface"), [make_box(backend="surface")])
    caps = SurfaceCSGCapConstructionRecord(
        operation="difference",
        cap_payloads=(
            SurfaceCSGGeneratedCapPatchPayloadRecord(
                source_patch=source_patch,
                cap_family="planar",
                patch=PlanarSurfacePatch(family="planar"),
                cut_curve_ids=("cut-a",),
            ),
        ),
    )

    provenance = build_surface_csg_result_provenance_map(assembly, operands, cap_construction=caps)

    assert provenance.supported is False
    assert all(isinstance(diagnostic, SurfaceCSGProvenanceDiagnostic) for diagnostic in provenance.diagnostics)
    assert provenance.diagnostics[0].code == "missing-boundary-attachment"


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


def test_surface_csg_fragment_graph_preserves_classification_edges_and_cut_provenance() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.25, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0))
    plan = plan_surface_csg_operation("union", (left, right))

    graph = build_surface_csg_fragment_graph(plan)

    assert isinstance(graph, SurfaceCSGFragmentGraphRecord)
    assert graph.supported is True
    assert graph.diagnostics == ()
    assert graph.intersection_stage is not None
    assert graph.intersection_stage.supported is True
    assert all(isinstance(edge, SurfaceCSGFragmentClassificationEdgeRecord) for edge in graph.classification_edges)
    assert len(graph.classification_edges) == len(graph.intersection_stage.patch_classifications)
    assert {edge.role for edge in graph.classification_edges} == {"survive", "discard"}
    assert any(edge.cut_curve_ids for edge in graph.classification_edges)
    assert graph.canonical_payload()["supported"] is True


def test_surface_csg_fragment_graph_refuses_non_executable_plans() -> None:
    plan = plan_surface_csg_operation("union", (object(), object()))

    graph = build_surface_csg_fragment_graph(plan)

    assert graph.supported is False
    assert all(isinstance(diagnostic, SurfaceCSGFragmentGraphDiagnostic) for diagnostic in graph.diagnostics)
    assert graph.diagnostics[0].code == "non-executable-plan"
    assert graph.classification_edges == ()


def test_surface_csg_cap_patch_builder_generates_planar_payloads_from_cut_cap_edges() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.25, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0))
    graph = build_surface_csg_fragment_graph(plan_surface_csg_operation("difference", (left, right)))

    caps = build_surface_csg_cap_patches(graph)

    assert isinstance(caps, SurfaceCSGCapConstructionRecord)
    assert caps.supported is True
    assert caps.diagnostics == ()
    assert caps.cap_payloads
    assert all(isinstance(payload, SurfaceCSGGeneratedCapPatchPayloadRecord) for payload in caps.cap_payloads)
    assert {payload.cap_family for payload in caps.cap_payloads} == {"planar"}
    assert all(isinstance(payload.patch, PlanarSurfacePatch) for payload in caps.cap_payloads)
    assert all("generated-csg-cap" in payload.patch.capability_flags for payload in caps.cap_payloads)
    assert all(payload.patch.kernel_metadata()["generated_role"] == "csg_cap" for payload in caps.cap_payloads)
    assert caps.canonical_payload()["supported"] is True


def test_surface_csg_cap_patch_builder_refuses_non_planar_cap_families() -> None:
    left = make_box(size=(1.0, 1.0, 1.0), backend="surface")
    right = make_sphere(radius=0.5, backend="surface")
    operands = SurfaceBooleanOperands(operation="difference", bodies=(left, right))
    plan = SurfaceCSGOperationPlan(operation="difference", operands=operands)
    graph = SurfaceCSGFragmentGraphRecord(
        operation="difference",
        plan=plan,
        classification_edges=(
            SurfaceCSGFragmentClassificationEdgeRecord(
                patch=SurfaceBooleanPatchRef(1, 0),
                relation="inside",
                role="cut_cap",
                cut_curve_ids=("cut-revolution",),
            ),
        ),
    )

    caps = build_surface_csg_cap_patches(graph)

    assert caps.supported is False
    assert all(isinstance(diagnostic, SurfaceCSGUnsupportedCapDiagnostic) for diagnostic in caps.diagnostics)
    assert caps.diagnostics[0].code == "unsupported-cap-family"
    assert caps.diagnostics[0].cap_family == "revolution"


def test_surface_csg_cut_boundary_trims_attach_generated_cap_payloads() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.25, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0))
    graph = build_surface_csg_fragment_graph(plan_surface_csg_operation("difference", (left, right)))
    caps = build_surface_csg_cap_patches(graph)

    boundaries = build_surface_csg_cut_boundary_trims(graph, caps)

    assert isinstance(boundaries, SurfaceCSGCutBoundaryRecord)
    assert boundaries.supported is True
    assert boundaries.diagnostics == ()
    assert len(boundaries.trim_attachments) == len(caps.cap_payloads)
    assert all(isinstance(attachment, SurfaceCSGTrimAttachmentRecord) for attachment in boundaries.trim_attachments)
    assert {attachment.exposure for attachment in boundaries.trim_attachments} == {"shared"}
    assert all(attachment.trim_loop.category == "outer" for attachment in boundaries.trim_attachments)
    assert all(attachment.cut_curve_ids for attachment in boundaries.trim_attachments)
    assert boundaries.canonical_payload()["supported"] is True


def test_surface_csg_cut_boundary_trims_report_open_generated_cap_boundaries() -> None:
    patch_ref = SurfaceBooleanPatchRef(0, 0)
    payload = SurfaceCSGGeneratedCapPatchPayloadRecord(
        source_patch=patch_ref,
        cap_family="planar",
        patch=PlanarSurfacePatch(family="planar"),
        cut_curve_ids=(),
    )
    plan = SurfaceCSGOperationPlan(
        operation="difference",
        operands=SurfaceBooleanOperands(operation="difference", bodies=(make_box(backend="surface"), make_box(backend="surface"))),
    )
    graph = SurfaceCSGFragmentGraphRecord(operation="difference", plan=plan)
    caps = SurfaceCSGCapConstructionRecord(operation="difference", cap_payloads=(payload,))

    boundaries = build_surface_csg_cut_boundary_trims(graph, caps)

    assert boundaries.supported is False
    assert all(isinstance(diagnostic, SurfaceCSGBoundaryExposureDiagnostic) for diagnostic in boundaries.diagnostics)
    assert boundaries.diagnostics[0].code == "open-boundary"
    assert boundaries.trim_attachments[0].exposure == "open"


def test_surface_csg_result_shell_assembly_builds_body_candidate_from_graph_and_caps() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.25, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0))
    graph = build_surface_csg_fragment_graph(plan_surface_csg_operation("difference", (left, right)))
    caps = build_surface_csg_cap_patches(graph)
    boundaries = build_surface_csg_cut_boundary_trims(graph, caps)

    assembly = assemble_surface_csg_result_shells(graph, caps, boundaries)

    assert isinstance(assembly, SurfaceCSGShellAssemblyRecord)
    assert assembly.supported is True
    assert assembly.diagnostics == ()
    assert assembly.classification == "closed"
    assert len(assembly.shells) == 1
    assert all(isinstance(record, SurfaceCSGShellOrderingRecord) for record in assembly.shell_ordering)
    assert len(assembly.provenance) == assembly.shells[0].patch_count
    assert any(
        patch.kernel_metadata().get("generated_role") == "csg_generated_cap"
        for patch in assembly.shells[0].iter_patches(world=True)
    )
    body = assembly.to_body(metadata={"kernel": {"boolean_operation": "difference"}})
    assert isinstance(body, SurfaceBody)
    assert body.patch_count == assembly.shells[0].patch_count
    assert assembly.canonical_payload()["shell_ordering"][0]["result_shell_index"] == 0


def test_surface_csg_result_shell_assembly_reports_cut_boundary_diagnostics() -> None:
    patch_ref = SurfaceBooleanPatchRef(0, 0)
    payload = SurfaceCSGGeneratedCapPatchPayloadRecord(
        source_patch=patch_ref,
        cap_family="planar",
        patch=PlanarSurfacePatch(family="planar"),
        cut_curve_ids=(),
    )
    plan = SurfaceCSGOperationPlan(
        operation="difference",
        operands=SurfaceBooleanOperands(
            operation="difference",
            bodies=(make_box(backend="surface"), make_box(backend="surface")),
        ),
    )
    graph = SurfaceCSGFragmentGraphRecord(
        operation="difference",
        plan=plan,
        classification_edges=(
            SurfaceCSGFragmentClassificationEdgeRecord(
                patch=patch_ref,
                relation="inside",
                role="cut_cap",
            ),
        ),
    )
    caps = SurfaceCSGCapConstructionRecord(operation="difference", cap_payloads=(payload,))
    boundaries = build_surface_csg_cut_boundary_trims(graph, caps)

    assembly = assemble_surface_csg_result_shells(graph, caps, boundaries)

    assert assembly.supported is False
    assert all(isinstance(diagnostic, SurfaceCSGReconstructionDiagnostic) for diagnostic in assembly.diagnostics)
    assert assembly.diagnostics[0].code == "invalid-cut-boundary"
    with pytest.raises(BooleanOperationError, match="Generated CSG cap payload"):
        assembly.to_body()


def test_surface_csg_result_shell_assembly_can_emit_stably_ordered_multi_shells() -> None:
    left = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (-0.25, 0.0, 0.0))
    right = _translated(make_box(size=(1.0, 1.0, 1.0), backend="surface"), (0.25, 0.0, 0.0))
    graph = build_surface_csg_fragment_graph(plan_surface_csg_operation("union", (left, right)))
    caps = build_surface_csg_cap_patches(graph)
    boundaries = build_surface_csg_cut_boundary_trims(graph, caps)

    assembly = assemble_surface_csg_result_shells(graph, caps, boundaries, multi_shell=True)

    assert assembly.supported is True
    assert len(assembly.shells) > 1
    assert tuple(record.result_shell_index for record in assembly.shell_ordering) == tuple(range(len(assembly.shells)))
    assert tuple(record.patch_count for record in assembly.shell_ordering) == (1,) * len(assembly.shells)


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
