from __future__ import annotations

import numpy as np
import pytest

import impression.modeling as modeling
import impression.modeling.csg as csg_module
from impression.mesh import Mesh
from impression.modeling import (
    LegacyPrimitiveMeshAssumptionClassificationRecord,
    LegacyPrimitiveMeshAssumptionFindingRecord,
    LegacyPrimitiveMeshAssumptionInventoryReport,
    PrimitiveCSGRouteRecord,
    PrimitivePatchProducerSelectionRecord,
    UnsupportedPrimitiveProducerDiagnostic,
    classify_legacy_primitive_mesh_assumption,
    inventory_legacy_primitive_mesh_assumptions,
    make_box,
    make_box_mesh,
    make_cone,
    make_cone_mesh,
    make_cylinder,
    make_cylinder_mesh,
    make_ngon,
    make_ngon_mesh,
    make_nhedron,
    make_nhedron_mesh,
    make_polyhedron,
    make_polyhedron_mesh,
    make_prism,
    make_prism_mesh,
    make_rect,
    make_sphere,
    make_sphere_mesh,
    make_torus,
    make_torus_mesh,
    primitive_csg_route_inventory,
    primitive_patch_producer_selection_inventory,
    scan_legacy_primitive_mesh_assumptions,
    select_primitive_patch_producer,
    unsupported_primitive_producer_diagnostic,
)
from impression.modeling._surface_primitives import (
    make_surface_box,
    make_surface_cone,
    make_surface_cylinder,
    make_surface_ngon,
    make_surface_nhedron,
    make_surface_polyhedron,
    make_surface_prism,
    make_surface_sphere,
    make_surface_torus,
)
from tests.reference_images import (
    ExpectedDiagnosticKeyRecord,
    NegativeDiagnosticFixtureRecord,
    evaluate_negative_diagnostic_fixture_matrix,
    normalize_diagnostic_snapshot,
)


def _mesh_boundary_negative_fixture(
    fixture_id: str,
    operation,
    *,
    expected_keys: tuple[ExpectedDiagnosticKeyRecord, ...],
) -> NegativeDiagnosticFixtureRecord:
    try:
        diagnostic = operation()
    except Exception as exc:  # noqa: BLE001 - negative fixture runner records refusal shape.
        snapshot = normalize_diagnostic_snapshot(
            {
                "code": type(exc).__name__,
                "message": str(exc),
            },
            fixture_id=fixture_id,
        )
    else:
        snapshot = normalize_diagnostic_snapshot(diagnostic, fixture_id=fixture_id)
    return NegativeDiagnosticFixtureRecord(
        fixture_id=fixture_id,
        domain="mesh-boundary",
        expected_keys=expected_keys,
        expected_snapshot=snapshot,
    )


def _seam_continuity_negative_fixture(
    fixture_id: str,
    diagnostic: object,
    *,
    expected_keys: tuple[ExpectedDiagnosticKeyRecord, ...],
) -> NegativeDiagnosticFixtureRecord:
    return NegativeDiagnosticFixtureRecord(
        fixture_id=fixture_id,
        domain="seam-continuity",
        expected_keys=expected_keys,
        expected_snapshot=normalize_diagnostic_snapshot(diagnostic, fixture_id=fixture_id),
    )
from impression.modeling._surface_ops import make_surface_linear_extrude, make_surface_rotate_extrude
from impression.modeling import (
    ADVANCED_PATCH_FAMILIES,
    PATCH_FAMILY_CAPABILITY_MATRIX,
    PATCH_FAMILY_FEATURE_COVERAGE,
    PATCH_FAMILY_PROMOTION_CRITERIA,
    REQUIRED_V1_PATCH_FAMILIES,
    SURFACE_BODY_COMPLETION_TRACKS,
    SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS,
    SURFACE_REFERENCE_ARTIFACT_CLASSES,
    SURFACE_REFERENCE_FIXTURE_CONTRACTS,
    SUPPORTED_SURFACE_PATCH_FAMILIES,
    SURFACE_SPEC_66_RETIREMENT_NOTE,
    AdapterLossiness,
    analysis_tessellation_request,
    BSplineSurfacePatch,
    boolean_union,
    build_higher_order_csg_refusal_diagnostic,
    build_available_family_completion_report,
    build_available_family_missing_evidence_diagnostic,
    build_available_family_no_hidden_mesh_fallback_diagnostic,
    build_dirty_available_family_reference_diagnostic,
    build_surface_boolean_unsupported_family_diagnostic,
    classify_higher_order_csg_pair,
    available_family_csg_classification_rows,
    available_family_producer_path_rows,
    available_family_seam_loft_rows,
    available_family_storage_tessellation_rows,
    compare_tessellation_modes,
    collect_available_family_reference_evidence,
    collect_surface_csg_no_mesh_fallback_evidence,
    collect_sweep_csg_event_seeds,
    DisplacementAuthoringRequest,
    DisplacementDomainMappingRecord,
    DisplacementEvaluationDiagnostic,
    DisplacementLossinessMetadataRecord,
    DisplacementPayloadDiagnostic,
    DisplacementSurfacePatch,
    DisplacementIdentityDiagnostic,
    DisplacementSourcePatchReferenceRecord,
    DisplacementSourceProvenanceRecord,
    DisplacementSourceResolutionResult,
    export_tessellation_request,
    feature_surface_handoff_diagnostic,
    FeatureSurfaceHandoffDiagnostic,
    FeatureSurfaceHandoffRecord,
    flatten_surface_scene,
    FrameTransportPolicyRecord,
    handoff_surface_scene,
    build_heightmap_mask_no_data_diagnostic,
    build_heightmap_import_diagnostic,
    build_displacement_payload_diagnostic,
    estimate_heightmap_normal,
    HeightmapAuthoringRequest,
    HeightmapEvaluationDiagnostic,
    HeightmapImportDiagnostic,
    HeightmapImportRequest,
    HeightmapMaskTessellationRecord,
    HeightmapNoDataDiagnostic,
    HeightmapSampleGridProvenanceRecord,
    heightmap_import_dependency_boundary,
    heightmap_sample_grid_provenance_record,
    heightmap_mask_tessellation_record,
    HeightmapSurfacePatch,
    IMPLICIT_FIELD_NODE_KINDS,
    ImplicitBoundsDiagnostic,
    ImplicitBudgetDiagnostic,
    ImplicitExtractionBudgetRecord,
    ImplicitApproximationMetadata,
    ImplicitFieldAuthoringRequest,
    ImplicitFieldEvaluationDomain,
    ImplicitFieldEvaluationResult,
    ImplicitFieldNode,
    ImplicitFieldProvenanceRecord,
    ImplicitRejectedNodeLocator,
    ImplicitFieldSafetyPolicy,
    ImplicitFieldValidationDiagnostic,
    ImplicitResidualClassificationRecord,
    ImplicitSurfacePatch,
    ImplicitTessellationBoundsDiagnostic,
    ImplicitUnsafeAuthoringDiagnostic,
    make_surface_consumer_collection,
    make_surface_composition,
    make_surface_mesh_adapter,
    make_surface_scene_group,
    make_surface_scene_node,
    make_sweep_csg_evaluator_adapter,
    make_surface_to_mesh_adapter_record,
    mesh_from_surface_body,
    NURBSConicConstructionDiagnostic,
    NURBSConicConstructionRequest,
    NURBSConicProfilePayload,
    NURBSRationalEvaluationMetadata,
    NURBSSurfacePatch,
    NURBSWeightValidationDiagnostic,
    normalize_tessellation_request,
    ParameterDomain,
    Path3D,
    PathFrameDegeneracyDiagnostic,
    PathFrameSampleRecord,
    PlanarSurfacePatch,
    preview_tessellation_request,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    validate_nurbs_weights,
    build_nurbs_circular_arc_control_net,
    build_nurbs_exact_conic_profile_payload,
    build_subdivision_approximation_diagnostic,
    classify_implicit_residual,
    evaluate_implicit_field_gradient,
    implicit_box_field,
    implicit_difference_field,
    implicit_field_provenance_record,
    implicit_sphere_field,
    implicit_union_field,
    evaluate_path_frame,
    interpolate_path_twist_scale,
    make_implicit_extraction_budget,
    SurfaceAdjacencyRecord,
    SurfaceBody,
    SurfaceBoundaryDescriptor,
    SurfaceBoundaryRef,
    SurfaceBodyCompletionEvidenceRecord,
    SurfaceReferenceArtifactClassRecord,
    SurfaceReferenceEvidenceMatrixReport,
    SurfaceReferenceFixtureContractRecord,
    SurfaceReferenceFixtureRequirementRecord,
    SurfaceSweepCSGEvaluatorAdapter,
    SurfaceSweepCSGEventSeedRecord,
    SurfaceSweepCSGFrameEventDiagnostic,
    SurfaceComposition,
    SurfaceCompositionError,
    SurfaceCompositionTraversalRecord,
    SurfaceContinuityMetadata,
    SurfaceContinuityRequest,
    SurfaceContinuitySupportRecord,
    SurfaceContinuityTolerancePolicy,
    SurfaceContinuityConstraintDiagnostic,
    SurfaceBoundaryDerivativeDiagnostic,
    SurfaceBoundaryDerivativeSample,
    SurfaceBoundaryDerivativeSummary,
    SurfaceContinuityResidualMetrics,
    SurfaceObservedContinuityClassRecord,
    SurfaceHigherOrderContinuityValidationReport,
    SurfaceContinuitySeamParameterLocator,
    SurfaceContinuityViolationRecord,
    SurfaceContinuityViolationDiagnostics,
    SurfaceContinuityEnforcementRequest,
    SurfaceContinuityEnforcementRefusalDiagnostic,
    SurfaceContinuityEnforcementResult,
    SurfaceConsumerCollection,
    SurfaceCollectionTessellationResult,
    SurfaceFamilyTessellationAdapter,
    SurfaceFamilyTessellationAdapterCoverageRecord,
    SurfaceFamilyBoundarySupportRecord,
    SURFACE_FAMILY_TESSELLATION_ADAPTERS,
    SUPPORTED_SEAM_CONTINUITY_CLASSES,
    SurfaceBooleanFamilyEligibilityResult,
    SurfaceBooleanFamilyPairSupport,
    SurfaceBooleanOperands,
    SurfaceBooleanSupportState,
    SurfaceBooleanUnsupportedFamilyDiagnostic,
    SurfaceUnsupportedContinuityDiagnostic,
    surface_family_boundary_support_matrix,
    SurfaceCSGAmbiguityDiagnostic,
    SurfaceCSGDegeneracyRecord,
    SurfaceCSGExecutableRowReport,
    SurfaceCSGHigherOrderRefusalDiagnostic,
    SurfaceCSGHigherOrderSupportRecord,
    SurfaceCSGPrimitiveAnalyticPairRecord,
    SurfaceCSGResidualRecord,
    SurfaceCSGRouteRegistryRow,
    SurfaceCSGRouteSupportDiagnostic,
    SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX,
    SURFACE_BOOLEAN_OPERATIONS,
    ANALYTIC_SURFACE_CSG_FAMILIES,
    HIGHER_ORDER_SURFACE_CSG_FAMILIES,
    PARAMETRIC_HIGHER_ORDER_SURFACE_CSG_FAMILIES,
    SAMPLED_SURFACE_CSG_FAMILIES,
    SurfaceSeamParticipationRecord,
    SurfaceSeamValidationResult,
    SubdivisionApproximationDiagnostic,
    SubdivisionAuthoringRequest,
    SubdivisionCageDiagnostic,
    SubdivisionCrease,
    SubdivisionImportDiagnostic,
    SubdivisionImportRequest,
    SubdivisionProducerProvenanceRecord,
    SubdivisionRefinementResult,
    SubdivisionSchemeRecord,
    SubdivisionSurfacePatch,
    SurfaceMeshAdapter,
    SurfaceSceneGroup,
    SurfaceSceneNode,
    SurfaceSeam,
    SurfaceSeamBoundaryUseRef,
    SurfaceSeamContinuityConstraint,
    SurfaceShell,
    SurfaceToMeshAdapterRecord,
    SURFACE_TO_MESH_HELPER_CONTRACT,
    SweepSurfacePatch,
    TessellationBoundaryViolationDiagnostic,
    TessellationRequest,
    TrimLoop,
    assert_implicit_tessellation_sampling_safety,
    assert_subdivision_tessellation_approximation,
    assert_surface_family_tessellation_adapter_coverage,
    assert_surface_reference_requirement_matrix_covers_capabilities,
    audit_all_patch_family_promotion_readiness,
    audit_patch_family_promotion_readiness,
    build_surface_unsupported_continuity_diagnostic,
    classify_surface_seam_continuity,
    evaluate_surface_body_completion_gate,
    evaluate_available_family_reference_evidence_gate,
    evaluate_surface_body_completion_reference_evidence_matrix,
    evaluate_surface_boundary_derivatives,
    compute_surface_continuity_residual_metrics,
    classify_surface_continuity_residuals,
    validate_higher_order_surface_continuity,
    build_surface_continuity_violation_locators,
    build_implicit_bounds_diagnostic,
    build_implicit_budget_diagnostic,
    build_implicit_unsafe_authoring_diagnostic,
    build_subdivision_cage_diagnostic,
    build_subdivision_import_diagnostic,
    format_surface_continuity_violation_diagnostic,
    check_surface_continuity_enforcement_eligibility,
    validate_surface_continuity_enforcement_result,
    evaluate_implicit_field,
    evaluate_implicit_field_domain,
    extract_surface_boundary_descriptor,
    displacement_lossiness_metadata_record,
    inspect_surface_family_tessellation_adapter_coverage,
    load_surface_reference_requirement_matrix,
    make_surface_body_completion_evidence_from_capabilities,
    make_available_family_promoted_reference_evidence,
    make_surface_body,
    make_surface_shell,
    make_implicit_surface,
    make_heightmap_surface_from_grid,
    make_displacement_surface,
    import_heightmap_surface,
    make_subdivision_surface,
    import_subdivision_cage,
    make_implicit_field_node,
    resolve_displacement_source_identity,
    normalize_surface_seam_continuity_constraint,
    prepare_surface_boolean_operands,
    assess_implicit_field_security,
    refine_subdivision_control_cage,
    normalize_subdivision_cage_import_payload,
    subdivision_producer_provenance_record,
    surface_adjacency_from_seams,
    surface_boolean_family_eligibility,
    surface_boolean_family_pair_support,
    classify_surface_csg_route_pair_class,
    classify_higher_order_csg_degeneracies,
    collect_higher_order_csg_residual,
    format_higher_order_csg_route_diagnostics,
    surface_csg_executable_row_report,
    surface_csg_route_lookup,
    surface_boolean_result,
    surface_body_completion_reference_evidence_matrix,
    surface_reference_artifact_classes,
    surface_reference_fixture_contracts,
    snapshot_available_family_completion_report,
    summarize_available_family_missing_evidence,
    summarize_available_family_producer_paths,
    surface_continuity_support,
    surface_csg_analytic_primitive_pair_support,
    surface_csg_completion_support_matrix,
    surface_csg_refusal_record,
    verify_available_family_csg_classification_rows,
    verify_available_family_producer_path_rows,
    verify_available_family_seam_loft_rows,
    verify_available_family_storage_tessellation_rows,
    verify_surface_csg_no_mesh_fallback_evidence,
    surface_composition_to_consumer_collection,
    surface_group,
    validate_feature_surface_handoff,
    validate_implicit_authoring_safety,
    validate_implicit_extraction_budget,
    validate_implicit_field_security,
    validate_surface_seam_continuity_constraint,
    validate_surface_seam_participation,
    validate_tessellation_helper_boundary_input,
    tessellate_surface_body,
    tessellate_surface_composition,
    tessellate_surface_consumer_collection,
    tessellate_surface_patch,
    tessellate_surface_shell,
    traverse_surface_composition,
)


def _make_two_patch_open_shell_body() -> SurfaceBody:
    patch_a = PlanarSurfacePatch(
        family="planar",
        origin=(0.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
    )
    patch_b = PlanarSurfacePatch(
        family="planar",
        origin=(1.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
    )
    seam = SurfaceSeam(
        seam_id="mid",
        boundaries=(
            SurfaceBoundaryRef(0, "right"),
            SurfaceBoundaryRef(1, "left"),
        ),
    )
    shell = make_surface_shell([patch_a, patch_b], seams=(seam,))
    return make_surface_body([shell])


def _make_all_patch_family_body() -> SurfaceBody:
    control_net = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.2], [0.0, 2.0, 0.0]],
            [[1.0, 0.0, 0.1], [1.0, 1.0, 0.4], [1.0, 2.0, 0.1]],
            [[2.0, 0.0, 0.0], [2.0, 1.0, 0.2], [2.0, 2.0, 0.0]],
        ],
        dtype=float,
    )
    knots = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    source = PlanarSurfacePatch(family="planar", metadata={"kernel": {"source": "displacement"}})
    patches = (
        PlanarSurfacePatch(family="planar"),
        RuledSurfacePatch(family="ruled"),
        RevolutionSurfacePatch(family="revolution"),
        BSplineSurfacePatch(family="bspline", degree_u=2, degree_v=2, knots_u=knots, knots_v=knots, control_net=control_net),
        NURBSSurfacePatch(
            family="nurbs",
            degree_u=2,
            degree_v=2,
            knots_u=knots,
            knots_v=knots,
            control_net=control_net,
            weights=np.ones((3, 3), dtype=float),
        ),
        SweepSurfacePatch(
            family="sweep",
            profile_points_uv=[(0.0, 0.0), (0.5, 0.25), (1.0, 0.0)],
            path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.5, 1.0), (0.0, 0.0, 2.0)]),
            frame_policy="fixed",
        ),
        SubdivisionSurfacePatch(
            family="subdivision",
            control_points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.1), (0.0, 1.0, 0.0)],
            faces=((0, 1, 2, 3),),
            subdivision_level=1,
        ),
        ImplicitSurfacePatch(
            family="implicit",
            field=make_implicit_field_node("sphere", parameters={"center": (0.0, 0.0, 0.0), "radius": 0.75}),
            bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        ),
        HeightmapSurfacePatch(
            family="heightmap",
            height_samples=np.array([[0.0, 0.25], [0.5, 0.75]], dtype=float),
            alpha_mask=np.array([[True, True], [False, True]], dtype=bool),
        ),
        DisplacementSurfacePatch(
            family="displacement",
            source_patch=source,
            displacement_samples=np.array([[0.0, 0.1], [0.2, 0.3]], dtype=float),
            alpha_mask=np.array([[True, True], [False, True]], dtype=bool),
            direction="z",
            projection_bounds=(-1.0, 1.0, -1.0, 1.0),
        ),
    )
    return make_surface_body([make_surface_shell(patches)])


def _make_closed_cube_body() -> SurfaceBody:
    patches = (
        PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 0.0), u_axis=(1.0, 0.0, 0.0), v_axis=(0.0, 1.0, 0.0)),  # front
        PlanarSurfacePatch(family="planar", origin=(1.0, 0.0, 0.0), u_axis=(0.0, 0.0, 1.0), v_axis=(0.0, 1.0, 0.0)),  # right
        PlanarSurfacePatch(family="planar", origin=(1.0, 0.0, 1.0), u_axis=(-1.0, 0.0, 0.0), v_axis=(0.0, 1.0, 0.0)),  # back
        PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 1.0), u_axis=(0.0, 0.0, -1.0), v_axis=(0.0, 1.0, 0.0)),  # left
        PlanarSurfacePatch(family="planar", origin=(0.0, 1.0, 0.0), u_axis=(1.0, 0.0, 0.0), v_axis=(0.0, 0.0, 1.0)),  # top
        PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 1.0), u_axis=(1.0, 0.0, 0.0), v_axis=(0.0, 0.0, -1.0)),  # bottom
    )
    seams = (
        SurfaceSeam("front-right", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left"))),
        SurfaceSeam("right-back", (SurfaceBoundaryRef(1, "right"), SurfaceBoundaryRef(2, "left"))),
        SurfaceSeam("back-left", (SurfaceBoundaryRef(2, "right"), SurfaceBoundaryRef(3, "left"))),
        SurfaceSeam("left-front", (SurfaceBoundaryRef(3, "right"), SurfaceBoundaryRef(0, "left"))),
        SurfaceSeam("front-top", (SurfaceBoundaryRef(0, "top"), SurfaceBoundaryRef(4, "bottom"))),
        SurfaceSeam("right-top", (SurfaceBoundaryRef(1, "top"), SurfaceBoundaryRef(4, "right"))),
        SurfaceSeam("back-top", (SurfaceBoundaryRef(2, "top"), SurfaceBoundaryRef(4, "top"))),
        SurfaceSeam("left-top", (SurfaceBoundaryRef(3, "top"), SurfaceBoundaryRef(4, "left"))),
        SurfaceSeam("front-bottom", (SurfaceBoundaryRef(0, "bottom"), SurfaceBoundaryRef(5, "top"))),
        SurfaceSeam("right-bottom", (SurfaceBoundaryRef(1, "bottom"), SurfaceBoundaryRef(5, "right"))),
        SurfaceSeam("back-bottom", (SurfaceBoundaryRef(2, "bottom"), SurfaceBoundaryRef(5, "bottom"))),
        SurfaceSeam("left-bottom", (SurfaceBoundaryRef(3, "bottom"), SurfaceBoundaryRef(5, "left"))),
    )
    shell = make_surface_shell(patches, seams=seams)
    return make_surface_body([shell])


def test_internal_surface_box_builder_is_closed_and_public_box_defaults_surface() -> None:
    surface_box = make_surface_box(size=(2.0, 4.0, 6.0), center=(1.0, 2.0, 3.0))
    public_box = make_box(size=(2.0, 4.0, 6.0), center=(1.0, 2.0, 3.0))

    assert isinstance(surface_box, SurfaceBody)
    assert isinstance(public_box, SurfaceBody)
    assert public_box.stable_identity == surface_box.stable_identity
    assert surface_box.shell_count == 1
    assert surface_box.patch_count == 6
    assert surface_box.bounds_estimate() == (0.0, 2.0, 0.0, 4.0, 0.0, 6.0)

    result = tessellate_surface_body(surface_box, export_tessellation_request())
    assert result.classification == "closed"
    assert result.analysis.is_watertight is True


def test_public_box_supports_surface_backend_and_preserves_explicit_mesh_bridge() -> None:
    surface_box = make_box(size=(2.0, 4.0, 6.0), center=(1.0, 2.0, 3.0), backend="surface", color="blue")
    mesh_box = make_box(size=(2.0, 4.0, 6.0), center=(1.0, 2.0, 3.0), backend="mesh")

    assert isinstance(surface_box, SurfaceBody)
    assert surface_box.consumer_metadata() == {"color": "blue"}
    assert mesh_box.n_faces > 0

    result = tessellate_surface_body(surface_box, export_tessellation_request())
    assert result.classification == "closed"
    assert result.analysis.is_watertight is True


def test_internal_surface_linear_extrude_returns_surface_body() -> None:
    shape = make_rect(size=(2.0, 1.0), center=(0.0, 0.0))

    surface_body = make_surface_linear_extrude(shape, height=3.0)

    assert isinstance(surface_body, SurfaceBody)
    assert surface_body.shell_count == 1
    assert surface_body.patch_count == 6
    assert [patch.family for patch in surface_body.iter_patches(world=False)].count("ruled") == 4
    assert [patch.family for patch in surface_body.iter_patches(world=False)].count("planar") == 2

    result = tessellate_surface_body(surface_body, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_internal_surface_linear_extrude_uses_attached_transform_for_direction_and_center() -> None:
    shape = make_rect(size=(2.0, 1.0), center=(0.0, 0.0))
    body = make_surface_linear_extrude(
        shape,
        height=2.0,
        direction=(0.0, 1.0, 0.0),
        center=(5.0, 6.0, 7.0),
    )

    bounds = body.bounds_estimate()
    assert bounds == (4.0, 6.0, 6.0, 8.0, 6.5, 7.5)


def test_internal_surface_rotate_extrude_returns_closed_surface_body() -> None:
    shape = make_rect(size=(1.0, 2.0), center=(2.0, 0.0))

    surface_body = make_surface_rotate_extrude(shape, angle_deg=360.0, segments=32)

    assert isinstance(surface_body, SurfaceBody)
    assert surface_body.shell_count == 1
    assert surface_body.patch_count == 1
    assert [patch.family for patch in surface_body.iter_patches(world=False)] == ["revolution"]
    assert surface_body.bounds_estimate() == (-2.5, 2.5, -2.5, 2.5, -1.0, 1.0)

    result = tessellate_surface_body(surface_body, export_tessellation_request())
    assert result.classification == "closed"
    assert result.analysis.is_watertight is True
    assert result.analysis.boundary_edges == 0
    assert result.analysis.nonmanifold_edges == 0


def test_internal_surface_rotate_extrude_uses_axis_and_plane_frame() -> None:
    shape = make_rect(size=(1.0, 2.0), center=(2.0, 0.0))
    body = make_surface_rotate_extrude(
        shape,
        angle_deg=360.0,
        axis_origin=(5.0, 6.0, 7.0),
        axis_direction=(1.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        segments=32,
    )

    bounds = body.bounds_estimate()
    assert bounds == (4.0, 6.0, 3.5, 8.5, 4.5, 9.5)


def test_internal_surface_rotate_extrude_partial_sweep_adds_caps_and_stays_open() -> None:
    shape = make_rect(size=(1.0, 2.0), center=(2.0, 0.0))

    body = make_surface_rotate_extrude(shape, angle_deg=180.0, cap_ends=True, segments=24)

    assert isinstance(body, SurfaceBody)
    assert body.shell_count == 1
    assert body.patch_count == 3
    assert [patch.family for patch in body.iter_patches(world=False)].count("revolution") == 1
    assert [patch.family for patch in body.iter_patches(world=False)].count("planar") == 2

    result = tessellate_surface_body(body, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_internal_surface_rotate_extrude_rejects_profiles_that_cross_axis() -> None:
    shape = make_rect(size=(2.0, 1.0), center=(0.0, 0.0))

    with pytest.raises(ValueError, match="crosses the revolution axis"):
        make_surface_rotate_extrude(shape, angle_deg=360.0, segments=24)


def test_internal_surface_cylinder_returns_surface_body_and_public_api_defaults_surface() -> None:
    surface_cylinder = make_surface_cylinder(radius=1.0, height=2.0, resolution=32)
    public_cylinder = make_cylinder(radius=1.0, height=2.0, resolution=32)
    mesh_cylinder = make_cylinder(radius=1.0, height=2.0, resolution=32, backend="mesh")

    assert isinstance(surface_cylinder, SurfaceBody)
    assert isinstance(public_cylinder, SurfaceBody)
    assert public_cylinder.stable_identity == surface_cylinder.stable_identity
    assert surface_cylinder.shell_count == 1
    assert surface_cylinder.patch_count == 6
    assert [patch.family for patch in surface_cylinder.iter_patches(world=False)].count("revolution") == 4
    assert [patch.family for patch in surface_cylinder.iter_patches(world=False)].count("planar") == 2
    assert mesh_cylinder.n_faces > 0

    result = tessellate_surface_body(surface_cylinder, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_internal_surface_cylinder_uses_attached_transform_for_direction_and_center() -> None:
    body = make_surface_cylinder(
        radius=1.0,
        height=2.0,
        center=(5.0, 6.0, 7.0),
        direction=(1.0, 0.0, 0.0),
        resolution=32,
    )

    bounds = body.bounds_estimate()
    assert bounds == (4.0, 6.0, 5.0, 7.0, 6.0, 8.0)


def test_internal_surface_cone_returns_surface_body_and_public_api_defaults_surface() -> None:
    surface_cone = make_surface_cone(bottom_diameter=2.0, top_diameter=0.0, height=2.0, resolution=32)
    public_cone = make_cone(bottom_diameter=2.0, top_diameter=0.0, height=2.0, resolution=32)
    mesh_cone = make_cone(bottom_diameter=2.0, top_diameter=0.0, height=2.0, resolution=32, backend="mesh")

    assert isinstance(surface_cone, SurfaceBody)
    assert isinstance(public_cone, SurfaceBody)
    assert public_cone.stable_identity == surface_cone.stable_identity
    assert surface_cone.shell_count == 1
    assert surface_cone.patch_count == 5
    assert [patch.family for patch in surface_cone.iter_patches(world=False)].count("revolution") == 4
    assert [patch.family for patch in surface_cone.iter_patches(world=False)].count("planar") == 1
    assert mesh_cone.n_faces > 0

    result = tessellate_surface_body(surface_cone, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_internal_surface_cone_uses_attached_transform_for_direction_and_center() -> None:
    body = make_surface_cone(
        bottom_diameter=2.0,
        top_diameter=0.0,
        height=2.0,
        center=(5.0, 6.0, 7.0),
        direction=(0.0, 1.0, 0.0),
        resolution=32,
    )

    bounds = body.bounds_estimate()
    assert bounds == (4.0, 6.0, 5.0, 7.0, 6.0, 8.0)


def test_internal_surface_ngon_returns_surface_body_and_public_api_defaults_surface() -> None:
    surface_ngon = make_surface_ngon(sides=6, radius=1.0, height=2.0)
    public_ngon = make_ngon(sides=6, radius=1.0, height=2.0)
    mesh_ngon = make_ngon(sides=6, radius=1.0, height=2.0, backend="mesh")

    assert isinstance(surface_ngon, SurfaceBody)
    assert isinstance(public_ngon, SurfaceBody)
    assert public_ngon.stable_identity == surface_ngon.stable_identity
    assert surface_ngon.shell_count == 1
    assert surface_ngon.patch_count == 8
    assert [patch.family for patch in surface_ngon.iter_patches(world=False)].count("ruled") == 6
    assert [patch.family for patch in surface_ngon.iter_patches(world=False)].count("planar") == 2
    assert mesh_ngon.n_faces > 0

    result = tessellate_surface_body(surface_ngon, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_internal_surface_prism_returns_surface_body_and_public_api_defaults_surface() -> None:
    surface_prism = make_surface_prism(base_size=(2.0, 1.0), top_size=(1.0, 0.5), height=2.0)
    public_prism = make_prism(base_size=(2.0, 1.0), top_size=(1.0, 0.5), height=2.0)
    mesh_prism = make_prism(base_size=(2.0, 1.0), top_size=(1.0, 0.5), height=2.0, backend="mesh")

    assert isinstance(surface_prism, SurfaceBody)
    assert isinstance(public_prism, SurfaceBody)
    assert public_prism.stable_identity == surface_prism.stable_identity
    assert surface_prism.shell_count == 1
    assert surface_prism.patch_count == 6
    assert [patch.family for patch in surface_prism.iter_patches(world=False)].count("ruled") == 4
    assert [patch.family for patch in surface_prism.iter_patches(world=False)].count("planar") == 2
    assert mesh_prism.n_faces > 0

    result = tessellate_surface_body(surface_prism, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_internal_surface_torus_returns_closed_surface_body_and_public_api_defaults_surface() -> None:
    surface_torus = make_surface_torus(major_radius=2.0, minor_radius=0.5, n_theta=32, n_phi=16)
    public_torus = make_torus(major_radius=2.0, minor_radius=0.5, n_theta=32, n_phi=16)
    mesh_torus = make_torus(major_radius=2.0, minor_radius=0.5, n_theta=32, n_phi=16, backend="mesh")

    assert isinstance(surface_torus, SurfaceBody)
    assert isinstance(public_torus, SurfaceBody)
    assert public_torus.stable_identity == surface_torus.stable_identity
    assert surface_torus.shell_count == 1
    assert surface_torus.patch_count == 1
    assert [patch.family for patch in surface_torus.iter_patches(world=False)] == ["revolution"]
    assert mesh_torus.n_faces > 0

    result = tessellate_surface_body(surface_torus, export_tessellation_request())
    assert result.classification == "closed"
    assert result.analysis.is_watertight is True
    assert result.analysis.boundary_edges == 0
    assert result.analysis.nonmanifold_edges == 0


def test_public_torus_supports_surface_backend_and_preserves_explicit_mesh_bridge() -> None:
    surface_torus = make_torus(
        major_radius=2.0,
        minor_radius=0.5,
        n_theta=32,
        n_phi=16,
        backend="surface",
        color="purple",
    )
    mesh_torus = make_torus(major_radius=2.0, minor_radius=0.5, n_theta=32, n_phi=16, backend="mesh")

    assert isinstance(surface_torus, SurfaceBody)
    assert surface_torus.consumer_metadata() == {"color": "purple"}
    assert mesh_torus.n_faces > 0

    result = tessellate_surface_body(surface_torus, export_tessellation_request())
    assert result.classification == "closed"
    assert result.analysis.is_watertight is True


def test_internal_surface_torus_uses_attached_transform_for_direction_and_center() -> None:
    body = make_surface_torus(
        major_radius=2.0,
        minor_radius=0.5,
        center=(5.0, 6.0, 7.0),
        direction=(1.0, 0.0, 0.0),
        n_theta=32,
        n_phi=16,
    )

    bounds = body.bounds_estimate()
    assert bounds == (4.5, 5.5, 3.5, 8.5, 4.5, 9.5)


def test_internal_surface_sphere_returns_closed_surface_body_and_public_api_defaults_surface() -> None:
    surface_sphere = make_surface_sphere(radius=1.0, center=(0.0, 0.0, 0.0), theta_resolution=32, phi_resolution=16)
    public_sphere = make_sphere(radius=1.0, center=(0.0, 0.0, 0.0), theta_resolution=32, phi_resolution=16)
    mesh_sphere = make_sphere(radius=1.0, center=(0.0, 0.0, 0.0), theta_resolution=32, phi_resolution=16, backend="mesh")

    assert isinstance(surface_sphere, SurfaceBody)
    assert isinstance(public_sphere, SurfaceBody)
    assert public_sphere.stable_identity == surface_sphere.stable_identity
    assert surface_sphere.shell_count == 1
    assert surface_sphere.patch_count == 1
    assert [patch.family for patch in surface_sphere.iter_patches(world=False)] == ["revolution"]
    assert mesh_sphere.n_faces > 0

    result = tessellate_surface_body(surface_sphere, export_tessellation_request())
    assert result.classification == "closed"
    assert result.analysis.is_watertight is True
    assert result.analysis.boundary_edges == 0
    assert result.analysis.nonmanifold_edges == 0
    assert result.analysis.degenerate_faces == 0


def test_internal_surface_sphere_respects_center() -> None:
    body = make_surface_sphere(radius=1.0, center=(5.0, 6.0, 7.0), theta_resolution=32, phi_resolution=16)

    bounds = body.bounds_estimate()
    assert bounds == (4.0, 6.0, 5.0, 7.0, 6.0, 8.0)


def test_internal_surface_polyhedron_returns_surface_body_and_public_api_defaults_surface() -> None:
    surface_polyhedron = make_surface_polyhedron(faces=6, radius=1.0)
    public_polyhedron = make_polyhedron(faces=6, radius=1.0)
    mesh_polyhedron = make_polyhedron(faces=6, radius=1.0, backend="mesh")

    assert isinstance(surface_polyhedron, SurfaceBody)
    assert isinstance(public_polyhedron, SurfaceBody)
    assert public_polyhedron.stable_identity == surface_polyhedron.stable_identity
    assert surface_polyhedron.shell_count == 1
    assert surface_polyhedron.patch_count == 6
    assert [patch.family for patch in surface_polyhedron.iter_patches(world=False)].count("planar") == 6
    assert mesh_polyhedron.n_faces > 0

    result = tessellate_surface_body(surface_polyhedron, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_internal_surface_nhedron_wraps_polyhedron_and_public_api_defaults_surface() -> None:
    surface_nhedron = make_surface_nhedron(faces=8, radius=1.0)
    public_nhedron = make_nhedron(faces=8, radius=1.0)
    mesh_nhedron = make_nhedron(faces=8, radius=1.0, backend="mesh")

    assert isinstance(surface_nhedron, SurfaceBody)
    assert isinstance(public_nhedron, SurfaceBody)
    assert public_nhedron.stable_identity == surface_nhedron.stable_identity
    assert surface_nhedron.shell_count == 1
    assert surface_nhedron.patch_count == 8
    assert [patch.family for patch in surface_nhedron.iter_patches(world=False)].count("planar") == 8
    assert mesh_nhedron.n_faces > 0

    result = tessellate_surface_body(surface_nhedron, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_public_surface_backends_reject_unknown_backend_values() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        make_box(backend="wireframe")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("surface_factory", "mesh_factory", "kwargs"),
    [
        (make_box, make_box_mesh, {"size": (1.0, 2.0, 3.0), "color": "red"}),
        (make_cylinder, make_cylinder_mesh, {"radius": 0.5, "height": 1.0, "resolution": 16}),
        (make_cone, make_cone_mesh, {"bottom_diameter": 1.0, "top_diameter": 0.25, "height": 1.5, "resolution": 16}),
        (make_ngon, make_ngon_mesh, {"sides": 5, "radius": 1.0, "height": 0.5}),
        (make_nhedron, make_nhedron_mesh, {"faces": 6, "radius": 1.0}),
        (make_polyhedron, make_polyhedron_mesh, {"faces": 8, "radius": 1.0}),
        (make_prism, make_prism_mesh, {"base_size": (1.0, 0.5), "top_size": (0.75, 0.25), "height": 1.0}),
        (make_sphere, make_sphere_mesh, {"radius": 0.75, "theta_resolution": 16, "phi_resolution": 8}),
        (make_torus, make_torus_mesh, {"major_radius": 1.0, "minor_radius": 0.2, "n_theta": 16, "n_phi": 8}),
    ],
)
def test_explicit_primitive_mesh_compatibility_apis_return_meshes(surface_factory, mesh_factory, kwargs) -> None:
    surface_result = surface_factory(**kwargs)
    mesh_result = mesh_factory(**kwargs)

    assert isinstance(surface_result, SurfaceBody)
    assert isinstance(mesh_result, Mesh)
    assert mesh_result.n_faces > 0


def test_primitive_csg_route_inventory_covers_surface_defaults_and_mesh_compatibility_names() -> None:
    inventory = primitive_csg_route_inventory()
    payloads = [record.canonical_payload() for record in inventory]

    assert all(isinstance(record, PrimitiveCSGRouteRecord) for record in inventory)
    assert {record.surface_constructor for record in inventory} == {
        "make_box",
        "make_cone",
        "make_cylinder",
        "make_ngon",
        "make_nhedron",
        "make_polyhedron",
        "make_prism",
        "make_sphere",
        "make_torus",
    }
    assert all(payload["csg_gate"] == "assert_no_hidden_surface_csg_mesh_fallback" for payload in payloads)
    assert all(str(record.explicit_mesh_constructor).endswith("_mesh") for record in inventory)


def test_legacy_primitive_mesh_assumption_inventory_classifies_stale_and_accepted_sites(tmp_path) -> None:
    sources = {
        "tests/stale_preview.py": "\n".join(
            [
                "mesh_to_pyvista(make_box(size=(1, 1, 1)))",
                "body = make_cylinder(radius=1.0, height=2.0)",
                "count = body.n_faces",
            ]
        ),
        "tests/accepted_preview.py": "\n".join(
            [
                "mesh = make_box_mesh(size=(1, 1, 1))",
                "preview = mesh_to_pyvista(tessellate_surface_body(make_sphere(radius=1.0)).mesh)",
                "assert make_torus(major_radius=1.0, minor_radius=0.2).patch_count > 0",
            ]
        ),
    }

    report = inventory_legacy_primitive_mesh_assumptions(sources)

    assert isinstance(report, LegacyPrimitiveMeshAssumptionInventoryReport)
    assert report.passed is False
    assert len(report.findings) == 5
    assert len(report.stale_findings) == 2
    assert all(isinstance(finding, LegacyPrimitiveMeshAssumptionFindingRecord) for finding in report.findings)
    assert {finding.primitive_constructor for finding in report.stale_findings} == {"make_box", "make_cylinder"}
    assert {finding.classification.classification for finding in report.findings} == {
        "obsolete mesh-primary test",
        "explicit mesh compatibility consumer",
        "tessellation-boundary consumer",
        "surface-native consumer",
    }
    assert all("tessellate_surface_body" in finding.classification.rewrite_rule for finding in report.stale_findings)
    payload = report.canonical_payload()
    assert payload["stale_count"] == 2
    assert payload["finding_count"] == 5

    fixture = tmp_path / "inventory_fixture.py"
    fixture.write_text("mesh_to_pyvista(make_prism(height=1.0))\n", encoding="utf-8")
    scanned = scan_legacy_primitive_mesh_assumptions(tmp_path)
    assert len(scanned.stale_findings) == 1
    assert scanned.stale_findings[0].path == "inventory_fixture.py"


def test_legacy_primitive_mesh_assumption_classifier_rejects_unknown_primitives() -> None:
    classification = classify_legacy_primitive_mesh_assumption(
        "mesh_from_surface_body(make_box(size=(1, 1, 1)))",
        "make_box",
    )

    assert isinstance(classification, LegacyPrimitiveMeshAssumptionClassificationRecord)
    assert classification.classification == "tessellation-boundary consumer"
    assert classification.stale_assumption is False
    with pytest.raises(ValueError, match="Unsupported primitive constructor"):
        classify_legacy_primitive_mesh_assumption("mesh_to_pyvista(make_widget())", "make_widget")


def test_private_surface_builders_do_not_leak_through_public_modeling_namespace() -> None:
    assert not hasattr(modeling, "make_surface_box")
    assert not hasattr(modeling, "make_surface_ngon")
    assert not hasattr(modeling, "make_surface_cylinder")
    assert not hasattr(modeling, "make_surface_cone")
    assert not hasattr(modeling, "make_surface_prism")
    assert not hasattr(modeling, "make_surface_sphere")
    assert not hasattr(modeling, "make_surface_torus")
    assert not hasattr(modeling, "make_surface_polyhedron")
    assert not hasattr(modeling, "make_surface_nhedron")
    assert not hasattr(modeling, "make_surface_linear_extrude")
    assert not hasattr(modeling, "make_surface_rotate_extrude")

    assert callable(make_surface_box)
    assert callable(make_surface_ngon)
    assert callable(make_surface_cylinder)
    assert callable(make_surface_cone)
    assert callable(make_surface_prism)
    assert callable(make_surface_sphere)
    assert callable(make_surface_torus)
    assert callable(make_surface_polyhedron)
    assert callable(make_surface_nhedron)
    assert callable(make_surface_linear_extrude)
    assert callable(make_surface_rotate_extrude)


def test_patch_family_scope_constants_are_explicit() -> None:
    assert REQUIRED_V1_PATCH_FAMILIES == ("planar", "ruled", "revolution")
    assert "nurbs" in SUPPORTED_SURFACE_PATCH_FAMILIES
    assert PATCH_FAMILY_CAPABILITY_MATRIX["bspline"].support_phase == "available"
    assert PATCH_FAMILY_CAPABILITY_MATRIX["nurbs"].support_phase == "available"
    assert PATCH_FAMILY_CAPABILITY_MATRIX["implicit"].support_phase == "available"
    assert PATCH_FAMILY_CAPABILITY_MATRIX["heightmap"].support_phase == "available"
    assert PATCH_FAMILY_CAPABILITY_MATRIX["displacement"].support_phase == "available"
    assert PATCH_FAMILY_CAPABILITY_MATRIX["planar"].support_phase == "available"
    assert "architecturally deferred" in SURFACE_SPEC_66_RETIREMENT_NOTE
    assert PATCH_FAMILY_FEATURE_COVERAGE["planar"] == (
        "surface-store",
        "caps",
        "planar-primitives",
        "trimmed-faces",
        "tessellation",
        ".impress",
        "diagnostics",
        "no-hidden-fallback",
    )
    assert "no-hidden-fallback" in PATCH_FAMILY_FEATURE_COVERAGE["bspline"]
    assert "no-hidden-fallback" in PATCH_FAMILY_FEATURE_COVERAGE["nurbs"]


def test_surface_body_completion_gate_requires_explicit_non_documentation_evidence() -> None:
    evidence = (
        SurfaceBodyCompletionEvidenceRecord(
            track="patch-family",
            state="verified",
            spec="Surface Spec 242",
            implementation_owner="src/impression/modeling/surface.py",
            evidence_type="unit-test",
            source="documentation",
        ),
    )

    report = evaluate_surface_body_completion_gate(evidence)

    assert report.passed is False
    assert any(diagnostic.code == "documentation-only-evidence" for diagnostic in report.diagnostics)
    assert any(diagnostic.track == "csg" and diagnostic.code == "missing-track-evidence" for diagnostic in report.diagnostics)


def test_surface_body_completion_gate_passes_with_all_required_track_evidence() -> None:
    evidence = tuple(
        SurfaceBodyCompletionEvidenceRecord(
            track=track,
            state="verified",
            spec=f"{track}:fixture",
            implementation_owner="tests/test_surface.py",
            evidence_type="unit-test",
        )
        for track in SURFACE_BODY_COMPLETION_TRACKS
    )

    report = evaluate_surface_body_completion_gate(evidence)

    assert report.passed is True
    assert report.diagnostics == ()


def test_surface_body_completion_capability_evidence_is_implementation_backed() -> None:
    evidence = make_surface_body_completion_evidence_from_capabilities()

    assert evidence
    assert all(record.source == "implementation" for record in evidence)
    assert any(record.track == "patch-family" and record.spec == "patch-family:planar" for record in evidence)


def test_surface_body_completion_reference_evidence_matrix_requires_promoted_artifacts() -> None:
    requirements = surface_body_completion_reference_evidence_matrix()
    evidence = tuple(
        SurfaceBodyCompletionEvidenceRecord(
            track=requirement.track,
            state="verified",
            spec=f"{requirement.track}:{evidence_type}",
            implementation_owner="tests/test_surface.py",
            evidence_type=evidence_type,
        )
        for requirement in requirements
        for evidence_type in requirement.required_evidence_types
    )

    report = evaluate_surface_body_completion_reference_evidence_matrix(evidence)

    assert SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS == requirements
    assert all(isinstance(requirement, SurfaceReferenceFixtureRequirementRecord) for requirement in requirements)
    assert isinstance(report, SurfaceReferenceEvidenceMatrixReport)
    assert report.passed is True
    assert report.diagnostics == ()


def test_surface_reference_requirement_matrix_loads_artifact_classes_and_fixture_contracts() -> None:
    requirements = load_surface_reference_requirement_matrix()
    artifact_classes = surface_reference_artifact_classes()
    fixture_contracts = surface_reference_fixture_contracts()

    assert requirements == SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS
    assert artifact_classes == SURFACE_REFERENCE_ARTIFACT_CLASSES
    assert fixture_contracts == SURFACE_REFERENCE_FIXTURE_CONTRACTS
    assert all(isinstance(record, SurfaceReferenceArtifactClassRecord) for record in artifact_classes)
    assert all(isinstance(record, SurfaceReferenceFixtureContractRecord) for record in fixture_contracts)
    assert {record.artifact_class for record in artifact_classes} >= {
        ".impress-roundtrip",
        "reference-artifact",
        "refusal-diagnostic",
        "tessellation-artifact",
    }
    assert {contract.track for contract in fixture_contracts} == set(SURFACE_BODY_COMPLETION_TRACKS)


def test_surface_reference_requirement_matrix_coverage_assertion_reports_missing_and_duplicate_tracks() -> None:
    requirements = load_surface_reference_requirement_matrix()

    assert assert_surface_reference_requirement_matrix_covers_capabilities(requirements) == requirements

    broken = (
        SurfaceReferenceFixtureRequirementRecord("patch-family", ("unit-test",)),
        SurfaceReferenceFixtureRequirementRecord("patch-family", ("reference-artifact",)),
        SurfaceReferenceFixtureRequirementRecord("unknown-track", ("unit-test",)),
    )

    with pytest.raises(AssertionError, match="missing reference requirement tracks"):
        assert_surface_reference_requirement_matrix_covers_capabilities(broken)
    with pytest.raises(AssertionError, match="duplicate reference requirement tracks"):
        assert_surface_reference_requirement_matrix_covers_capabilities(broken, tracks=("patch-family", "unknown-track"))


def test_surface_body_completion_reference_evidence_matrix_rejects_missing_and_dirty_artifacts() -> None:
    dirty = SurfaceBodyCompletionEvidenceRecord(
        track="primitive",
        state="verified",
        spec="dirty:primitive-reference",
        implementation_owner="project/reference-artifacts/dirty",
        evidence_type="tessellation-artifact",
        source="dirty-artifact",
    )

    report = evaluate_surface_body_completion_reference_evidence_matrix((dirty,))

    assert report.passed is False
    assert any(diagnostic.code == "dirty-artifact-not-promoted" for diagnostic in report.diagnostics)
    assert any(
        diagnostic.code == "missing-reference-evidence" and diagnostic.track == ".impress"
        for diagnostic in report.diagnostics
    )


def test_available_family_producer_path_rows_cover_every_supported_family() -> None:
    rows = available_family_producer_path_rows()
    report = verify_available_family_producer_path_rows(rows)
    grouped = summarize_available_family_producer_paths(rows)

    assert report.passed is True
    assert set(grouped) == set(SUPPORTED_SURFACE_PATCH_FAMILIES)
    assert all(any(row.supported for row in family_rows) for family_rows in grouped.values())

    broken = tuple(row for row in rows if row.family != "heightmap")
    broken_report = verify_available_family_producer_path_rows(broken)
    assert broken_report.passed is False
    assert any(diagnostic.family == "heightmap" and diagnostic.code == "missing-operation-row" for diagnostic in broken_report.diagnostics)

    diagnostic = build_available_family_missing_evidence_diagnostic("implicit", "producer-path", operation="field-node-payload")
    assert diagnostic.family == "implicit"
    assert diagnostic.operation == "field-node-payload"


def test_available_family_storage_and_tessellation_rows_are_registry_backed() -> None:
    rows = available_family_storage_tessellation_rows()
    report = verify_available_family_storage_tessellation_rows(rows)

    assert report.passed is True
    assert {
        (row.family, row.operation)
        for row in rows
        if row.family in SUPPORTED_SURFACE_PATCH_FAMILIES
    } >= {
        (family, operation)
        for family in SUPPORTED_SURFACE_PATCH_FAMILIES
        for operation in (".impress", "tessellation")
    }

    broken = tuple(row for row in rows if not (row.family == "implicit" and row.operation == ".impress"))
    broken_report = verify_available_family_storage_tessellation_rows(broken)
    assert broken_report.passed is False
    assert any(diagnostic.family == "implicit" and diagnostic.operation == ".impress" for diagnostic in broken_report.diagnostics)


def test_available_family_seam_loft_rows_make_non_applicable_loft_explicit() -> None:
    rows = available_family_seam_loft_rows()
    report = verify_available_family_seam_loft_rows(rows)

    assert report.passed is True
    assert any(row.family == "implicit" and row.operation == "loft" and row.supported is False for row in rows)
    assert any(row.family == "planar" and row.operation == "no-hidden-fallback" and row.supported for row in rows)

    broken = tuple(row for row in rows if not (row.family == "heightmap" and row.operation == "no-hidden-fallback"))
    broken_report = verify_available_family_seam_loft_rows(broken)
    assert broken_report.passed is False
    assert any(diagnostic.family == "heightmap" and diagnostic.code == "missing-operation-row" for diagnostic in broken_report.diagnostics)

    diagnostic = build_available_family_no_hidden_mesh_fallback_diagnostic("heightmap", "loft")
    assert diagnostic.code == "missing-no-hidden-mesh-fallback"
    assert "heightmap" in diagnostic.message


def test_available_family_reference_evidence_gate_rejects_missing_and_dirty_artifacts() -> None:
    evidence = tuple(
        SurfaceBodyCompletionEvidenceRecord(
            track=requirement.track,
            state="verified",
            spec=f"{requirement.track}:{evidence_type}",
            implementation_owner="tests/test_surface.py",
            evidence_type=evidence_type,
        )
        for requirement in SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS
        for evidence_type in requirement.required_evidence_types
    )

    summaries = collect_available_family_reference_evidence(evidence)
    report = evaluate_available_family_reference_evidence_gate(evidence)

    assert report.passed is True
    assert all(summary.passed for summary in summaries)

    dirty = SurfaceBodyCompletionEvidenceRecord(
        track="primitive",
        state="verified",
        spec="dirty:primitive-reference",
        implementation_owner="project/reference-artifacts/dirty",
        evidence_type="tessellation-artifact",
        source="dirty-artifact",
    )
    dirty_report = evaluate_available_family_reference_evidence_gate((dirty,))
    dirty_diagnostic = build_dirty_available_family_reference_diagnostic(dirty)

    assert dirty_report.passed is False
    assert dirty_diagnostic.code == "dirty-reference-artifact"
    assert any(diagnostic.code == "dirty-reference-artifact" for diagnostic in dirty_report.diagnostics)


def test_available_family_completion_report_is_deterministic_and_evidence_based() -> None:
    promoted_evidence = make_available_family_promoted_reference_evidence()
    promoted_report = build_available_family_completion_report()
    assert promoted_report.passed is True
    assert {
        (record.track, record.evidence_type)
        for record in promoted_evidence
    } == {
        (requirement.track, evidence_type)
        for requirement in SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS
        for evidence_type in requirement.required_evidence_types
    }

    evidence = tuple(
        SurfaceBodyCompletionEvidenceRecord(
            track=requirement.track,
            state="verified",
            spec=f"{requirement.track}:{evidence_type}",
            implementation_owner="tests/test_surface.py",
            evidence_type=evidence_type,
        )
        for requirement in SURFACE_BODY_REFERENCE_EVIDENCE_REQUIREMENTS
        for evidence_type in requirement.required_evidence_types
    )

    report = build_available_family_completion_report(evidence=evidence)
    snapshot = snapshot_available_family_completion_report(report)

    assert report.passed is True
    assert report.families == tuple(sorted(SUPPORTED_SURFACE_PATCH_FAMILIES))
    assert snapshot.canonical_payload() == snapshot_available_family_completion_report(report).canonical_payload()
    assert summarize_available_family_missing_evidence(report.operation_reports, report.reference_report) == ()

    missing_reference_report = build_available_family_completion_report(evidence=())
    assert missing_reference_report.passed is False
    assert snapshot_available_family_completion_report(missing_reference_report).diagnostic_count > 0


def test_patch_family_promotion_readiness_audits_each_criterion_separately() -> None:
    records = audit_all_patch_family_promotion_readiness()
    by_family = {record.family: record for record in records}

    assert set(by_family) == set(SUPPORTED_SURFACE_PATCH_FAMILIES)
    assert tuple(PATCH_FAMILY_PROMOTION_CRITERIA) == by_family["planar"].supported_criteria + tuple(
        gap.criterion for gap in by_family["planar"].gaps
    )
    assert by_family["planar"].promotable is True
    assert by_family["bspline"].promotable is True
    assert by_family["bspline"].current_phase == "available"
    assert by_family["nurbs"].promotable is True
    assert by_family["nurbs"].current_phase == "available"
    assert by_family["sweep"].promotable is True
    assert by_family["sweep"].current_phase == "available"
    assert by_family["subdivision"].promotable is True
    assert by_family["subdivision"].current_phase == "available"
    assert by_family["implicit"].promotable is True
    assert by_family["implicit"].gaps == ()


def test_patch_family_promotion_readiness_reports_missing_family_record() -> None:
    record = audit_patch_family_promotion_readiness("torus")

    assert record.promotable is False
    assert record.gaps[0].criterion == "record"
    assert "PATCH_FAMILY_CAPABILITY_MATRIX" in record.gaps[0].message


def test_parameter_domain_validates_positive_spans() -> None:
    with pytest.raises(ValueError, match="u_range must have positive span"):
        ParameterDomain((1.0, 1.0), (0.0, 1.0))

    with pytest.raises(ValueError, match="v_range must have positive span"):
        ParameterDomain((0.0, 1.0), (2.0, 2.0))


def test_parameter_domain_contains_with_epsilon() -> None:
    domain = ParameterDomain((0.0, 2.0), (-1.0, 1.0))
    assert domain.contains(0.0, -1.0)
    assert domain.contains(2.0 + 1e-10, 1.0 - 1e-10)
    assert not domain.contains(2.1, 0.0)


def test_trim_loop_normalizes_orientation_and_validates_domain() -> None:
    outer = TrimLoop([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)], category="outer").normalized()
    inner = TrimLoop([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)], category="inner").normalized()

    assert outer.is_clockwise is False
    assert inner.is_clockwise is True

    domain = ParameterDomain((0.0, 1.0), (0.0, 1.0))
    outer.validate_against_domain(domain)
    inner.validate_against_domain(domain)

    with pytest.raises(ValueError, match="outside the patch domain"):
        TrimLoop([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0)], category="outer").validate_against_domain(domain)


def test_planar_patch_evaluates_point_frame_bounds_and_identity() -> None:
    patch = PlanarSurfacePatch(
        family="planar",
        domain=ParameterDomain((0.0, 2.0), (0.0, 3.0)),
        capability_flags={"trimmable", "planar"},
        trim_loops=(
            TrimLoop([(0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)], category="outer"),
            TrimLoop([(0.5, 0.5), (1.5, 0.5), (1.5, 2.5), (0.5, 2.5)], category="inner"),
        ),
        metadata={"kind": "panel"},
        origin=(1.0, 2.0, 3.0),
        u_axis=(2.0, 0.0, 0.0),
        v_axis=(0.0, 3.0, 0.0),
    )

    point = patch.point_at(0.5, 0.25)
    assert np.allclose(point, np.array([2.0, 2.75, 3.0]))

    du, dv = patch.derivatives_at(0.5, 0.25)
    assert np.allclose(du, np.array([2.0, 0.0, 0.0]))
    assert np.allclose(dv, np.array([0.0, 3.0, 0.0]))

    u_axis, v_axis, normal = patch.frame_at(0.5, 0.25)
    assert np.allclose(u_axis, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(v_axis, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(normal, np.array([0.0, 0.0, 1.0]))

    assert patch.outer_trim is not None
    assert len(patch.inner_trims) == 1
    assert patch.capability_flags == frozenset({"trimmable", "planar"})
    assert patch.bounds_estimate() == (1.0, 5.0, 2.0, 11.0, 3.0, 3.0)
    assert isinstance(patch.stable_identity, str)
    assert patch.stable_identity == patch.cache_key


def test_planar_patch_applies_attached_transform() -> None:
    patch = PlanarSurfacePatch(family="planar", origin=(1.0, 0.0, 0.0))
    transform = np.eye(4)
    transform[:3, 3] = [5.0, 0.0, 0.0]
    moved = patch.with_transform(transform)

    assert np.allclose(moved.point_at(0.0, 0.0), np.array([6.0, 0.0, 0.0]))
    assert moved.stable_identity != patch.stable_identity


def test_ruled_patch_evaluates_point_frame_bounds_and_identity() -> None:
    patch = RuledSurfacePatch(
        family="ruled",
        domain=ParameterDomain((0.0, 2.0), (0.0, 3.0)),
        capability_flags={"trimmable", "sidewall"},
        metadata={"kind": "ruled-panel"},
        start_curve=((0.0, 0.0, 0.0), (0.0, 3.0, 0.0)),
        end_curve=((2.0, 0.0, 1.0), (2.0, 3.0, 1.0)),
    )

    point = patch.point_at(1.0, 1.5)
    assert np.allclose(point, np.array([1.0, 1.5, 0.5]))

    du, dv = patch.derivatives_at(1.0, 1.5)
    assert np.allclose(du, np.array([1.0, 0.0, 0.5]))
    assert np.allclose(dv, np.array([0.0, 1.0, 0.0]), atol=1e-3)

    u_axis, v_axis, normal = patch.frame_at(1.0, 1.5)
    assert np.allclose(u_axis, np.array([2.0, 0.0, 1.0]) / np.linalg.norm([2.0, 0.0, 1.0]))
    assert np.allclose(v_axis, np.array([0.0, 1.0, 0.0]), atol=1e-3)
    assert np.allclose(normal, np.array([-1.0, 0.0, 2.0]) / np.linalg.norm([-1.0, 0.0, 2.0]), atol=1e-3)

    assert patch.capability_flags == frozenset({"trimmable", "sidewall"})
    assert patch.bounds_estimate() == (0.0, 2.0, 0.0, 3.0, 0.0, 1.0)
    assert isinstance(patch.stable_identity, str)
    assert patch.stable_identity == patch.cache_key


def test_ruled_patch_rejects_mismatched_curve_shapes() -> None:
    with pytest.raises(ValueError, match="identical shape"):
        RuledSurfacePatch(
            family="ruled",
            start_curve=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            end_curve=((1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 1.0, 0.0)),
        )


def test_revolution_patch_evaluates_point_frame_bounds_and_identity() -> None:
    patch = RevolutionSurfacePatch(
        family="revolution",
        domain=ParameterDomain((0.0, 2.0), (0.0, 4.0)),
        capability_flags={"surface-of-revolution", "trimmable"},
        profile_curve=((1.0, 0.0, 0.0), (1.0, 0.0, 4.0)),
        axis_origin=(0.0, 0.0, 0.0),
        axis_direction=(0.0, 0.0, 1.0),
        start_angle_deg=0.0,
        sweep_angle_deg=180.0,
    )

    point = patch.point_at(1.0, 2.0)
    assert np.allclose(point, np.array([0.0, 1.0, 2.0]), atol=1e-6)

    du, dv = patch.derivatives_at(1.0, 2.0)
    assert np.allclose(du, np.array([-np.pi / 2.0, 0.0, 0.0]), atol=1e-3)
    assert np.allclose(dv, np.array([0.0, 0.0, 1.0]), atol=1e-3)

    u_axis, v_axis, normal = patch.frame_at(1.0, 2.0)
    assert np.allclose(u_axis, np.array([-1.0, 0.0, 0.0]), atol=1e-3)
    assert np.allclose(v_axis, np.array([0.0, 0.0, 1.0]), atol=1e-3)
    assert np.allclose(normal, np.array([0.0, 1.0, 0.0]), atol=1e-3)

    assert patch.capability_flags == frozenset({"surface-of-revolution", "trimmable"})
    assert patch.bounds_estimate() == (-1.0, 1.0, 0.0, 1.0, 0.0, 4.0)
    assert isinstance(patch.stable_identity, str)
    assert patch.stable_identity == patch.cache_key


def test_revolution_patch_rejects_zero_sweep() -> None:
    with pytest.raises(ValueError, match="must be non-zero"):
        RevolutionSurfacePatch(
            family="revolution",
            profile_curve=((1.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
            sweep_angle_deg=0.0,
        )


def test_bspline_surface_patch_owns_knots_control_net_and_domain() -> None:
    patch = BSplineSurfacePatch(
        family="bspline",
        degree_u=2,
        degree_v=1,
        knots_u=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        knots_v=(0.0, 0.0, 1.0, 1.0),
        control_net=[
            [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [(0.5, 0.0, 0.25), (0.5, 1.0, 0.25)],
            [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
        ],
    )

    assert patch.family == "bspline"
    assert patch.degree_u == 2
    assert patch.degree_v == 1
    assert patch.control_net.shape == (3, 2, 3)
    assert patch.domain.u_range == (0.0, 1.0)
    assert patch.geometry_payload()["knots_u"] == (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"family": "nurbs"}, "family must be 'bspline'"),
        ({"family": "bspline", "degree_u": 0}, "degree_u must be >= 1"),
        ({"family": "bspline", "knots_u": (0.0, 1.0, 0.0, 1.0)}, "knots_u must be nondecreasing"),
        (
            {"family": "bspline", "control_net": [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]},
            "at least two control points",
        ),
        (
            {"family": "bspline", "knots_u": (0.0, 0.0, 0.5, 1.0, 1.0)},
            "knot vector length",
        ),
        (
            {
                "family": "bspline",
                "domain": ParameterDomain((0.0, 2.0), (0.0, 1.0)),
            },
            "domain must match",
        ),
    ],
)
def test_bspline_surface_patch_rejects_invalid_record_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BSplineSurfacePatch(**kwargs)


def test_bspline_surface_patch_evaluates_points_derivatives_and_boundaries() -> None:
    patch = BSplineSurfacePatch(
        family="bspline",
        transform_matrix=[
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        control_net=[
            [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [(1.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        ],
    )

    assert np.allclose(patch.point_at(0.0, 0.0), (10.0, 0.0, 0.0))
    assert np.allclose(patch.point_at(1.0, 1.0), (11.0, 1.0, 1.0))
    assert np.allclose(patch.point_at(0.5, 0.5), (10.5, 0.5, 0.25))

    du, dv = patch.derivatives_at(0.5, 0.5)
    assert np.allclose(du, (1.0, 0.0, 0.5))
    assert np.allclose(dv, (0.0, 1.0, 0.5))


def test_bspline_surface_patch_sampling_uses_surface_evaluator() -> None:
    patch = BSplineSurfacePatch(family="bspline")

    samples = patch.sample_grid(2, 2)

    assert samples.shape == (2, 2, 3)
    assert np.allclose(samples[0, 0], (0.0, 0.0, 0.0))
    assert np.allclose(samples[1, 1], (1.0, 1.0, 0.0))


def test_nurbs_surface_patch_evaluates_rational_points_and_derivatives() -> None:
    patch = NURBSSurfacePatch(
        family="nurbs",
        control_net=[
            [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
        ],
        weights=[
            [1.0, 1.0],
            [1.0, 4.0],
        ],
    )

    point = patch.point_at(0.5, 0.5)
    assert np.allclose(point, (5.0 / 7.0, 5.0 / 7.0, 0.0))
    metadata = patch.rational_evaluation_metadata(0.5, 0.5)
    assert isinstance(metadata, NURBSRationalEvaluationMetadata)
    assert metadata.denominator > 0.0
    assert metadata.weight_shape == (2, 2)
    assert metadata.canonical_payload()["point"] == pytest.approx(point.tolist())

    du, dv = patch.derivatives_at(0.5, 0.5)
    epsilon = 1e-6
    finite_du = (patch.point_at(0.5 + epsilon, 0.5) - patch.point_at(0.5 - epsilon, 0.5)) / (2.0 * epsilon)
    finite_dv = (patch.point_at(0.5, 0.5 + epsilon) - patch.point_at(0.5, 0.5 - epsilon)) / (2.0 * epsilon)
    assert np.allclose(du, finite_du, atol=1e-6)
    assert np.allclose(dv, finite_dv, atol=1e-6)


def test_nurbs_surface_patch_with_unit_weights_matches_bspline_patch() -> None:
    control_net = [
        [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [(1.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
    ]
    bspline = BSplineSurfacePatch(family="bspline", control_net=control_net)
    nurbs = NURBSSurfacePatch(family="nurbs", control_net=control_net, weights=np.ones((2, 2)))

    assert np.allclose(nurbs.point_at(0.25, 0.75), bspline.point_at(0.25, 0.75))
    assert np.allclose(nurbs.derivatives_at(0.25, 0.75)[0], bspline.derivatives_at(0.25, 0.75)[0])
    assert np.allclose(nurbs.derivatives_at(0.25, 0.75)[1], bspline.derivatives_at(0.25, 0.75)[1])


def test_nurbs_weight_validator_reports_all_malformed_weight_diagnostics() -> None:
    diagnostics = validate_nurbs_weights(
        [[1.0, float("nan")], [0.0, 1.0]],
        control_net_shape=(2, 2),
    )

    assert all(isinstance(diagnostic, NURBSWeightValidationDiagnostic) for diagnostic in diagnostics)
    assert {diagnostic.code for diagnostic in diagnostics} == {"nonfinite-weight", "nonpositive-weight"}
    assert diagnostics[0].canonical_payload()["shape"] == (2, 2)


def test_nurbs_exact_conic_helpers_build_circle_and_arc_payloads() -> None:
    circle = build_nurbs_exact_conic_profile_payload(
        NURBSConicConstructionRequest(conic_kind="circle", radius=2.0)
    )
    arc = build_nurbs_circular_arc_control_net(radius=1.5, start_angle_deg=0.0, end_angle_deg=90.0)

    assert isinstance(circle, NURBSConicProfilePayload)
    assert circle.supported is True
    assert circle.degree == 2
    assert circle.control_points_uv.shape == (9, 2)
    assert circle.weights.shape == (9,)
    assert circle.knots == (0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0)
    assert circle.metadata["exact_rational_conic"] is True
    assert arc.control_points_uv.shape == (3, 2)
    assert arc.weights[1] == pytest.approx(np.sqrt(0.5))


def test_nurbs_exact_conic_helpers_build_ellipse_and_report_unsupported_requests() -> None:
    ellipse = build_nurbs_exact_conic_profile_payload(
        NURBSConicConstructionRequest(conic_kind="ellipse", radii=(3.0, 1.5), metadata={"profile_id": "outer"})
    )
    unsupported = build_nurbs_exact_conic_profile_payload(NURBSConicConstructionRequest(conic_kind="parabola"))
    invalid = build_nurbs_exact_conic_profile_payload(NURBSConicConstructionRequest(conic_kind="circle", radius=0.0))

    assert ellipse.supported is True
    assert ellipse.metadata["profile_id"] == "outer"
    assert np.max(np.abs(ellipse.control_points_uv[:, 0])) == pytest.approx(3.0)
    assert isinstance(unsupported.diagnostics[0], NURBSConicConstructionDiagnostic)
    assert unsupported.supported is False
    assert unsupported.diagnostics[0].code == "unsupported-conic-kind"
    assert invalid.diagnostics[0].code == "invalid-radius"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"family": "bspline"}, "family must be 'nurbs'"),
        ({"family": "nurbs", "weights": [[1.0, 1.0]]}, "weight-shape-mismatch"),
        ({"family": "nurbs", "weights": [[1.0, 0.0], [1.0, 1.0]]}, "nonpositive-weight"),
        ({"family": "nurbs", "weights": [[1.0, float("nan")], [1.0, 1.0]]}, "nonfinite-weight"),
    ],
)
def test_nurbs_surface_patch_rejects_invalid_weight_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NURBSSurfacePatch(**kwargs)


def test_path_frame_transport_policy_evaluates_twist_scale_and_frame_samples() -> None:
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)])
    policy = FrameTransportPolicyRecord(policy="parallel_transport", twist_degrees=(0.0, 90.0), scale=(1.0, 2.0))

    frame = evaluate_path_frame(path, 0.5, policy)
    twist, scale = interpolate_path_twist_scale(policy, 0.5)

    assert isinstance(frame, PathFrameSampleRecord)
    assert frame.diagnostics == ()
    assert twist == pytest.approx(45.0)
    assert scale == pytest.approx(1.5)
    assert frame.twist_degrees == pytest.approx(45.0)
    assert frame.scale == pytest.approx(1.5)
    assert np.linalg.norm(frame.u_axis) == pytest.approx(1.0)
    assert np.dot(frame.u_axis, frame.w_axis) == pytest.approx(0.0)
    assert frame.canonical_payload()["parameter"] == pytest.approx(0.5)


def test_path_frame_transport_policy_reports_degenerate_tangent() -> None:
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])

    frame = evaluate_path_frame(path, 0.5)

    assert isinstance(frame.diagnostics[0], PathFrameDegeneracyDiagnostic)
    assert frame.diagnostics[0].code == "degenerate-tangent"


def test_sweep_surface_patch_owns_profile_path_and_frame_policy() -> None:
    path = Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)])
    patch = SweepSurfacePatch(
        family="sweep",
        profile_points_uv=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        path=path,
        frame_policy="fixed",
        profile_reference="profile:outer",
        path_reference="path:centerline",
    )

    payload = patch.geometry_payload()
    assert patch.family == "sweep"
    assert patch.frame_policy == "fixed"
    assert patch.profile_reference == "profile:outer"
    assert patch.path_reference == "path:centerline"
    assert np.allclose(payload["path_points"], path.sample())
    assert np.allclose(payload["profile_points_uv"], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"family": "ruled"}, "family must be 'sweep'"),
        ({"family": "sweep", "profile_points_uv": [(0.0, 0.0)]}, "at least two 2D points"),
        ({"family": "sweep", "path": object()}, "path must be a Path3D"),
        ({"family": "sweep", "frame_policy": "magic"}, "frame_policy"),
        ({"family": "sweep", "profile_reference": "  "}, "profile_reference"),
        ({"family": "sweep", "path_reference": "  "}, "path_reference"),
    ],
)
def test_sweep_surface_patch_rejects_invalid_payload_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SweepSurfacePatch(**kwargs)


def test_sweep_surface_patch_evaluates_profile_along_path() -> None:
    patch = SweepSurfacePatch(
        family="sweep",
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]),
        profile_points_uv=[(0.0, 0.0), (1.0, 0.0)],
    )

    assert np.allclose(patch.point_at(0.0, 0.0), (0.0, 0.0, 0.0))
    assert np.allclose(patch.point_at(1.0, 1.0), (1.0, 0.0, 2.0))

    du, dv = patch.derivatives_at(0.5, 0.5)
    assert np.allclose(du, (0.0, 0.0, 2.0), atol=1e-5)
    assert np.allclose(dv, (1.0, 0.0, 0.0), atol=1e-5)


def test_sweep_csg_evaluator_adapter_collects_ordered_event_seeds() -> None:
    patch = SweepSurfacePatch(
        family="sweep",
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]),
        profile_points_uv=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        profile_reference="profile:outer",
        path_reference="path:center",
    )

    seeds = collect_sweep_csg_event_seeds(patch, frame_sample_count=3)
    adapter = make_sweep_csg_evaluator_adapter(patch, frame_sample_count=3)

    assert seeds
    assert all(isinstance(seed, SurfaceSweepCSGEventSeedRecord) for seed in seeds)
    assert tuple(seed.parameter for seed in seeds) == tuple(sorted(seed.parameter for seed in seeds))
    assert isinstance(adapter, SurfaceSweepCSGEvaluatorAdapter)
    assert adapter.supported is True
    assert any(seed.kind == "path-endpoint" for seed in adapter.event_seeds)
    assert any(seed.kind == "profile-vertex" for seed in adapter.event_seeds)
    assert any(seed.kind == "frame-sample" for seed in adapter.event_seeds)
    assert np.allclose(adapter.point_at(1.0, 1.0), patch.point_at(1.0, 1.0))
    assert adapter.canonical_payload()["supported"] is True


def test_sweep_csg_evaluator_adapter_reports_frame_singularity_without_mesh() -> None:
    patch = SweepSurfacePatch(
        family="sweep",
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]),
        profile_points_uv=[(0.0, 0.0), (1.0, 0.0)],
    )

    adapter = make_sweep_csg_evaluator_adapter(patch, frame_sample_count=3)

    assert adapter.supported is False
    assert adapter.diagnostics
    assert all(isinstance(diagnostic, SurfaceSweepCSGFrameEventDiagnostic) for diagnostic in adapter.diagnostics)
    assert any(diagnostic.code == "frame-singularity" for diagnostic in adapter.diagnostics)
    assert all("mesh" not in diagnostic.message.lower() for diagnostic in adapter.diagnostics)


def test_sweep_surface_patch_tessellates_through_surface_boundary() -> None:
    patch = SweepSurfacePatch(
        family="sweep",
        path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
        profile_points_uv=[(0.0, 0.0), (1.0, 0.0)],
    )

    mesh = tessellate_surface_patch(patch)

    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0
    assert mesh.metadata["surface_family"] == "sweep"


def test_subdivision_surface_patch_owns_control_cage_and_creases() -> None:
    patch = SubdivisionSurfacePatch(
        family="subdivision",
        control_points=[
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        faces=((0, 1, 2, 3),),
        creases=(SubdivisionCrease((1, 0), sharpness=2.5),),
        subdivision_level=2,
    )

    payload = patch.geometry_payload()
    assert patch.family == "subdivision"
    assert patch.control_points.shape == (4, 3)
    assert patch.faces == ((0, 1, 2, 3),)
    assert patch.creases[0].edge == (0, 1)
    assert payload["scheme"] == "catmull_clark"
    assert payload["subdivision_level"] == 2
    assert payload["creases"] == [{"edge": (0, 1), "sharpness": 2.5}]
    assert patch.bounds_estimate() == (0.0, 2.0, 0.0, 1.0, 0.0, 0.0)
    assert isinstance(patch.stable_identity, str)


def test_subdivision_scheme_record_and_approximation_diagnostic_are_inspectable() -> None:
    patch = SubdivisionSurfacePatch(
        family="subdivision",
        creases=(SubdivisionCrease((0, 1), sharpness=1.0),),
        subdivision_level=2,
    )

    scheme = patch.scheme_record()
    diagnostic = patch.approximation_diagnostic()
    explicit = build_subdivision_approximation_diagnostic(patch)

    assert isinstance(scheme, SubdivisionSchemeRecord)
    assert scheme.canonical_payload() == {
        "scheme": "catmull_clark",
        "level": 2,
        "crease_count": 1,
        "approximation": "finite_catmull_clark",
    }
    assert isinstance(diagnostic, SubdivisionApproximationDiagnostic)
    assert diagnostic.code == "finite-subdivision-approximation"
    assert "not hidden mesh fallback" in diagnostic.message
    assert explicit.canonical_payload()["level"] == 2


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"family": "sweep"}, "family must be 'subdivision'"),
        ({"family": "subdivision", "control_points": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]}, "at least three"),
        ({"family": "subdivision", "faces": ((0, 1),)}, "at least three vertices"),
        ({"family": "subdivision", "faces": ((0, 1, 1),)}, "may not repeat"),
        ({"family": "subdivision", "faces": ((0, 1, 4),)}, "outside the cage"),
        (
            {"family": "subdivision", "creases": (SubdivisionCrease((0, 2), sharpness=1.0),)},
            "existing cage edges",
        ),
        (
            {
                "family": "subdivision",
                "creases": (SubdivisionCrease((0, 1), sharpness=1.0), SubdivisionCrease((1, 0), sharpness=2.0)),
            },
            "unique per cage edge",
        ),
        ({"family": "subdivision", "subdivision_level": -1}, "subdivision_level"),
        ({"family": "subdivision", "scheme": "loop"}, "scheme"),
    ],
)
def test_subdivision_surface_patch_rejects_invalid_payload_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SubdivisionSurfacePatch(**kwargs)


def test_subdivision_surface_patch_evaluates_finite_approximation() -> None:
    patch = SubdivisionSurfacePatch(family="subdivision")

    point = patch.point_at(0.5, 0.5)
    du, dv = patch.derivatives_at(0.5, 0.5)

    assert np.all(np.isfinite(point))
    assert np.all(np.isfinite(du))
    assert np.all(np.isfinite(dv))


def test_subdivision_refinement_runs_catmull_clark_and_preserves_crease_sharpness() -> None:
    shared_edge_patch = SubdivisionSurfacePatch(
        family="subdivision",
        control_points=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 1.0),
            (2.0, 1.0, 1.0),
        ],
        faces=((0, 1, 2, 3), (1, 4, 5, 2)),
        subdivision_level=1,
    )
    creased_patch = SubdivisionSurfacePatch(
        family="subdivision",
        control_points=shared_edge_patch.control_points,
        faces=shared_edge_patch.faces,
        creases=(SubdivisionCrease((1, 2), sharpness=3.0),),
        subdivision_level=1,
    )

    smooth = refine_subdivision_control_cage(shared_edge_patch)
    creased = creased_patch.refined_cage()

    assert isinstance(smooth, SubdivisionRefinementResult)
    assert smooth.level == 1
    assert smooth.faces and all(len(face) == 4 for face in smooth.faces)
    assert smooth.metadata["approximation"] == "finite_catmull_clark"
    assert np.min(np.linalg.norm(smooth.control_points - np.array([1.0, 0.5, 0.0]), axis=1)) > 0.01
    assert np.min(np.linalg.norm(creased.control_points - np.array([1.0, 0.5, 0.0]), axis=1)) < 1e-12


def test_subdivision_surface_patch_tessellates_with_approximation_metadata() -> None:
    patch = SubdivisionSurfacePatch(family="subdivision", subdivision_level=1)
    original_control_points = patch.control_points.copy()
    original_faces = patch.faces
    original_creases = patch.creases
    request = export_tessellation_request(require_watertight=False)

    mesh = tessellate_surface_patch(patch, request)

    assert mesh.vertices.shape[0] > patch.control_points.shape[0]
    assert mesh.faces.shape[0] > 0
    assert mesh.metadata["surface_family"] == "subdivision"
    assert mesh.metadata["surface_patch_id"] == patch.stable_identity
    assert mesh.metadata["subdivision_scheme"] == "catmull_clark"
    assert mesh.metadata["subdivision_level"] >= 2
    assert mesh.metadata["subdivision_approximation"] == "finite_catmull_clark"
    assert mesh.metadata["tessellation_adapter_boundary"] == "tessellation"
    assert mesh.metadata["tessellation_approximation_boundary"] == "tessellation"
    assert mesh.metadata["adapter_lossiness"] == "lossy"
    assert_subdivision_tessellation_approximation(patch, mesh, request)
    np.testing.assert_allclose(patch.control_points, original_control_points)
    assert patch.faces == original_faces
    assert patch.creases == original_creases


def test_subdivision_surface_public_helper_wraps_valid_cage_as_surface_body() -> None:
    body = make_subdivision_surface(
        control_points=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        faces=((0, 1, 2, 3),),
        creases=({"edge": (0, 1), "sharpness": 2.0},),
        subdivision_level=1,
    )
    patch = body.shells[0].patches[0]

    assert isinstance(body, SurfaceBody)
    assert isinstance(patch, SubdivisionSurfacePatch)
    assert patch.family == "subdivision"
    assert patch.metadata["kernel"]["authoring_boundary"] == "surface-native"
    assert patch.creases[0].sharpness == pytest.approx(2.0)
    provenance = subdivision_producer_provenance_record(patch)
    assert isinstance(provenance, SubdivisionProducerProvenanceRecord)
    assert provenance.family == "subdivision"
    assert provenance.cage_vertex_count == 4
    assert body.metadata["kernel"]["producer_provenance"]["patch_id"] == patch.stable_identity


def test_subdivision_authoring_request_and_cage_diagnostic_are_surface_native() -> None:
    request = SubdivisionAuthoringRequest(
        control_points=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        faces=((0, 1, 2, 3),),
        creases=(SubdivisionCrease((1, 0), sharpness=1.5),),
        metadata={"kernel": {"producer": "unit-test"}},
    )

    body = make_subdivision_surface(request=request)
    diagnostic = build_subdivision_cage_diagnostic(request)
    invalid = build_subdivision_cage_diagnostic(
        {
            "control_points": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
            "faces": ((0, 1, 4),),
        }
    )

    assert request.cage_vertex_count == 4
    assert request.cage_face_count == 1
    assert isinstance(diagnostic, SubdivisionCageDiagnostic)
    assert diagnostic.valid is True
    assert invalid.valid is False
    assert "outside the cage" in invalid.message
    assert isinstance(body.shells[0].patches[0], SubdivisionSurfacePatch)
    assert body.shells[0].patches[0].metadata["kernel"]["producer"] == "unit-test"


def test_subdivision_cage_import_adapter_normalizes_native_payload_without_mesh_fallback() -> None:
    request = SubdivisionImportRequest(
        payload={
            "control_points": [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            "faces": ((0, 1, 2, 3),),
            "creases": ({"edge": (0, 1), "sharpness": 2.0},),
            "subdivision_level": 1,
        },
        source_format="author-cage-json",
        source_id="fixture:cage",
    )

    normalized = normalize_subdivision_cage_import_payload(request)
    diagnostic = build_subdivision_import_diagnostic(request)
    body = import_subdivision_cage(request)
    patch = body.shells[0].patches[0]

    assert normalized.source_format == "author-cage-json"
    assert isinstance(diagnostic, SubdivisionImportDiagnostic)
    assert diagnostic.supported is True
    assert isinstance(patch, SubdivisionSurfacePatch)
    assert patch.creases == (SubdivisionCrease((0, 1), sharpness=2.0),)
    assert patch.metadata["kernel"]["operation"] == "subdivision-cage-import"
    assert patch.metadata["kernel"]["import_source_id"] == "fixture:cage"


def test_subdivision_cage_import_adapter_refuses_mesh_shaped_or_unsupported_payloads() -> None:
    mesh_diagnostic = build_subdivision_import_diagnostic(
        SubdivisionImportRequest(
            payload={"vertices": [(0.0, 0.0, 0.0)], "faces": ((0, 1, 2),)},
            source_format="mesh",
        )
    )
    unsupported_scheme = build_subdivision_import_diagnostic(
        {
            "control_points": [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
            ],
            "faces": ((0, 1, 2),),
            "scheme": "loop",
        }
    )

    assert mesh_diagnostic.supported is False
    assert "no mesh fallback" in mesh_diagnostic.message
    assert unsupported_scheme.supported is False
    assert "scheme" in unsupported_scheme.message
    with pytest.raises(ValueError, match="max_control_points"):
        normalize_subdivision_cage_import_payload(
            SubdivisionImportRequest(
                payload={
                    "control_points": [
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        (1.0, 1.0, 0.0),
                    ],
                    "faces": ((0, 1, 2),),
                },
                max_control_points=2,
            )
        )


def test_subdivision_surface_public_helper_refuses_invalid_cage_or_crease() -> None:
    with pytest.raises(ValueError, match="outside the cage"):
        make_subdivision_surface(
            control_points=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
            ],
            faces=((0, 1, 4),),
        )
    with pytest.raises(ValueError, match="existing cage edges"):
        make_subdivision_surface(
            control_points=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            faces=((0, 1, 2, 3),),
            creases=({"edge": (0, 2), "sharpness": 1.0},),
        )


def test_implicit_field_node_payload_is_allow_listed_and_canonical() -> None:
    sphere = make_implicit_field_node("sphere", parameters={"radius": 2, "center": np.array([0.0, 1.0, 2.0])})
    translated = make_implicit_field_node(
        "translate",
        parameters={"offset": (1.0, 0.0, 0.0)},
        children=(sphere,),
    )

    payload = translated.canonical_payload()
    assert "sphere" in IMPLICIT_FIELD_NODE_KINDS
    assert payload["kind"] == "translate"
    assert payload["parameters"] == {"offset": (1.0, 0.0, 0.0)}
    assert payload["children"][0]["parameters"] == {"center": (0.0, 1.0, 2.0), "radius": 2.0}


def test_implicit_surface_patch_owns_field_tree_and_bounds() -> None:
    patch = ImplicitSurfacePatch(
        family="implicit",
        field={
            "kind": "union",
            "children": [
                {"kind": "sphere", "parameters": {"radius": 1.0}},
                {"kind": "box", "parameters": {"half_extents": (1.0, 2.0, 3.0)}},
            ],
        },
        bounds=(-2.0, 2.0, -3.0, 3.0, -4.0, 4.0),
    )

    payload = patch.geometry_payload()
    assert patch.family == "implicit"
    assert isinstance(patch.field, ImplicitFieldNode)
    assert patch.bounds_estimate() == (-2.0, 2.0, -3.0, 3.0, -4.0, 4.0)
    assert payload["field"]["kind"] == "union"
    assert len(payload["field"]["children"]) == 2
    assert isinstance(patch.stable_identity, str)


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: ImplicitFieldNode(kind="python_eval"), "Unsupported implicit field node kind"),
        (lambda: ImplicitFieldNode(kind="sphere", parameters={"radius": float("nan")}), "must be finite"),
        (lambda: ImplicitFieldNode(kind="sphere", parameters={"callback": object()}), "unsupported type"),
        (lambda: ImplicitSurfacePatch(family="bspline"), "family must be 'implicit'"),
        (lambda: ImplicitSurfacePatch(family="implicit", field={"kind": "sphere", "code": "x"}), "Unsupported implicit field node fields"),
        (lambda: ImplicitSurfacePatch(family="implicit", bounds=(0.0, 0.0, -1.0, 1.0, -1.0, 1.0)), "positive span"),
    ],
)
def test_implicit_surface_patch_rejects_unsupported_payloads(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_implicit_surface_patch_refuses_parametric_extraction_without_exact_surface() -> None:
    patch = ImplicitSurfacePatch(family="implicit")

    with pytest.raises(NotImplementedError, match="no canonical parametric point_at"):
        patch.point_at(0.5, 0.5)

    with pytest.raises(NotImplementedError, match="no canonical parametric derivatives"):
        patch.derivatives_at(0.5, 0.5)


def test_implicit_field_security_accepts_safe_bounded_tree() -> None:
    node = make_implicit_field_node(
        "union",
        children=(
            make_implicit_field_node("sphere", parameters={"radius": 1.0}),
            make_implicit_field_node("box", parameters={"half_extents": (1.0, 1.0, 1.0)}),
        ),
    )

    diagnostic = validate_implicit_field_security(node)

    assert isinstance(diagnostic, ImplicitFieldValidationDiagnostic)
    assert diagnostic.safe is True
    assert diagnostic.node_count == 3
    assert diagnostic.max_depth == 2
    assert diagnostic.canonical_payload()["reason"] == ""


@pytest.mark.parametrize(
    "node, policy, message",
    [
        (
            make_implicit_field_node("sphere", parameters={"code": "x + y"}),
            None,
            "executable shape",
        ),
        (
            make_implicit_field_node("sphere", parameters={"label": "__import__('os')"}),
            None,
            "executable shape",
        ),
        (
            make_implicit_field_node("union", children=tuple(make_implicit_field_node("sphere") for _index in range(3))),
            ImplicitFieldSafetyPolicy(max_children_per_node=2),
            "max_children_per_node",
        ),
        (
            make_implicit_field_node("union", children=(make_implicit_field_node("translate", children=(make_implicit_field_node("sphere"),)),)),
            ImplicitFieldSafetyPolicy(max_depth=2),
            "max_depth",
        ),
        (
            make_implicit_field_node("union", children=(make_implicit_field_node("sphere"), make_implicit_field_node("box"))),
            ImplicitFieldSafetyPolicy(max_nodes=2),
            "max_nodes",
        ),
    ],
)
def test_implicit_field_security_refuses_unsafe_payloads(
    node: ImplicitFieldNode,
    policy: ImplicitFieldSafetyPolicy | None,
    message: str,
) -> None:
    diagnostic = assess_implicit_field_security(node, policy=policy)
    assert diagnostic.safe is False
    assert message in diagnostic.reason

    with pytest.raises(ValueError, match=message):
        validate_implicit_field_security(node, policy=policy)


def test_implicit_unsafe_authoring_diagnostic_reports_rejected_path() -> None:
    node = make_implicit_field_node(
        "union",
        children=(
            make_implicit_field_node("sphere", parameters={"radius": 1.0}),
            make_implicit_field_node("sphere", parameters={"label": "__import__('os')"}),
        ),
    )

    diagnostic = build_implicit_unsafe_authoring_diagnostic(node)

    assert isinstance(diagnostic, ImplicitUnsafeAuthoringDiagnostic)
    assert diagnostic.safe is False
    assert isinstance(diagnostic.locator, ImplicitRejectedNodeLocator)
    assert diagnostic.locator.path == "field.children[1].parameters.label"
    assert diagnostic.locator.node_kind == "sphere"
    with pytest.raises(ValueError, match=r"field\.children\[1\]\.parameters\.label"):
        validate_implicit_authoring_safety(node)


def test_implicit_surface_patch_constructor_runs_security_validator() -> None:
    with pytest.raises(ValueError, match="Unsafe implicit field payload"):
        ImplicitSurfacePatch(family="implicit", field=make_implicit_field_node("sphere", parameters={"eval": "boom"}))


def test_implicit_field_evaluator_handles_primitives_and_combinators() -> None:
    sphere = make_implicit_field_node("sphere", parameters={"radius": 1.0})
    shifted_box = make_implicit_field_node(
        "translate",
        parameters={"offset": (2.0, 0.0, 0.0)},
        children=(make_implicit_field_node("box", parameters={"half_extents": (0.5, 0.5, 0.5)}),),
    )
    union = make_implicit_field_node("union", children=(sphere, shifted_box))
    difference = make_implicit_field_node("difference", children=(sphere, make_implicit_field_node("sphere", parameters={"radius": 0.25})))

    sphere_result = evaluate_implicit_field(sphere, (0.0, 0.0, 0.0))
    union_result = evaluate_implicit_field(union, (2.0, 0.0, 0.0))
    difference_result = evaluate_implicit_field(difference, (0.0, 0.0, 0.0))

    assert isinstance(sphere_result, ImplicitFieldEvaluationResult)
    assert sphere_result.value == pytest.approx(-1.0)
    assert sphere_result.inside is True
    assert union_result.value == pytest.approx(-0.5)
    assert difference_result.value == pytest.approx(0.25)


def test_implicit_field_domain_evaluator_samples_bounded_grid() -> None:
    node = make_implicit_field_node("plane", parameters={"normal": (0.0, 0.0, 1.0), "offset": 0.0})
    domain = ImplicitFieldEvaluationDomain(bounds=(-1.0, 1.0, -1.0, 1.0, -2.0, 2.0), samples=(3, 2, 4))

    values = evaluate_implicit_field_domain(node, domain)

    assert values.shape == (4, 2, 3)
    assert values[0, 0, 0] == pytest.approx(-2.0)
    assert values[-1, -1, -1] == pytest.approx(2.0)


def test_implicit_gradient_budget_and_residual_records_are_inspectable() -> None:
    node = make_implicit_field_node("sphere", parameters={"radius": 1.0})
    budget = make_implicit_extraction_budget(bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), samples=(4, 4, 4))
    gradient = evaluate_implicit_field_gradient(node, (1.0, 0.0, 0.0))
    surface = classify_implicit_residual(5e-7, tolerance=1e-6)
    outside = classify_implicit_residual(0.25, tolerance=1e-6)

    assert isinstance(budget, ImplicitExtractionBudgetRecord)
    assert budget.sample_count == 64
    assert budget.evaluation_domain().samples == (4, 4, 4)
    assert budget.canonical_payload()["sample_count"] == 64
    assert gradient == pytest.approx((1.0, 0.0, 0.0), abs=1e-5)
    assert isinstance(surface, ImplicitResidualClassificationRecord)
    assert surface.classification == "surface"
    assert outside.classification == "outside"

    with pytest.raises(ValueError, match="exceeds max_sample_count"):
        make_implicit_extraction_budget(
            bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            samples=(10, 10, 10),
            max_sample_count=999,
        )


def test_implicit_budget_and_bounds_diagnostics_refuse_before_extraction() -> None:
    valid_bounds = build_implicit_bounds_diagnostic((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
    invalid_bounds = build_implicit_bounds_diagnostic((0.0, 0.0, -1.0, 1.0, -1.0, 1.0))
    valid_budget = build_implicit_budget_diagnostic(
        bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        samples=(4, 4, 4),
    )
    invalid_budget = build_implicit_budget_diagnostic(
        bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        samples=(10, 10, 10),
        max_sample_count=999,
    )

    assert isinstance(valid_bounds, ImplicitBoundsDiagnostic)
    assert valid_bounds.bounded is True
    assert invalid_bounds.bounded is False
    assert invalid_bounds.locator == "bounds"
    assert isinstance(valid_budget, ImplicitBudgetDiagnostic)
    assert valid_budget.executable is True
    assert invalid_budget.executable is False
    assert invalid_budget.locator == "max_sample_count"
    assert validate_implicit_extraction_budget(
        bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        samples=(2, 2, 2),
    ).sample_count == 8
    with pytest.raises(ValueError, match="max_sample_count"):
        validate_implicit_extraction_budget(
            bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            samples=(10, 10, 10),
            max_sample_count=999,
        )


def test_implicit_surface_patch_exposes_field_value_and_domain_helpers() -> None:
    patch = ImplicitSurfacePatch(family="implicit", field=make_implicit_field_node("sphere", parameters={"radius": 2.0}))

    result = patch.field_value_at((0.0, 0.0, 0.0))
    values = patch.evaluate_domain(samples=(2, 2, 2))

    assert result.value == pytest.approx(-2.0)
    assert values.shape == (2, 2, 2)


@pytest.mark.parametrize(
    "node, point, message",
    [
        (make_implicit_field_node("sphere", parameters={"radius": 0.0}), (0.0, 0.0, 0.0), "radius"),
        (make_implicit_field_node("box", parameters={"half_extents": (1.0, 0.0, 1.0)}), (0.0, 0.0, 0.0), "half_extents"),
        (make_implicit_field_node("union"), (0.0, 0.0, 0.0), "at least 1 children"),
        (make_implicit_field_node("translate", children=(make_implicit_field_node("sphere"), make_implicit_field_node("box"))), (0.0, 0.0, 0.0), "at most 1 children"),
        (make_implicit_field_node("scale", parameters={"factor": 0.0}, children=(make_implicit_field_node("sphere"),)), (0.0, 0.0, 0.0), "factor"),
    ],
)
def test_implicit_field_evaluator_refuses_invalid_runtime_payloads(node: ImplicitFieldNode, point: tuple[float, float, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_implicit_field(node, point)


def test_implicit_surface_patch_tessellates_with_bounds_and_approximation_metadata() -> None:
    patch = ImplicitSurfacePatch(family="implicit", field=make_implicit_field_node("sphere", parameters={"radius": 0.75}))
    safety = assert_implicit_tessellation_sampling_safety(patch, preview_tessellation_request())

    mesh = tessellate_surface_patch(patch, preview_tessellation_request())

    diagnostic = mesh.metadata["implicit_bounds_diagnostic"]
    approximation = mesh.metadata["implicit_approximation"]
    assert safety.within_bounds is True
    assert safety.canonical_payload() == diagnostic
    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0
    assert mesh.metadata["surface_family"] == "implicit"
    assert mesh.metadata["surface_patch_id"] == patch.stable_identity
    assert diagnostic["within_bounds"] is True
    assert diagnostic["estimated_cells"] <= diagnostic["max_cells"]
    assert approximation["method"] == "bounded_sampled_sign_change_quads"
    assert approximation["exact"] is False
    assert approximation["active_cells"] > 0
    assert mesh.metadata["implicit_approximation_boundary"] == "tessellation"
    assert mesh.metadata["tessellation_adapter_boundary"] == "tessellation"
    assert mesh.metadata["tessellation_approximation_boundary"] == "tessellation"
    assert mesh.metadata["adapter_lossiness"] == "lossy"


def test_implicit_surface_public_helper_wraps_safe_bounded_field() -> None:
    body = make_implicit_surface(
        field=make_implicit_field_node("sphere", parameters={"radius": 0.75}),
        bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    )
    patch = body.shells[0].patches[0]

    assert isinstance(body, SurfaceBody)
    assert isinstance(patch, ImplicitSurfacePatch)
    assert patch.family == "implicit"
    assert patch.metadata["kernel"]["authoring_boundary"] == "surface-native"
    assert patch.bounds == (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    provenance = implicit_field_provenance_record(patch)
    assert isinstance(provenance, ImplicitFieldProvenanceRecord)
    assert provenance.node_count == 1
    assert body.metadata["kernel"]["producer_provenance"]["patch_id"] == patch.stable_identity


def test_implicit_field_authoring_request_and_named_helpers_build_safe_graphs() -> None:
    field = implicit_difference_field(
        implicit_union_field(
            (
                implicit_sphere_field(radius=1.0),
                implicit_box_field(center=(0.25, 0.0, 0.0), half_extents=(0.5, 0.5, 0.5)),
            )
        ),
        cutters=(implicit_sphere_field(radius=0.25),),
    )
    request = ImplicitFieldAuthoringRequest(
        field=field,
        bounds=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5),
        metadata={"kernel": {"producer": "unit-test"}},
    )

    body = make_implicit_surface(request=request)
    patch = body.shells[0].patches[0]
    value = evaluate_implicit_field(patch.field, (2.0, 0.0, 0.0))

    assert isinstance(patch, ImplicitSurfacePatch)
    assert patch.metadata["kernel"]["producer"] == "unit-test"
    assert patch.metadata["kernel"]["operation"] == "implicit-authoring"
    assert request.canonical_payload()["field"]["kind"] == "difference"
    assert value.value > 0.0


def test_implicit_surface_public_helper_refuses_unsafe_or_malformed_fields() -> None:
    with pytest.raises(ValueError, match="Unsafe implicit field payload"):
        make_implicit_surface(
            field=make_implicit_field_node("sphere", parameters={"eval": "boom"}),
            bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        )
    with pytest.raises(ValueError, match="positive span"):
        make_implicit_surface(
            field=make_implicit_field_node("sphere", parameters={"radius": 0.75}),
            bounds=(0.0, 0.0, -1.0, 1.0, -1.0, 1.0),
        )


def test_implicit_tessellation_refuses_unbounded_sampling_request() -> None:
    patch = ImplicitSurfacePatch(
        family="implicit",
        field=make_implicit_field_node("sphere", parameters={"radius": 1.0}),
        bounds=(-100.0, 100.0, -100.0, 100.0, -100.0, 100.0),
    )

    with pytest.raises(ValueError, match="bounded sampling limit"):
        assert_implicit_tessellation_sampling_safety(patch, analysis_tessellation_request())
    with pytest.raises(ValueError, match="bounded sampling limit"):
        tessellate_surface_patch(patch, analysis_tessellation_request())


def test_implicit_tessellation_metadata_records_are_canonical() -> None:
    diagnostic = ImplicitTessellationBoundsDiagnostic(
        bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        samples=(5, 5, 5),
        estimated_cells=64,
    )
    metadata = ImplicitApproximationMetadata(
        method="bounded_sampled_sign_change_quads",
        isovalue=0.0,
        samples=(5, 5, 5),
        active_cells=8,
    )

    assert diagnostic.within_bounds is True
    assert diagnostic.canonical_payload()["samples"] == (5, 5, 5)
    assert metadata.canonical_payload()["exact"] is False


def test_surface_family_tessellation_adapter_registry_covers_supported_families() -> None:
    assert set(SURFACE_FAMILY_TESSELLATION_ADAPTERS) == set(SUPPORTED_SURFACE_PATCH_FAMILIES)
    for family, adapter in SURFACE_FAMILY_TESSELLATION_ADAPTERS.items():
        assert isinstance(adapter, SurfaceFamilyTessellationAdapter)
        assert adapter.family == family
        assert adapter.canonical_payload()["family"] == family


def test_surface_family_tessellation_adapter_coverage_inventory_is_traceable() -> None:
    records = assert_surface_family_tessellation_adapter_coverage()

    assert all(isinstance(record, SurfaceFamilyTessellationAdapterCoverageRecord) for record in records)
    assert {record.family for record in records} == set(SUPPORTED_SURFACE_PATCH_FAMILIES)
    assert all(record.covered for record in records)
    assert all(record.metadata_traceable for record in records)
    assert {record.family for record in records if record.required_for_available} == {
        family for family, capability in PATCH_FAMILY_CAPABILITY_MATRIX.items() if capability.support_phase == "available"
    }
    assert inspect_surface_family_tessellation_adapter_coverage() == records


def test_every_surface_family_tessellates_through_family_adapter_metadata() -> None:
    patches = {
        "planar": PlanarSurfacePatch(family="planar"),
        "ruled": RuledSurfacePatch(family="ruled"),
        "revolution": RevolutionSurfacePatch(
            family="revolution",
            profile_curve=((1.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
        ),
        "bspline": BSplineSurfacePatch(family="bspline"),
        "nurbs": NURBSSurfacePatch(family="nurbs"),
        "sweep": SweepSurfacePatch(
            family="sweep",
            path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
            profile_points_uv=[(0.0, 0.0), (1.0, 0.0)],
        ),
        "subdivision": SubdivisionSurfacePatch(family="subdivision", subdivision_level=1),
        "implicit": ImplicitSurfacePatch(
            family="implicit",
            field=make_implicit_field_node("sphere", parameters={"radius": 0.75}),
        ),
        "heightmap": HeightmapSurfacePatch(
            family="heightmap",
            height_samples=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float),
        ),
        "displacement": DisplacementSurfacePatch(
            family="displacement",
            source_patch=PlanarSurfacePatch(family="planar"),
            displacement_samples=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float),
            direction="z",
            projection_bounds=(0.0, 1.0, 0.0, 1.0),
        ),
    }

    for family in SUPPORTED_SURFACE_PATCH_FAMILIES:
        mesh = tessellate_surface_patch(patches[family], preview_tessellation_request())
        adapter_metadata = mesh.metadata["tessellation_family_adapter"]
        assert mesh.vertices.shape[0] > 0
        assert mesh.faces.shape[0] > 0
        assert mesh.metadata["surface_family"] == family
        assert adapter_metadata["family"] == family
        assert mesh.metadata["tessellation_adapter_boundary"] == "tessellation"


def test_sampled_surface_tessellation_adapters_preserve_payload_identity_and_lossiness_metadata() -> None:
    source = PlanarSurfacePatch(family="planar", metadata={"kernel": {"source": "sampled-displacement"}})
    patches = (
        HeightmapSurfacePatch(
            family="heightmap",
            height_samples=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float),
            alpha_mask=np.asarray([[True, True], [False, True]], dtype=bool),
        ),
        DisplacementSurfacePatch(
            family="displacement",
            source_patch=source,
            displacement_samples=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float),
            alpha_mask=np.asarray([[True, True], [False, True]], dtype=bool),
            direction="z",
            projection_bounds=(0.0, 1.0, 0.0, 1.0),
        ),
    )
    original_identities = [patch.stable_identity for patch in patches]

    meshes = [tessellate_surface_patch(patch, preview_tessellation_request()) for patch in patches]

    assert [patch.stable_identity for patch in patches] == original_identities
    for patch, mesh in zip(patches, meshes):
        assert mesh.vertices.shape[0] > 0
        assert mesh.metadata["surface_family"] == patch.family
        assert mesh.metadata["surface_patch_id"] == patch.stable_identity
        assert mesh.metadata["adapter_lossiness"] == "lossy"
        assert mesh.metadata["tessellation_adapter_boundary"] == "tessellation"
    assert meshes[1].metadata["displacement_source_family"] == "planar"
    assert meshes[1].metadata["displacement_source_patch_id"] == source.stable_identity


def test_displacement_source_identity_policy_records_embedded_source_payload() -> None:
    source = PlanarSurfacePatch(family="planar", metadata={"kernel": {"source": "displacement-base"}})
    patch = DisplacementSurfacePatch(
        family="displacement",
        source_patch=source,
        displacement_samples=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float),
        projection_bounds=(0.0, 1.0, 0.0, 1.0),
    )

    reference = patch.source_reference_record()
    diagnostic = patch.source_identity_diagnostic()

    assert isinstance(reference, DisplacementSourcePatchReferenceRecord)
    assert reference.source_family == "planar"
    assert reference.source_patch_id == source.stable_identity
    assert reference.embedded is True
    assert isinstance(diagnostic, DisplacementIdentityDiagnostic)
    assert diagnostic.code == "embedded-source-payload"
    assert "mesh" in diagnostic.message


def test_displacement_source_identity_resolver_reports_embedded_in_body_and_missing_sources() -> None:
    source = PlanarSurfacePatch(family="planar", metadata={"kernel": {"source": "displacement-base"}})
    embedded = resolve_displacement_source_identity(source_patch=source)
    in_body = resolve_displacement_source_identity(source_patch_id=source.stable_identity, candidate_patches=(source,))
    missing = resolve_displacement_source_identity(source_patch_id="external:patch", candidate_patches=(source,))

    assert isinstance(embedded, DisplacementSourceResolutionResult)
    assert embedded.resolved is True
    assert isinstance(embedded.provenance, DisplacementSourceProvenanceRecord)
    assert embedded.provenance.relationship == "embedded"
    assert in_body.resolved is True
    assert in_body.provenance is not None
    assert in_body.provenance.relationship == "in-body"
    assert missing.resolved is False
    assert missing.diagnostic.code == "external-source-refused"
    assert "cross-body" in missing.diagnostic.message


def test_displacement_payload_authoring_builder_creates_sampled_surface_body() -> None:
    source = PlanarSurfacePatch(family="planar", domain=ParameterDomain((0.0, 1.0), (0.0, 1.0)))
    request = DisplacementAuthoringRequest(
        source_patch=source,
        displacement_samples=np.asarray([[0.0, 0.1], [0.2, 0.3]], dtype=float),
        direction="z",
        projection_bounds=(0.0, 1.0, 0.0, 1.0),
        metadata={"kernel": {"producer": "unit-test"}},
    )

    diagnostic = build_displacement_payload_diagnostic(request)
    body = make_displacement_surface(request)
    patch = body.shells[0].patches[0]

    assert isinstance(diagnostic, DisplacementPayloadDiagnostic)
    assert diagnostic.valid is True
    assert isinstance(patch, DisplacementSurfacePatch)
    assert isinstance(patch.metadata["kernel"]["lossiness"], dict)
    assert patch.metadata["kernel"]["producer"] == "unit-test"
    assert patch.metadata["kernel"]["source_resolution"]["resolved"] is True
    assert isinstance(displacement_lossiness_metadata_record(), DisplacementLossinessMetadataRecord)


def test_displacement_payload_authoring_builder_refuses_callable_and_invalid_payloads() -> None:
    source = PlanarSurfacePatch(family="planar", domain=ParameterDomain((0.0, 1.0), (0.0, 1.0)))
    callable_diagnostic = build_displacement_payload_diagnostic(
        DisplacementAuthoringRequest(
            source_patch=source,
            displacement_samples=lambda _u, _v: 0.0,  # type: ignore[arg-type]
            projection_bounds=(0.0, 1.0, 0.0, 1.0),
        )
    )
    invalid_diagnostic = build_displacement_payload_diagnostic(
        DisplacementAuthoringRequest(
            source_patch=source,
            displacement_samples=np.asarray([[0.0, np.nan], [0.2, 0.3]], dtype=float),
            projection_bounds=(0.0, 1.0, 0.0, 1.0),
        )
    )

    assert callable_diagnostic.valid is False
    assert "Callable displacement" in callable_diagnostic.message
    assert invalid_diagnostic.valid is False
    assert "finite" in invalid_diagnostic.message
    with pytest.raises(ValueError, match="Callable displacement"):
        make_displacement_surface(
            DisplacementAuthoringRequest(
                source_patch=source,
                displacement_samples=lambda _u, _v: 0.0,  # type: ignore[arg-type]
                projection_bounds=(0.0, 1.0, 0.0, 1.0),
            )
        )


def test_displacement_domain_mapping_and_evaluation_diagnostic_are_inspectable() -> None:
    source = PlanarSurfacePatch(family="planar", domain=ParameterDomain((0.0, 2.0), (-1.0, 1.0)))
    patch = DisplacementSurfacePatch(
        family="displacement",
        source_patch=source,
        displacement_samples=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float),
        projection_bounds=(0.0, 2.0, -1.0, 1.0),
    )

    mapping = patch.domain_mapping_record()
    diagnostic = patch.evaluation_diagnostic()
    point = patch.point_at(1.0, 0.0)
    du, dv = patch.derivatives_at(1.0, 0.0)

    assert isinstance(mapping, DisplacementDomainMappingRecord)
    assert mapping.source_domain == ((0.0, 2.0), (-1.0, 1.0))
    assert mapping.displacement_domain == mapping.source_domain
    assert mapping.projection_bounds == (0.0, 2.0, -1.0, 1.0)
    assert isinstance(diagnostic, DisplacementEvaluationDiagnostic)
    assert diagnostic.code == "finite-difference-displacement-derivatives"
    assert np.all(np.isfinite(point))
    assert np.all(np.isfinite(du))
    assert np.all(np.isfinite(dv))


def test_heightmap_evaluation_normal_and_mask_tessellation_records_are_inspectable() -> None:
    patch = HeightmapSurfacePatch(
        family="heightmap",
        height_samples=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float),
        alpha_mask=np.asarray([[True, True], [False, True]], dtype=bool),
        alpha_mode="mask",
    )

    normal, diagnostic = estimate_heightmap_normal(patch, 0.5, 0.5)
    record = heightmap_mask_tessellation_record(patch)
    mesh = tessellate_surface_patch(patch, preview_tessellation_request())

    assert isinstance(diagnostic, HeightmapEvaluationDiagnostic)
    assert diagnostic.code == "heightmap-normal-estimated"
    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert isinstance(record, HeightmapMaskTessellationRecord)
    assert record.cell_count == 1
    assert record.skipped_cell_count == 1
    assert record.emitted_face_count == 0
    assert record.canonical_payload()["alpha_mode"] == "mask"
    assert mesh.faces.shape[0] == record.emitted_face_count


def test_heightmap_native_finite_grid_builder_records_provenance_and_mask_diagnostics() -> None:
    request = HeightmapAuthoringRequest(
        height_samples=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float),
        alpha_mask=np.asarray([[True, True], [False, True]], dtype=bool),
        alpha_mode="mask",
        xy_scale=(0.5, 0.25),
        center=(1.0, 2.0, 3.0),
        height_scale=2.0,
        metadata={"kernel": {"producer": "unit-test"}},
    )

    diagnostic = build_heightmap_mask_no_data_diagnostic(request)
    provenance = heightmap_sample_grid_provenance_record(request)
    body = make_heightmap_surface_from_grid(request)
    patch = body.shells[0].patches[0]

    assert isinstance(diagnostic, HeightmapNoDataDiagnostic)
    assert diagnostic.valid is True
    assert diagnostic.masked_sample_count == 1
    assert isinstance(provenance, HeightmapSampleGridProvenanceRecord)
    assert provenance.sample_shape == (2, 2)
    assert isinstance(patch, HeightmapSurfacePatch)
    assert patch.metadata["kernel"]["producer"] == "unit-test"
    assert patch.metadata["kernel"]["producer_provenance"]["operation"] == "heightmap-finite-grid-authoring"
    assert body.metadata["kernel"]["authoring_boundary"] == "surface-native"


def test_heightmap_native_finite_grid_builder_refuses_malformed_or_empty_masks() -> None:
    with pytest.raises(ValueError, match="finite"):
        HeightmapAuthoringRequest(height_samples=np.asarray([[0.0, np.nan], [0.5, 0.25]], dtype=float))
    request = HeightmapAuthoringRequest(
        height_samples=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float),
        alpha_mask=np.zeros((2, 2), dtype=bool),
        alpha_mode="mask",
    )
    diagnostic = build_heightmap_mask_no_data_diagnostic(request)

    assert diagnostic.valid is False
    assert diagnostic.masked_sample_count == 4
    with pytest.raises(ValueError, match="masks every sample"):
        make_heightmap_surface_from_grid(request)


def test_heightmap_optional_import_adapter_embeds_arrays_as_native_grid() -> None:
    request = HeightmapImportRequest(
        source=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float),
        height=2.0,
        xy_scale=(0.5, 0.5),
        metadata={"kernel": {"producer": "unit-test-import"}},
    )

    dependency = heightmap_import_dependency_boundary()
    diagnostic = build_heightmap_import_diagnostic(request)
    body = import_heightmap_surface(request)
    patch = body.shells[0].patches[0]

    assert isinstance(dependency, HeightmapImportDiagnostic)
    assert dependency.supported is True
    assert diagnostic.supported is True
    assert isinstance(patch, HeightmapSurfacePatch)
    assert patch.height_samples.shape == (2, 2)
    assert patch.metadata["kernel"]["operation"] == "heightmap-import"
    assert patch.metadata["kernel"]["producer_provenance"]["operation"] == "heightmap-finite-grid-authoring"
    assert patch.metadata["kernel"]["producer"] == "unit-test-import"
    assert patch.metadata["kernel"]["import_source_kind"] == "ndarray"


def test_heightmap_optional_import_adapter_refuses_external_references_and_over_budget_inputs() -> None:
    external = HeightmapImportRequest(
        source=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float),
        embed_samples=False,
    )
    over_budget = HeightmapImportRequest(
        source=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float),
        max_sample_count=3,
    )

    external_diagnostic = build_heightmap_import_diagnostic(external)
    over_budget_diagnostic = build_heightmap_import_diagnostic(over_budget)

    assert external_diagnostic.supported is False
    assert "external references" in external_diagnostic.message
    assert over_budget_diagnostic.supported is False
    assert "max_sample_count" in over_budget_diagnostic.message
    with pytest.raises(ValueError, match="external references"):
        import_heightmap_surface(external)


def test_spline_surface_tessellation_adapters_preserve_control_payloads_and_boundary_metadata() -> None:
    control_net = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.2], [0.0, 2.0, 0.0]],
            [[1.0, 0.0, 0.1], [1.0, 1.0, 0.4], [1.0, 2.0, 0.1]],
            [[2.0, 0.0, 0.0], [2.0, 1.0, 0.2], [2.0, 2.0, 0.0]],
        ],
        dtype=float,
    )
    knots = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    patches = (
        BSplineSurfacePatch(
            family="bspline",
            degree_u=2,
            degree_v=2,
            knots_u=knots,
            knots_v=knots,
            control_net=control_net,
        ),
        NURBSSurfacePatch(
            family="nurbs",
            degree_u=2,
            degree_v=2,
            knots_u=knots,
            knots_v=knots,
            control_net=control_net,
            weights=np.ones((3, 3), dtype=float),
        ),
        SweepSurfacePatch(
            family="sweep",
            profile_points_uv=[(0.0, 0.0), (0.5, 0.25), (1.0, 0.0)],
            path=Path3D.from_points([(0.0, 0.0, 0.0), (0.0, 0.5, 1.0), (0.0, 0.0, 2.0)]),
            frame_policy="fixed",
        ),
    )
    original_identities = [patch.stable_identity for patch in patches]

    meshes = [tessellate_surface_patch(patch, preview_tessellation_request()) for patch in patches]

    assert [patch.stable_identity for patch in patches] == original_identities
    for patch, mesh in zip(patches, meshes):
        assert mesh.vertices.shape[0] > 0
        assert mesh.faces.shape[0] > 0
        assert mesh.metadata["surface_family"] == patch.family
        assert mesh.metadata["surface_patch_id"] == patch.stable_identity
        assert mesh.metadata["adapter_lossiness"] == "lossy"
        assert mesh.metadata["tessellation_sampling_kind"] == "parametric-loop"
        assert mesh.metadata["tessellation_adapter_boundary"] == "tessellation"


def test_family_adapter_dispatch_refuses_unsupported_surface_family() -> None:
    patch = PlanarSurfacePatch(family="unsupported")

    with pytest.raises(ValueError, match="Unsupported surface patch family"):
        tessellate_surface_patch(patch)


def test_shell_tessellation_records_family_adapters_and_welds_supported_seam_boundaries() -> None:
    planar = PlanarSurfacePatch(family="planar")
    ruled = RuledSurfacePatch(
        family="ruled",
        start_curve=((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        end_curve=((2.0, 0.0, 0.0), (2.0, 1.0, 0.0)),
    )
    shell = SurfaceShell(
        patches=(planar, ruled),
        seams=(
            SurfaceSeam(
                seam_id="planar-ruled",
                boundaries=(SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")),
            ),
        ),
    )

    mesh = tessellate_surface_shell(shell, preview_tessellation_request())

    assert mesh.vertices.shape[0] == 6
    assert {adapter["family"] for adapter in mesh.metadata["tessellation_family_adapters"]} == {"planar", "ruled"}
    assert all(adapter["supports_seam_boundaries"] is True for adapter in mesh.metadata["tessellation_family_adapters"])


def test_cross_family_boundary_descriptor_uses_patch_evaluator_without_mesh() -> None:
    patch = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=((1.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
    )

    descriptor = extract_surface_boundary_descriptor(patch, "left", sample_count=5)

    assert isinstance(descriptor, SurfaceBoundaryDescriptor)
    assert descriptor.family == "revolution"
    assert descriptor.comparison_kind == "parametric-edge"
    assert descriptor.exact is True
    assert descriptor.parameter_points.shape == (5, 2)


def test_cross_family_seam_validation_records_participation_continuity_and_adjacency() -> None:
    planar = PlanarSurfacePatch(family="planar")
    bspline = BSplineSurfacePatch(
        family="bspline",
        control_net=[
            [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
            [(2.0, 0.0, 0.0), (2.0, 1.0, 0.0)],
        ],
    )
    seam = SurfaceSeam(
        seam_id="planar-bspline",
        boundaries=(SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")),
        continuity="C0",
    )
    shell = SurfaceShell(patches=(planar, bspline), seams=(seam,))

    result = validate_surface_seam_participation(shell, seam)
    continuity = classify_surface_seam_continuity(shell, seam)
    adjacency = surface_adjacency_from_seams(shell)

    assert isinstance(result, SurfaceSeamValidationResult)
    assert isinstance(result.continuity, SurfaceContinuityMetadata)
    assert all(isinstance(record, SurfaceSeamParticipationRecord) for record in result.participation)
    assert result.compatible is True
    assert result.continuity.classified == "C0"
    assert result.continuity.exact_comparison is True
    assert continuity.classified == "C0"
    assert len(result.adjacency_updates) == 2
    assert adjacency == result.adjacency_updates
    assert {record.descriptor.family for record in result.participation} == {"planar", "bspline"}


def test_cross_family_seam_validation_reports_incompatible_boundaries() -> None:
    planar = PlanarSurfacePatch(family="planar")
    bspline = BSplineSurfacePatch(
        family="bspline",
        control_net=[
            [(3.0, 0.0, 0.0), (3.0, 1.0, 0.0)],
            [(4.0, 0.0, 0.0), (4.0, 1.0, 0.0)],
        ],
    )
    seam = SurfaceSeam("gap", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")))
    shell = SurfaceShell(patches=(planar, bspline), seams=(seam,))

    result = validate_surface_seam_participation(shell, seam)

    assert result.compatible is False
    assert result.continuity.classified == "incompatible"
    assert "boundary positions differ" in result.diagnostics[0]


def test_cross_family_seam_validation_reports_unsupported_continuity_request() -> None:
    planar = PlanarSurfacePatch(family="planar")
    ruled = RuledSurfacePatch(
        family="ruled",
        start_curve=((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        end_curve=((2.0, 0.0, 0.0), (2.0, 1.0, 0.0)),
    )
    seam = SurfaceSeam("unsupported-continuity", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")), continuity="G1")
    shell = SurfaceShell(patches=(planar, ruled), seams=(seam,))

    result = validate_surface_seam_participation(shell, seam)

    assert result.compatible is False
    assert result.continuity.classified == "incompatible"
    assert any("unsupported continuity request" in diagnostic for diagnostic in result.diagnostics)


def test_surface_continuity_support_records_supported_and_future_classes() -> None:
    supported = surface_continuity_support(SurfaceContinuityRequest("G0", source="loft"))
    future = surface_continuity_support("G1")

    assert SUPPORTED_SEAM_CONTINUITY_CLASSES == ("C0", "G0")
    assert isinstance(supported, SurfaceContinuitySupportRecord)
    assert supported.supported is True
    assert supported.support_state == "supported"
    assert future.supported is False
    assert future.support_state == "not-yet-implemented"
    assert "supported classes are C0, G0" in future.diagnostic


def test_surface_unsupported_continuity_diagnostic_is_structured() -> None:
    diagnostic = build_surface_unsupported_continuity_diagnostic("C2")

    assert isinstance(diagnostic, SurfaceUnsupportedContinuityDiagnostic)
    assert diagnostic.requested == "C2"
    assert diagnostic.supported_classes == ("C0", "G0")
    assert "unsupported continuity request" in diagnostic.message
    with pytest.raises(ValueError, match="Supported seam continuity requests"):
        build_surface_unsupported_continuity_diagnostic("C0")


def test_surface_seam_continuity_constraint_normalizes_higher_order_requests() -> None:
    seam = SurfaceSeam(
        "join",
        (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")),
        continuity="g2",
    )
    policy = SurfaceContinuityTolerancePolicy(
        position_tolerance=1e-8,
        tangent_tolerance=1e-5,
        curvature_tolerance=1e-4,
    )

    constraint = normalize_surface_seam_continuity_constraint(seam, tolerance_policy=policy)

    assert isinstance(constraint, SurfaceSeamContinuityConstraint)
    assert constraint.requested == "G2"
    assert constraint.tolerance_policy is policy
    assert tuple(use.role for use in constraint.boundary_uses) == ("first", "second")
    assert all(isinstance(use, SurfaceSeamBoundaryUseRef) for use in constraint.boundary_uses)
    assert validate_surface_seam_continuity_constraint(constraint) == ()
    assert constraint.canonical_payload()["requested"] == "G2"


def test_surface_seam_continuity_constraint_supports_explicit_request_source_and_open_boundary() -> None:
    seam = SurfaceSeam("open-top", (SurfaceBoundaryRef(0, "top"),), continuity="C0")
    request = SurfaceContinuityRequest("C1", source="authored-loft")

    constraint = normalize_surface_seam_continuity_constraint(seam, request=request)

    assert constraint.requested == "C1"
    assert constraint.source == "authored-loft"
    assert tuple(use.role for use in constraint.boundary_uses) == ("open",)
    assert validate_surface_seam_continuity_constraint(constraint) == ()


def test_surface_seam_continuity_constraint_validator_reports_invalid_requests() -> None:
    constraint = SurfaceSeamContinuityConstraint(
        seam_id="bad",
        requested="H1",
        boundary_uses=(
            SurfaceSeamBoundaryUseRef("bad", SurfaceBoundaryRef(0, "right"), role="first"),
            SurfaceSeamBoundaryUseRef("bad", SurfaceBoundaryRef(0, "right"), role="second"),
        ),
    )

    diagnostics = validate_surface_seam_continuity_constraint(constraint)

    assert all(isinstance(diagnostic, SurfaceContinuityConstraintDiagnostic) for diagnostic in diagnostics)
    assert {diagnostic.code for diagnostic in diagnostics} == {"invalid-continuity", "duplicate-boundary"}
    assert diagnostics[0].canonical_payload()["seam_id"] == "bad"
    with pytest.raises(ValueError, match="positive finite"):
        SurfaceContinuityTolerancePolicy(tangent_tolerance=0.0)


def test_surface_boundary_derivative_evaluator_samples_planar_boundary() -> None:
    patch = PlanarSurfacePatch(
        family="planar",
        origin=(0.0, 0.0, 0.0),
        u_axis=(2.0, 0.0, 0.0),
        v_axis=(0.0, 3.0, 0.0),
    )

    summary = evaluate_surface_boundary_derivatives(patch, "right", patch_index=2, sample_count=3)

    assert isinstance(summary, SurfaceBoundaryDerivativeSummary)
    assert summary.supported is True
    assert summary.diagnostics == ()
    assert len(summary.samples) == 3
    assert all(isinstance(sample, SurfaceBoundaryDerivativeSample) for sample in summary.samples)
    assert all(sample.boundary == SurfaceBoundaryRef(2, "right") for sample in summary.samples)
    assert all(sample.du == pytest.approx((2.0, 0.0, 0.0)) for sample in summary.samples)
    assert all(sample.dv == pytest.approx((0.0, 3.0, 0.0)) for sample in summary.samples)
    assert all(sample.normal == pytest.approx((0.0, 0.0, 1.0)) for sample in summary.samples)
    assert all(sample.tangent == pytest.approx((0.0, 1.0, 0.0)) for sample in summary.samples)
    assert summary.canonical_payload()["supported"] is True


def test_surface_boundary_derivative_evaluator_samples_ruled_and_revolution_boundaries() -> None:
    ruled = RuledSurfacePatch(
        family="ruled",
        start_curve=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        end_curve=((1.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )
    revolution = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=((1.0, 0.0, -1.0), (1.0, 0.0, 1.0)),
    )

    ruled_summary = evaluate_surface_boundary_derivatives(ruled, "top", sample_count=4)
    revolution_summary = evaluate_surface_boundary_derivatives(revolution, "left", sample_count=4)

    assert ruled_summary.supported is True
    assert revolution_summary.supported is True
    assert len(ruled_summary.samples) == 4
    assert len(revolution_summary.samples) == 4
    assert all(np.linalg.norm(sample.normal) == pytest.approx(1.0) for sample in ruled_summary.samples)
    assert all(np.linalg.norm(sample.normal) == pytest.approx(1.0) for sample in revolution_summary.samples)
    assert ruled_summary.residual_metadata["second_derivative_method"] == "finite-difference"


def test_surface_boundary_derivative_evaluator_reports_unsupported_implicit_family() -> None:
    summary = evaluate_surface_boundary_derivatives(ImplicitSurfacePatch(family="implicit"), "left")

    assert summary.supported is False
    assert summary.samples == ()
    assert all(isinstance(diagnostic, SurfaceBoundaryDerivativeDiagnostic) for diagnostic in summary.diagnostics)
    assert summary.diagnostics[0].code == "unsupported-family"


def test_higher_order_continuity_residual_validation_accepts_c1_planar_seam() -> None:
    first_patch = PlanarSurfacePatch(family="planar")
    second_patch = PlanarSurfacePatch(family="planar", origin=(1.0, 0.0, 0.0))
    seam = SurfaceSeam("c1", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")), continuity="C1")
    constraint = normalize_surface_seam_continuity_constraint(seam)
    first = evaluate_surface_boundary_derivatives(first_patch, "right", patch_index=0, sample_count=3)
    second = evaluate_surface_boundary_derivatives(second_patch, "left", patch_index=1, sample_count=3)

    report = validate_higher_order_surface_continuity(constraint, first, second)

    assert isinstance(report, SurfaceHigherOrderContinuityValidationReport)
    assert report.passed is True
    assert isinstance(report.residuals, SurfaceContinuityResidualMetrics)
    assert isinstance(report.observed, SurfaceObservedContinuityClassRecord)
    assert report.observed.passed_requested is True
    assert "C1" in report.observed.observed_classes
    assert report.residuals.max_position_delta == pytest.approx(0.0)
    assert report.canonical_payload()["passed"] is True


def test_higher_order_continuity_residual_validation_fails_without_downgrading_request() -> None:
    first_patch = PlanarSurfacePatch(family="planar")
    second_patch = PlanarSurfacePatch(
        family="planar",
        origin=(1.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.25),
    )
    seam = SurfaceSeam("g1", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")), continuity="G1")
    constraint = normalize_surface_seam_continuity_constraint(seam)
    first = evaluate_surface_boundary_derivatives(first_patch, "right", patch_index=0, sample_count=3)
    second = evaluate_surface_boundary_derivatives(second_patch, "left", patch_index=1, sample_count=3)

    report = validate_higher_order_surface_continuity(constraint, first, second)

    assert report.passed is False
    assert report.observed is not None
    assert report.observed.requested == "G1"
    assert report.observed.passed_requested is False
    assert "G1" not in report.observed.observed_classes
    assert report.residuals is not None
    assert report.residuals.max_normal_delta > constraint.tolerance_policy.tangent_tolerance


def test_higher_order_continuity_residual_classifier_reports_observed_classes() -> None:
    residuals = SurfaceContinuityResidualMetrics(
        max_position_delta=0.0,
        max_tangent_delta=0.0,
        max_normal_delta=0.0,
        max_second_derivative_delta=0.0,
        sample_count=2,
    )
    policy = SurfaceContinuityTolerancePolicy()

    observed = classify_surface_continuity_residuals("G2", residuals, policy)

    assert observed.passed_requested is True
    assert observed.observed_classes == ("C0", "G0", "G1", "C1", "G2", "C2")
    with pytest.raises(ValueError, match="at least one paired sample"):
        compute_surface_continuity_residual_metrics(
            SurfaceBoundaryDerivativeSummary(family="planar", boundary_id="left"),
            SurfaceBoundaryDerivativeSummary(family="planar", boundary_id="right"),
        )


def test_higher_order_continuity_violation_locators_name_parameter_hotspots() -> None:
    first_patch = PlanarSurfacePatch(family="planar")
    second_patch = PlanarSurfacePatch(
        family="planar",
        origin=(1.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.25),
    )
    seam = SurfaceSeam("g1", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")), continuity="G1")
    constraint = normalize_surface_seam_continuity_constraint(seam)
    first = evaluate_surface_boundary_derivatives(first_patch, "right", patch_index=0, sample_count=3)
    second = evaluate_surface_boundary_derivatives(second_patch, "left", patch_index=1, sample_count=3)
    report = validate_higher_order_surface_continuity(constraint, first, second)

    diagnostics = build_surface_continuity_violation_locators(report, first, second)

    assert isinstance(diagnostics, SurfaceContinuityViolationDiagnostics)
    assert diagnostics.has_violations is True
    assert all(isinstance(violation, SurfaceContinuityViolationRecord) for violation in diagnostics.violations)
    normal = next(violation for violation in diagnostics.violations if violation.residual_kind == "normal")
    assert isinstance(normal.locator, SurfaceContinuitySeamParameterLocator)
    assert normal.locator.seam_id == "g1"
    assert normal.locator.first_boundary == SurfaceBoundaryRef(0, "right")
    assert normal.locator.second_boundary == SurfaceBoundaryRef(1, "left")
    assert "failed requested G1 normal residual" in normal.message
    assert format_surface_continuity_violation_diagnostic(normal) == normal.message
    assert diagnostics.canonical_payload()["has_violations"] is True


def test_seam_continuity_negative_diagnostic_fixtures_feed_matrix() -> None:
    unsupported = build_surface_unsupported_continuity_diagnostic("C2")
    invalid_constraint = SurfaceSeamContinuityConstraint(
        seam_id="bad",
        requested="H1",
        boundary_uses=(
            SurfaceSeamBoundaryUseRef("bad", SurfaceBoundaryRef(0, "right"), role="first"),
            SurfaceSeamBoundaryUseRef("bad", SurfaceBoundaryRef(0, "right"), role="second"),
        ),
    )
    invalid_constraint_diagnostics = validate_surface_seam_continuity_constraint(invalid_constraint)
    first_patch = PlanarSurfacePatch(family="planar")
    second_patch = PlanarSurfacePatch(
        family="planar",
        origin=(1.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.25),
    )
    seam = SurfaceSeam("g1", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")), continuity="G1")
    constraint = normalize_surface_seam_continuity_constraint(seam)
    first = evaluate_surface_boundary_derivatives(first_patch, "right", patch_index=0, sample_count=3)
    second = evaluate_surface_boundary_derivatives(second_patch, "left", patch_index=1, sample_count=3)
    report = validate_higher_order_surface_continuity(constraint, first, second)
    locator_diagnostics = build_surface_continuity_violation_locators(report, first, second)
    normal_violation = next(
        violation for violation in locator_diagnostics.violations if violation.residual_kind == "normal"
    )

    fixtures = (
        _seam_continuity_negative_fixture(
            "seam-continuity/unsupported-request",
            {
                "message": unsupported.message,
                "requested": unsupported.requested,
                "supported_classes": unsupported.supported_classes,
            },
            expected_keys=(
                ExpectedDiagnosticKeyRecord(("requested",), "C2"),
                ExpectedDiagnosticKeyRecord(("supported_classes",), ("C0", "G0")),
                ExpectedDiagnosticKeyRecord(("message",)),
            ),
        ),
        _seam_continuity_negative_fixture(
            "seam-continuity/invalid-constraint",
            invalid_constraint_diagnostics[0],
            expected_keys=(
                ExpectedDiagnosticKeyRecord(("code",), "invalid-continuity"),
                ExpectedDiagnosticKeyRecord(("seam_id",), "bad"),
                ExpectedDiagnosticKeyRecord(("message",)),
            ),
        ),
        _seam_continuity_negative_fixture(
            "seam-continuity/failed-residual-locator",
            {
                "locator": normal_violation.locator.canonical_payload(),
                "observed": report.observed.canonical_payload() if report.observed is not None else None,
                "requested": normal_violation.requested,
                "residual_kind": normal_violation.residual_kind,
                "residual_value": normal_violation.residual_value,
                "seam_id": normal_violation.seam_id,
                "tolerance": normal_violation.tolerance,
            },
            expected_keys=(
                ExpectedDiagnosticKeyRecord(("requested",), "G1"),
                ExpectedDiagnosticKeyRecord(("residual_kind",), "normal"),
                ExpectedDiagnosticKeyRecord(("locator", "seam_id"), "g1"),
                ExpectedDiagnosticKeyRecord(("locator", "first_boundary", "boundary_id"), "right"),
                ExpectedDiagnosticKeyRecord(("locator", "second_boundary", "boundary_id"), "left"),
                ExpectedDiagnosticKeyRecord(("observed", "passed_requested"), False),
            ),
        ),
    )

    matrix = evaluate_negative_diagnostic_fixture_matrix(fixtures, required_domains=("seam-continuity",))

    assert matrix.passed is True
    assert not matrix.diagnostics
    assert matrix.domain_coverage[0].fixture_count == 3


def test_higher_order_continuity_violation_locators_are_empty_for_passing_report() -> None:
    first_patch = PlanarSurfacePatch(family="planar")
    second_patch = PlanarSurfacePatch(family="planar", origin=(1.0, 0.0, 0.0))
    seam = SurfaceSeam("c1", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")), continuity="C1")
    constraint = normalize_surface_seam_continuity_constraint(seam)
    first = evaluate_surface_boundary_derivatives(first_patch, "right", patch_index=0, sample_count=3)
    second = evaluate_surface_boundary_derivatives(second_patch, "left", patch_index=1, sample_count=3)
    report = validate_higher_order_surface_continuity(constraint, first, second)

    diagnostics = build_surface_continuity_violation_locators(report, first, second)

    assert report.passed is True
    assert diagnostics.has_violations is False
    assert diagnostics.violations == ()


def test_surface_continuity_enforcement_refuses_validation_only_and_source_mutation() -> None:
    seam = SurfaceSeam("c1", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")), continuity="C1")
    constraint = normalize_surface_seam_continuity_constraint(seam)
    request = SurfaceContinuityEnforcementRequest(
        operation_id="loft-1",
        producer="loft",
        constraint=constraint,
        owns_generated_geometry=False,
        mutates_source_geometry=True,
    )

    diagnostics = check_surface_continuity_enforcement_eligibility(request)

    assert all(isinstance(diagnostic, SurfaceContinuityEnforcementRefusalDiagnostic) for diagnostic in diagnostics)
    assert {diagnostic.code for diagnostic in diagnostics} == {"validation-only", "source-mutation-forbidden"}
    assert diagnostics[0].canonical_payload()["operation_id"] == "loft-1"


def test_surface_continuity_enforcement_accepts_operation_owned_passing_output() -> None:
    first_patch = PlanarSurfacePatch(family="planar")
    second_patch = PlanarSurfacePatch(family="planar", origin=(1.0, 0.0, 0.0))
    seam = SurfaceSeam("c1", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")), continuity="C1")
    constraint = normalize_surface_seam_continuity_constraint(seam)
    first = evaluate_surface_boundary_derivatives(first_patch, "right", patch_index=0, sample_count=3)
    second = evaluate_surface_boundary_derivatives(second_patch, "left", patch_index=1, sample_count=3)
    report = validate_higher_order_surface_continuity(constraint, first, second)
    request = SurfaceContinuityEnforcementRequest(
        operation_id="blend-1",
        producer="blend",
        constraint=constraint,
        owns_generated_geometry=True,
    )

    result = validate_surface_continuity_enforcement_result(request, report)

    assert isinstance(result, SurfaceContinuityEnforcementResult)
    assert result.accepted is True
    assert result.diagnostics == ()
    assert result.validation_report is report
    assert result.canonical_payload()["accepted"] is True


def test_surface_continuity_enforcement_rejects_operation_owned_failed_output() -> None:
    first_patch = PlanarSurfacePatch(family="planar")
    second_patch = PlanarSurfacePatch(
        family="planar",
        origin=(1.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.25),
    )
    seam = SurfaceSeam("g1", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left")), continuity="G1")
    constraint = normalize_surface_seam_continuity_constraint(seam)
    first = evaluate_surface_boundary_derivatives(first_patch, "right", patch_index=0, sample_count=3)
    second = evaluate_surface_boundary_derivatives(second_patch, "left", patch_index=1, sample_count=3)
    report = validate_higher_order_surface_continuity(constraint, first, second)
    request = SurfaceContinuityEnforcementRequest(
        operation_id="sweep-1",
        producer="sweep",
        constraint=constraint,
        owns_generated_geometry=True,
    )

    result = validate_surface_continuity_enforcement_result(request, report)

    assert result.accepted is False
    assert {diagnostic.code for diagnostic in result.diagnostics} == {"validation-failed"}
    assert "failed requested G1" in result.diagnostics[0].message


def test_subdivision_boundary_descriptor_records_approximation_and_implicit_seams_refuse() -> None:
    descriptor = extract_surface_boundary_descriptor(SubdivisionSurfacePatch(family="subdivision"), "left")
    heightmap_descriptor = extract_surface_boundary_descriptor(
        HeightmapSurfacePatch(family="heightmap", height_samples=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float)),
        "left",
    )
    displacement_descriptor = extract_surface_boundary_descriptor(
        DisplacementSurfacePatch(
            family="displacement",
            source_patch=PlanarSurfacePatch(family="planar"),
            displacement_samples=np.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=float),
            projection_bounds=(0.0, 1.0, 0.0, 1.0),
        ),
        "left",
    )

    assert descriptor.exact is False
    assert descriptor.approximation_metadata["method"] == "finite_subdivision_boundary"
    assert heightmap_descriptor.exact is False
    assert heightmap_descriptor.approximation_metadata["method"] == "sampled_heightmap_boundary"
    assert heightmap_descriptor.approximation_metadata["heightmap_shape"] == (2, 2)
    assert displacement_descriptor.exact is False
    assert displacement_descriptor.approximation_metadata["method"] == "sampled_displacement_boundary"
    assert displacement_descriptor.approximation_metadata["sample_shape"] == (2, 2)

    with pytest.raises(ValueError, match="cannot participate in parametric seams"):
        SurfaceShell(
            patches=(ImplicitSurfacePatch(family="implicit"), PlanarSurfacePatch(family="planar")),
            seams=(SurfaceSeam("implicit-planar", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left"))),),
        )


def test_advanced_family_boundary_support_matrix_names_exact_approximate_and_refusal_states() -> None:
    records = surface_family_boundary_support_matrix()
    by_family = {record.family: record for record in records}

    assert all(isinstance(record, SurfaceFamilyBoundarySupportRecord) for record in records)
    assert set(by_family) == set(ADVANCED_PATCH_FAMILIES)
    assert by_family["bspline"].boundary_support == "exact"
    assert by_family["nurbs"].higher_order_residuals is True
    assert by_family["subdivision"].boundary_support == "approximate"
    assert by_family["subdivision"].approximation_method == "finite_subdivision_boundary"
    assert by_family["heightmap"].approximation_method == "sampled_heightmap_boundary"
    assert by_family["displacement"].approximation_method == "sampled_displacement_boundary"
    assert by_family["implicit"].boundary_support == "unsupported"
    assert "no canonical parametric seam boundary" in by_family["implicit"].diagnostic


def test_planar_patch_rejects_collinear_axes_and_multiple_outer_trims() -> None:
    with pytest.raises(ValueError, match="linearly independent"):
        PlanarSurfacePatch(
            family="planar",
            origin=(0.0, 0.0, 0.0),
            u_axis=(1.0, 0.0, 0.0),
            v_axis=(2.0, 0.0, 0.0),
        )

    with pytest.raises(ValueError, match="more than one outer trim"):
        PlanarSurfacePatch(
            family="planar",
            trim_loops=(
                TrimLoop([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], category="outer"),
                TrimLoop([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)], category="outer"),
            ),
        )


def test_surface_shell_validates_seams_and_adjacency_and_applies_transform() -> None:
    patch_a = PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 0.0))
    patch_b = PlanarSurfacePatch(family="planar", origin=(5.0, 0.0, 0.0))
    seam = SurfaceSeam(
        seam_id="shared-0",
        boundaries=(
            SurfaceBoundaryRef(0, "right"),
            SurfaceBoundaryRef(1, "left"),
        ),
        continuity="C1",
    )
    adjacency = SurfaceAdjacencyRecord(
        source=SurfaceBoundaryRef(0, "right"),
        target=SurfaceBoundaryRef(1, "left"),
        seam_id="shared-0",
        continuity="C1",
    )
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 2.0, 0.0]
    shell = SurfaceShell(
        patches=(patch_a, patch_b),
        connected=False,
        seams=(seam,),
        adjacency=(adjacency,),
        transform_matrix=transform,
        metadata={"shell_role": "outer"},
    )

    assert shell.patch_count == 2
    assert len(shell.adjacency_for_patch(0)) == 1
    world_patches = shell.iter_patches(world=True)
    assert np.allclose(world_patches[0].point_at(0.0, 0.0), np.array([0.0, 2.0, 0.0]))
    assert shell.bounds_estimate() == (0.0, 6.0, 2.0, 3.0, 0.0, 0.0)
    assert isinstance(shell.stable_identity, str)

    with pytest.raises(ValueError, match="unknown seam_id"):
        SurfaceShell(
            patches=(patch_a, patch_b),
            seams=(seam,),
            adjacency=(
                SurfaceAdjacencyRecord(
                    source=SurfaceBoundaryRef(0, "right"),
                    target=SurfaceBoundaryRef(1, "left"),
                    seam_id="missing",
                ),
            ),
        )


def test_surface_body_requires_ordered_shells_and_composes_transform() -> None:
    shell_a = make_surface_shell([PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 0.0))])
    shell_b = make_surface_shell([PlanarSurfacePatch(family="planar", origin=(0.0, 5.0, 0.0))])
    body = make_surface_body([shell_a, shell_b], metadata={"name": "demo"})

    assert body.shell_count == 2
    assert body.patch_count == 2
    assert body.iter_shells(world=False) == (shell_a, shell_b)
    assert body.iter_patches(world=False) == tuple(shell_a.patches + shell_b.patches)
    assert body.bounds_estimate() == (0.0, 1.0, 0.0, 6.0, 0.0, 0.0)

    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 4.0]
    moved = body.with_transform(transform)
    moved_bounds = moved.bounds_estimate()
    assert moved_bounds == (0.0, 1.0, 0.0, 6.0, 4.0, 4.0)
    assert moved.stable_identity != body.stable_identity
    assert moved.cache_key == moved.stable_identity

    with pytest.raises(ValueError, match="at least one shell"):
        SurfaceBody(())


def test_metadata_namespacing_and_override_rules_are_explicit() -> None:
    patch = PlanarSurfacePatch(
        family="planar",
        metadata={
            "kernel": {"color": "blue", "detail": "patch"},
            "consumer": {"preview_label": "Patch A"},
        },
    )
    shell = SurfaceShell(
        patches=(patch,),
        metadata={
            "kernel": {"detail": "shell", "layer": "mid"},
            "consumer": {"preview_label": "Shell A"},
        },
    )
    body = SurfaceBody(
        shells=(shell,),
        metadata={
            "kernel": {"assembly": "body"},
            "consumer": {"legend": "Body A"},
        },
    )

    assert patch.kernel_metadata() == {"color": "blue", "detail": "patch"}
    assert patch.consumer_metadata() == {"preview_label": "Patch A"}
    assert shell.merged_kernel_metadata(body.kernel_metadata()) == {
        "assembly": "body",
        "detail": "shell",
        "layer": "mid",
    }
    assert patch.merged_kernel_metadata(shell.merged_kernel_metadata(body.kernel_metadata())) == {
        "assembly": "body",
        "detail": "patch",
        "layer": "mid",
        "color": "blue",
    }
    assert patch.merged_consumer_metadata(shell.merged_consumer_metadata(body.consumer_metadata())) == {
        "legend": "Body A",
        "preview_label": "Patch A",
    }

    with pytest.raises(ValueError, match="may only use 'kernel' and 'consumer'"):
        PlanarSurfacePatch(
            family="planar",
            metadata={"kernel": {}, "other": {}},
        ).kernel_metadata()


def test_tessellation_request_normalization_is_deterministic() -> None:
    preview = normalize_tessellation_request(preview_tessellation_request())
    export = normalize_tessellation_request(export_tessellation_request())
    analysis = normalize_tessellation_request(analysis_tessellation_request())
    overridden = normalize_tessellation_request(
        TessellationRequest(
            intent="export",
            quality_preset="fine",
            chord_tolerance=0.125,
            max_edge_length=0.2,
        )
    )

    assert preview.quality_preset == "preview"
    assert export.quality_preset == "fine"
    assert analysis.quality_preset == "analysis"
    assert export.require_watertight is True
    assert overridden.chord_tolerance == pytest.approx(0.125)
    assert overridden.max_edge_length == pytest.approx(0.2)
    assert overridden.cache_key == normalize_tessellation_request(
        TessellationRequest(
            intent="export",
            quality_preset="fine",
            chord_tolerance=0.125,
            max_edge_length=0.2,
        )
    ).cache_key


def test_tessellation_helper_contract_accepts_only_surface_boundary_inputs() -> None:
    body = make_surface_box(size=(1.0, 1.0, 1.0))
    record = make_surface_to_mesh_adapter_record(body, preview_tessellation_request())
    diagnostic = validate_tessellation_helper_boundary_input({"kind": "box", "size": (1.0, 1.0, 1.0)})

    assert isinstance(record, SurfaceToMeshAdapterRecord)
    assert record.source_type == "SurfaceBody"
    assert record.source_identity == body.stable_identity
    assert record.helper_contract == SURFACE_TO_MESH_HELPER_CONTRACT
    assert record.helper_contract.consumes_authored_primitive_arguments is False
    assert isinstance(diagnostic, TessellationBoundaryViolationDiagnostic)
    assert diagnostic.received_type == "dict"
    assert "SurfaceBody" in diagnostic.expected_inputs


def test_mesh_boundary_negative_fixtures_feed_diagnostic_matrix() -> None:
    fixtures = (
        _mesh_boundary_negative_fixture(
            "mesh-boundary/dict-primitive-input",
            lambda: validate_tessellation_helper_boundary_input({"kind": "box", "size": (1.0, 1.0, 1.0)}),
            expected_keys=(
                ExpectedDiagnosticKeyRecord(("helper_name",), "surface-to-mesh-adapter"),
                ExpectedDiagnosticKeyRecord(("received_type",), "dict"),
                ExpectedDiagnosticKeyRecord(("message",)),
            ),
        ),
        _mesh_boundary_negative_fixture(
            "mesh-boundary/list-points-input",
            lambda: validate_tessellation_helper_boundary_input([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]),
            expected_keys=(
                ExpectedDiagnosticKeyRecord(("helper_name",), "surface-to-mesh-adapter"),
                ExpectedDiagnosticKeyRecord(("received_type",), "list"),
                ExpectedDiagnosticKeyRecord(("message",)),
            ),
        ),
        _mesh_boundary_negative_fixture(
            "mesh-boundary/adapter-record-refuses-primitive-args",
            lambda: make_surface_to_mesh_adapter_record({"kind": "sphere", "radius": 1.0}),  # type: ignore[arg-type]
            expected_keys=(
                ExpectedDiagnosticKeyRecord(("code",), "ValueError"),
                ExpectedDiagnosticKeyRecord(("message",)),
            ),
        ),
    )

    report = evaluate_negative_diagnostic_fixture_matrix(fixtures, required_domains=("mesh-boundary",))

    assert report.passed is True
    assert report.domain_coverage[0].fixture_count == 3
    assert any(
        fixture.fixture_id == "mesh-boundary/adapter-record-refuses-primitive-args"
        and "tessellation boundary" in fixture.expected_snapshot.payload["message"]  # type: ignore[index, union-attr]
        for fixture in fixtures
    )
    assert all(fixture.expected_snapshot is not None for fixture in fixtures)


def test_surface_primitive_family_appropriateness_matrix_stays_exact_and_native() -> None:
    primitive_families = {
        "box": {patch.family for patch in make_surface_box(size=(1.0, 1.0, 1.0)).iter_patches()},
        "polyhedron": {patch.family for patch in make_surface_polyhedron(faces=4, radius=0.5).iter_patches()},
        "prism": {patch.family for patch in make_surface_prism(base_size=(1.0, 1.0), top_size=(0.75, 0.5), height=1.0).iter_patches()},
        "cylinder": {patch.family for patch in make_surface_cylinder(radius=0.5, height=1.0).iter_patches()},
        "cone": {patch.family for patch in make_surface_cone(bottom_diameter=1.0, top_diameter=0.2, height=1.0).iter_patches()},
        "sphere": {patch.family for patch in make_surface_sphere(radius=0.5).iter_patches()},
        "torus": {patch.family for patch in make_surface_torus(major_radius=1.0, minor_radius=0.2).iter_patches()},
    }

    assert primitive_families["box"] == {"planar"}
    assert primitive_families["polyhedron"] == {"planar"}
    assert "ruled" in primitive_families["prism"]
    assert primitive_families["prism"] <= {"planar", "ruled"}
    assert "revolution" in primitive_families["cylinder"]
    assert "revolution" in primitive_families["cone"]
    assert primitive_families["sphere"] == {"revolution"}
    assert primitive_families["torus"] == {"revolution"}

    heightmap_patch = HeightmapSurfacePatch(family="heightmap", height_samples=np.asarray([[0.0, 1.0], [0.25, 0.5]]))
    displacement_patch = DisplacementSurfacePatch(
        family="displacement",
        source_patch=PlanarSurfacePatch(family="planar"),
        displacement_samples=np.asarray([[0.0, 1.0], [0.25, 0.5]]),
        projection_bounds=(0.0, 1.0, 0.0, 1.0),
    )
    assert heightmap_patch.family == "heightmap"
    assert displacement_patch.family == "displacement"


def test_public_primitive_patch_producer_selection_is_explicit_and_surface_native() -> None:
    inventory = primitive_patch_producer_selection_inventory()
    by_caller = {record.caller_id: record for record in inventory}

    assert all(isinstance(record, PrimitivePatchProducerSelectionRecord) for record in inventory)
    assert set(by_caller) == {record.caller_id for record in primitive_csg_route_inventory()}
    assert select_primitive_patch_producer("primitive.make_sphere").selected_patch_families == ("revolution",)
    assert select_primitive_patch_producer("primitive.make_box").mesh_substitution_allowed is False

    result = make_box(size=(1.0, 1.0, 1.0))
    assert isinstance(result, SurfaceBody)
    assert {patch.family for patch in result.iter_patches()} == {"planar"}
    with pytest.raises(ValueError, match="Unsupported primitive surface producer"):
        select_primitive_patch_producer("primitive.make_unknown")


def test_unsupported_primitive_producer_diagnostic_is_explicit() -> None:
    diagnostic = unsupported_primitive_producer_diagnostic(
        "primitive.make_unknown",
        requested_backend="surface",
        reason="no surface constructor is registered",
    )

    assert isinstance(diagnostic, UnsupportedPrimitiveProducerDiagnostic)
    assert diagnostic.primitive == "unknown"
    assert "no surface constructor" in diagnostic.message
    assert diagnostic.canonical_payload()["requested_backend"] == "surface"


def test_planar_patch_tessellates_to_mesh_from_domain_or_trim() -> None:
    plain_patch = PlanarSurfacePatch(family="planar")
    plain_mesh = tessellate_surface_patch(plain_patch)
    assert plain_mesh.n_vertices == 4
    assert plain_mesh.n_faces == 2

    trimmed_patch = PlanarSurfacePatch(
        family="planar",
        trim_loops=(
            TrimLoop([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], category="outer"),
            TrimLoop([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)], category="inner"),
        ),
    )
    trimmed_mesh = tessellate_surface_patch(trimmed_patch)
    assert trimmed_mesh.n_vertices == 8
    assert trimmed_mesh.n_faces > 2
    assert trimmed_mesh.metadata["surface_family"] == "planar"


def test_ruled_patch_tessellates_to_mesh_from_domain() -> None:
    patch = RuledSurfacePatch(
        family="ruled",
        start_curve=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        end_curve=((1.0, 0.0, 0.5), (1.0, 1.0, 0.5)),
    )
    mesh = tessellate_surface_patch(patch)

    assert mesh.n_vertices == 4
    assert mesh.n_faces == 2
    assert mesh.metadata["surface_family"] == "ruled"


def test_revolution_patch_tessellates_to_mesh_from_domain() -> None:
    patch = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=((1.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
        axis_origin=(0.0, 0.0, 0.0),
        axis_direction=(0.0, 0.0, 1.0),
        sweep_angle_deg=180.0,
    )
    mesh = tessellate_surface_patch(patch)

    assert mesh.n_vertices > 4
    assert mesh.n_faces > 2
    assert mesh.metadata["surface_family"] == "revolution"


def test_surface_body_tessellation_classifies_open_and_closed_outputs() -> None:
    open_body = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar")])])
    open_result = tessellate_surface_body(open_body, preview_tessellation_request())
    assert open_result.classification == "open"
    assert open_result.analysis.is_watertight is False
    assert open_result.mesh.metadata["surface_output_classification"] == "open"

    closed_declared_body = make_surface_body(
        [make_surface_shell([PlanarSurfacePatch(family="planar")])],
        metadata={"kernel": {"surface_classification": "closed"}},
    )
    closed_declared_result = tessellate_surface_body(closed_declared_body, export_tessellation_request(require_watertight=False))
    assert closed_declared_result.classification == "open"

    with pytest.raises(ValueError, match="not closed-valid"):
        tessellate_surface_body(closed_declared_body, export_tessellation_request())


def test_seam_first_shell_tessellation_reuses_shared_boundary_vertices() -> None:
    body = _make_two_patch_open_shell_body()
    result = tessellate_surface_body(body, preview_tessellation_request())

    assert result.classification == "open"
    assert result.mesh.n_vertices == 6
    assert result.mesh.n_faces == 4
    assert result.analysis.boundary_edges > 0


def test_seam_first_shell_tessellation_reuses_shared_boundary_vertices_for_ruled_patches() -> None:
    patch_a = RuledSurfacePatch(
        family="ruled",
        start_curve=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        end_curve=((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
    )
    patch_b = RuledSurfacePatch(
        family="ruled",
        start_curve=((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        end_curve=((2.0, 0.0, 0.0), (2.0, 1.0, 0.0)),
    )
    shell = make_surface_shell(
        [patch_a, patch_b],
        seams=(
            SurfaceSeam(
                seam_id="mid",
                boundaries=(
                    SurfaceBoundaryRef(0, "right"),
                    SurfaceBoundaryRef(1, "left"),
                ),
            ),
        ),
    )
    body = make_surface_body([shell])

    result = tessellate_surface_body(body, preview_tessellation_request())

    assert result.classification == "open"
    assert result.mesh.n_vertices == 6
    assert result.mesh.n_faces == 4
    assert result.analysis.boundary_edges > 0


def test_closed_valid_body_tessellates_watertight_from_shell_truth() -> None:
    cube = _make_closed_cube_body()
    result = tessellate_surface_body(cube, export_tessellation_request())

    assert result.classification == "closed"
    assert result.analysis.is_watertight is True
    assert result.analysis.boundary_edges == 0
    assert result.analysis.nonmanifold_edges == 0
    assert result.analysis.degenerate_faces == 0


def test_surface_mesh_adapter_and_bridge_contract_are_explicit() -> None:
    body = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar")])])
    adapter = make_surface_mesh_adapter(export_tessellation_request(require_watertight=False))
    assert isinstance(adapter, SurfaceMeshAdapter)
    assert adapter.visibility == "internal"
    assert adapter.lossiness == "lossy"
    assert adapter.supported_consumers == ("preview", "export", "analysis")
    assert "Retire" in adapter.sunset_condition
    assert AdapterLossiness.__args__ == ("lossless", "lossy")

    result = adapter.convert(body)
    direct_mesh = mesh_from_surface_body(body, export_tessellation_request(require_watertight=False))
    assert result.mesh.n_faces == direct_mesh.n_faces
    assert result.request.intent == "export"


def test_surface_consumer_collection_preserves_order_and_identity() -> None:
    body_a = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 0.0))])])
    body_b = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar", origin=(2.0, 0.0, 0.0))])])
    collection = make_surface_consumer_collection([body_b, body_a], source_prefix="scene")

    assert isinstance(collection, SurfaceConsumerCollection)
    assert collection.items[0].source_id == "scene-0"
    assert collection.items[1].source_id == "scene-1"
    assert collection.body_identities == (body_b.stable_identity, body_a.stable_identity)


def test_surface_boolean_family_eligibility_supports_current_planar_pair_matrix() -> None:
    left = make_surface_box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0))
    right = make_surface_box(size=(1.0, 1.0, 1.0), center=(2.0, 0.0, 0.0))
    operands = prepare_surface_boolean_operands("union", (left, right))

    eligibility = surface_boolean_family_eligibility(operands)

    assert isinstance(eligibility, SurfaceBooleanFamilyEligibilityResult)
    assert eligibility.supported is True
    assert eligibility.diagnostics == ()
    assert eligibility.failure_reason is None
    assert ("union", "planar", "planar") in SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX
    assert all(isinstance(pair, SurfaceBooleanFamilyPairSupport) for pair in eligibility.family_pairs)


def test_surface_boolean_family_pair_matrix_declares_every_known_family_pair() -> None:
    expected_count = len(SURFACE_BOOLEAN_OPERATIONS) * len(PATCH_FAMILY_CAPABILITY_MATRIX) ** 2

    assert len(SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX) == expected_count
    assert len(surface_csg_completion_support_matrix()) == expected_count
    for operation in SURFACE_BOOLEAN_OPERATIONS:
        for left_family in PATCH_FAMILY_CAPABILITY_MATRIX:
            for right_family in PATCH_FAMILY_CAPABILITY_MATRIX:
                record = surface_boolean_family_pair_support(operation, left_family, right_family)

                assert isinstance(record, SurfaceBooleanFamilyPairSupport)
                assert record.operation == operation
                assert record.left_family == left_family
                assert record.right_family == right_family
                assert record.support_state in {"exact", "declared-tolerance", "adapter", "unsupported", "not-yet-implemented"}
                analytic_pair = left_family in ANALYTIC_SURFACE_CSG_FAMILIES and right_family in ANALYTIC_SURFACE_CSG_FAMILIES
                pair_families = {left_family, right_family}
                analytic_spline_pair = (
                    bool(pair_families & {"bspline", "nurbs"})
                    and any(family in ANALYTIC_SURFACE_CSG_FAMILIES for family in pair_families)
                    and pair_families <= (ANALYTIC_SURFACE_CSG_FAMILIES | {"bspline", "nurbs"})
                )
                spline_pair = pair_families <= {"bspline", "nurbs"}
                if analytic_pair:
                    assert record.supported is True
                    assert record.support_state == "exact"
                    assert record.required_future_capability is None
                elif analytic_spline_pair:
                    assert record.supported is True
                    assert record.support_state == "declared-tolerance"
                    assert record.required_future_capability is None
                elif spline_pair:
                    assert record.supported is True
                    assert record.support_state == "declared-tolerance"
                    assert record.required_future_capability is None
                else:
                    assert record.supported is False
                    assert record.required_future_capability


def test_surface_csg_route_taxonomy_classifies_higher_order_parametric_pairs() -> None:
    assert PARAMETRIC_HIGHER_ORDER_SURFACE_CSG_FAMILIES == frozenset(
        {"bspline", "nurbs", "sweep", "subdivision"}
    )
    assert classify_surface_csg_route_pair_class("planar", "ruled") == "low-order-analytic"
    assert classify_surface_csg_route_pair_class("planar", "bspline") == "analytic-to-bspline"
    assert classify_surface_csg_route_pair_class("revolution", "nurbs") == "analytic-to-nurbs"
    assert classify_surface_csg_route_pair_class("ruled", "sweep") == "analytic-to-sweep"
    assert classify_surface_csg_route_pair_class("planar", "subdivision") == "analytic-to-subdivision"
    assert classify_surface_csg_route_pair_class("bspline", "nurbs") == "spline-nurbs-pair"
    assert classify_surface_csg_route_pair_class("sweep", "nurbs") == "sweep-pair"
    assert classify_surface_csg_route_pair_class("subdivision", "bspline") == "subdivision-pair"
    assert classify_surface_csg_route_pair_class("heightmap", "bspline") == "sampled-boundary"
    assert classify_surface_csg_route_pair_class("planar", "unknown") == "unsupported-family"


def test_surface_csg_route_lookup_exposes_executable_state_separately_from_family_availability() -> None:
    analytic = surface_csg_route_lookup("union", "planar", "ruled")
    higher_order = surface_csg_route_lookup("union", "planar", "bspline")
    rational = surface_csg_route_lookup("union", "planar", "nurbs")

    assert isinstance(analytic, SurfaceCSGRouteRegistryRow)
    assert analytic.pair_class == "low-order-analytic"
    assert analytic.route_id == "surface-csg.low-order-analytic.exact"
    assert analytic.supported is True
    assert analytic.executable is True
    assert analytic.diagnostic is None

    assert isinstance(higher_order, SurfaceCSGRouteRegistryRow)
    assert higher_order.pair_class == "analytic-to-bspline"
    assert higher_order.supported is True
    assert higher_order.executable is True
    assert higher_order.support_state == "declared-tolerance"
    assert higher_order.diagnostic is None
    assert rational.pair_class == "analytic-to-nurbs"
    assert rational.supported is True
    assert rational.executable is True
    assert rational.support_state == "declared-tolerance"
    assert rational.diagnostic is None


def test_surface_csg_executable_row_report_flags_analytic_bspline_as_executable() -> None:
    report = surface_csg_executable_row_report(families=("planar", "bspline"))

    assert isinstance(report, SurfaceCSGExecutableRowReport)
    assert report.passed is True
    assert len(report.rows) == len(SURFACE_BOOLEAN_OPERATIONS) * 4
    assert any(row.executable and row.pair_class == "low-order-analytic" for row in report.rows)
    executable = [row for row in report.rows if row.pair_class == "analytic-to-bspline"]
    assert executable
    assert all(row.executable for row in executable)
    assert all(row.support_state == "declared-tolerance" for row in executable)
    payload = report.canonical_payload()
    assert payload["passed"] is True
    assert payload["diagnostics"] == []


def test_higher_order_csg_residual_collector_records_declared_tolerance_metadata() -> None:
    route = surface_csg_route_lookup("intersection", "planar", "bspline")

    residual = collect_higher_order_csg_residual(
        route,
        max_residual=5e-7,
        tolerance=1e-6,
        iteration_count=4,
        converged=True,
        patch_ids=("left:0", "right:2"),
    )

    assert isinstance(residual, SurfaceCSGResidualRecord)
    assert residual.within_tolerance is True
    assert residual.route_id == route.route_id
    assert residual.patch_ids == ("left:0", "right:2")
    assert residual.canonical_payload()["within_tolerance"] is True
    with pytest.raises(ValueError, match="positive and finite"):
        collect_higher_order_csg_residual(route, max_residual=0.0, tolerance=0.0, iteration_count=0, converged=True)


def test_higher_order_csg_degeneracy_classifier_reports_blocking_and_nonblocking_states() -> None:
    route = surface_csg_route_lookup("union", "sweep", "nurbs")
    residual = collect_higher_order_csg_residual(
        route,
        max_residual=2e-4,
        tolerance=1e-6,
        iteration_count=12,
        converged=False,
    )

    degeneracies = classify_higher_order_csg_degeneracies(
        residual,
        ambiguous=True,
        overlap=True,
        singularity=True,
        budget_exhausted=True,
        location="body:left patch:0 u=0.5",
    )

    assert all(isinstance(record, SurfaceCSGDegeneracyRecord) for record in degeneracies)
    assert {record.code for record in degeneracies} == {
        "non-convergence",
        "ambiguous-route",
        "overlap",
        "singularity",
        "budget-refusal",
    }
    overlap = next(record for record in degeneracies if record.code == "overlap")
    assert overlap.blocking is False
    assert "mesh" not in " ".join(record.message.lower() for record in degeneracies)


def test_higher_order_csg_route_diagnostic_formatter_preserves_ambiguity_as_blocking_payload() -> None:
    route = surface_csg_route_lookup("difference", "sweep", "subdivision")
    residual = collect_higher_order_csg_residual(
        route,
        max_residual=1e-2,
        tolerance=1e-6,
        iteration_count=8,
        converged=False,
    )
    degeneracies = classify_higher_order_csg_degeneracies(
        residual,
        ambiguous=True,
        location="route:sweep/subdivision station:0",
    )

    diagnostics = format_higher_order_csg_route_diagnostics(
        route,
        residual=residual,
        degeneracies=degeneracies,
    )

    assert diagnostics
    assert any(isinstance(diagnostic, SurfaceCSGRouteSupportDiagnostic) for diagnostic in diagnostics)
    ambiguity = next(diagnostic for diagnostic in diagnostics if isinstance(diagnostic, SurfaceCSGAmbiguityDiagnostic))
    assert ambiguity.blocking is True
    assert ambiguity.location == "route:sweep/subdivision station:0"
    assert "authored ambiguity" in ambiguity.message


def test_available_family_csg_classification_rows_cover_supported_and_refused_pairs() -> None:
    rows = available_family_csg_classification_rows()
    report = verify_available_family_csg_classification_rows()
    expected_count = len(SURFACE_BOOLEAN_OPERATIONS) * len(PATCH_FAMILY_CAPABILITY_MATRIX) ** 2

    assert report.passed is True
    assert len(rows) == expected_count
    assert any(row.left_family == "planar" and row.right_family == "planar" and row.classification == "supported-exact" for row in rows)
    assert any(row.left_family == "planar" and row.right_family == "bspline" and row.classification == "supported-declared" for row in rows)
    assert any(row.left_family == "planar" and row.right_family == "heightmap" and row.classification == "sampled-boundary-refusal" for row in rows)

    subset_report = verify_available_family_csg_classification_rows(families=("planar", "bspline"))
    assert subset_report.passed is True
    assert len(subset_report.rows) == len(SURFACE_BOOLEAN_OPERATIONS) * 4


def test_available_family_csg_no_mesh_fallback_evidence_is_diagnostic_for_unsupported_pairs() -> None:
    evidence = collect_surface_csg_no_mesh_fallback_evidence(families=("planar", "heightmap"))
    report = verify_surface_csg_no_mesh_fallback_evidence(families=("planar", "heightmap"))

    assert report.passed is True
    assert any(record.result_kind == "supported-surface" for record in evidence)
    sampled_refusals = [record for record in evidence if record.result_kind == "diagnostic-refusal"]
    assert sampled_refusals
    assert all(record.mesh_fallback_attempted is False for record in sampled_refusals)
    assert all("mesh fallback" not in record.message.lower() for record in sampled_refusals)


def test_surface_csg_refusal_record_is_structured_and_constant_time_policy() -> None:
    diagnostic = surface_csg_refusal_record("union", "planar", "heightmap")

    assert isinstance(diagnostic, SurfaceBooleanUnsupportedFamilyDiagnostic)
    payload = diagnostic.canonical_payload()
    assert payload["operation"] == "union"
    assert payload["left_family"] == "planar"
    assert payload["right_family"] == "heightmap"
    assert payload["required_future_capability"]


def test_surface_csg_analytic_primitive_pair_support_uses_declared_tolerances() -> None:
    record = surface_csg_analytic_primitive_pair_support(
        "intersection",
        "planar",
        "revolution",
        policy={"snap_tolerance": 1e-6, "equality_tolerance": 1e-6},
    )

    assert isinstance(record, SurfaceCSGPrimitiveAnalyticPairRecord)
    assert record.supported is True
    assert record.support_state == "exact"
    assert record.tolerance_policy.snap_tolerance == pytest.approx(1e-6)
    assert record.diagnostic == ""


def test_surface_csg_analytic_primitive_pair_refuses_higher_order_without_mesh() -> None:
    record = surface_csg_analytic_primitive_pair_support("difference", "planar", "implicit")
    heightmap_record = surface_csg_analytic_primitive_pair_support("union", "planar", "heightmap")
    displacement_record = surface_csg_analytic_primitive_pair_support("intersection", "planar", "displacement")

    assert record.supported is False
    assert record.support_state == "unsupported"
    assert "sampled-tessellation-boundary" in record.diagnostic
    assert "mesh" not in record.diagnostic.lower()
    assert heightmap_record.supported is False
    assert heightmap_record.support_state == "unsupported"
    assert "sampled-tessellation-boundary" in heightmap_record.diagnostic
    assert "mesh" not in heightmap_record.diagnostic.lower()
    assert displacement_record.supported is False
    assert displacement_record.support_state == "unsupported"
    assert "sampled-tessellation-boundary" in displacement_record.diagnostic
    assert "mesh" not in displacement_record.diagnostic.lower()


def test_surface_boolean_family_eligibility_reports_unsupported_mixed_family_without_mesh_fallback() -> None:
    box = make_surface_box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0))
    heightmap_body = make_surface_body([make_surface_shell([HeightmapSurfacePatch(family="heightmap")])])
    operands = SurfaceBooleanOperands(operation="union", bodies=(box, heightmap_body))

    eligibility = surface_boolean_family_eligibility(operands)
    result = surface_boolean_result("union", operands)

    assert eligibility.supported is False
    assert eligibility.required_future_capabilities
    assert all(isinstance(diagnostic, SurfaceBooleanUnsupportedFamilyDiagnostic) for diagnostic in eligibility.diagnostics)
    assert result.status == "unsupported"
    assert result.body is None
    assert result.failure_reason is not None
    assert "unsupported surface boolean family pair" in result.failure_reason
    assert "heightmap" in result.failure_reason


def test_surface_boolean_unsupported_family_diagnostic_builder_refuses_supported_pair() -> None:
    supported = SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX[("union", "planar", "planar")]

    with pytest.raises(ValueError, match="Supported surface boolean family pairs"):
        build_surface_boolean_unsupported_family_diagnostic(supported)


def test_higher_order_csg_solver_boundary_names_advanced_family_refusals() -> None:
    advanced_families = tuple(family for family in HIGHER_ORDER_SURFACE_CSG_FAMILIES if family not in {"bspline", "nurbs"})

    for family in advanced_families:
        support = classify_higher_order_csg_pair("intersection", "planar", family)
        diagnostic = build_higher_order_csg_refusal_diagnostic(support)

        assert isinstance(support, SurfaceCSGHigherOrderSupportRecord)
        assert isinstance(diagnostic, SurfaceCSGHigherOrderRefusalDiagnostic)
        assert support.supported is False
        expected_boundary = "sampled-tessellation-boundary" if family in SAMPLED_SURFACE_CSG_FAMILIES else "higher-order-exact-solver"
        assert support.solver_boundary == expected_boundary
        assert family in diagnostic.message
        assert "requires" in diagnostic.message


def test_higher_order_csg_refusal_is_reflected_in_family_diagnostics() -> None:
    support = surface_boolean_family_pair_support("union", "planar", "sweep")

    diagnostic = build_surface_boolean_unsupported_family_diagnostic(support)

    assert diagnostic.phase == "higher-order-exact-solver"
    assert "unsupported higher-order surface boolean pair" in diagnostic.required_future_capability
    assert "planar/sweep" in diagnostic.message


def test_surface_backend_boolean_api_uses_family_diagnostic_result_for_unsupported_pairs() -> None:
    box = make_surface_box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0))
    sweep_body = make_surface_body([make_surface_shell([SweepSurfacePatch(family="sweep")])])

    result = boolean_union((box, sweep_body), backend="surface")

    assert result.status == "unsupported"
    assert result.failure_reason is not None
    assert "mesh" not in result.failure_reason.lower()


def test_surface_boolean_family_refusal_gate_never_invokes_mesh_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_mesh_boolean(*args: object, **kwargs: object) -> None:
        raise AssertionError("surface CSG family refusal must not reach mesh boolean execution")

    monkeypatch.setattr(csg_module, "_apply_boolean", fail_mesh_boolean)
    box = make_surface_box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0))
    sweep_body = make_surface_body([make_surface_shell([SweepSurfacePatch(family="sweep")])])

    result = boolean_union((box, sweep_body), backend="surface")

    assert result.status == "unsupported"
    assert result.body is None
    assert result.failure_reason is not None
    assert "mesh" not in result.failure_reason.lower()


def test_cross_mode_drift_report_stays_on_same_surface_truth() -> None:
    body = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar")])])
    report = compare_tessellation_modes(body, preview_tessellation_request(), analysis_tessellation_request(require_watertight=False))

    assert report.baseline_body_identity == body.stable_identity
    assert report.comparison_body_identity == body.stable_identity
    assert report.bounds_max_delta == pytest.approx(0.0)
    assert report.classification_changed is False
    assert report.watertightness_changed is False
    assert report.within_default_bounds is True


def test_surface_scene_handoff_preserves_order_without_tessellating() -> None:
    body_a = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar", origin=(0.0, 0.0, 0.0))])])
    body_b = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar", origin=(1.0, 0.0, 0.0))])])
    translate = np.eye(4)
    translate[:3, 3] = [5.0, 0.0, 0.0]

    node_a = make_surface_scene_node("node-a", body_a, metadata={"label": "A"})
    node_b = make_surface_scene_node("node-b", body_b, transform_matrix=translate, metadata={"label": "B"})
    hidden = make_surface_scene_node("node-hidden", body_a, visible=False)
    root = make_surface_scene_group("root", [node_a, make_surface_scene_group("nested", [hidden, node_b])])

    assert isinstance(root, SurfaceSceneGroup)
    assert isinstance(node_a, SurfaceSceneNode)

    flattened = flatten_surface_scene(root, collection_metadata={"scene": "demo"})
    handed_off = handoff_surface_scene(root, collection_metadata={"scene": "demo"})

    assert flattened.metadata == {"scene": "demo"}
    assert handed_off.body_identities == flattened.body_identities
    assert [item.source_id for item in flattened.items] == ["node-a", "node-b"]
    assert flattened.items[0].metadata["label"] == "A"
    assert flattened.items[1].metadata["label"] == "B"
    assert np.allclose(flattened.items[1].body.bounds_estimate(), (6.0, 7.0, 0.0, 1.0, 0.0, 0.0))


def test_feature_surface_handoff_validator_accepts_surface_truth_only() -> None:
    body = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar")])])
    collection = make_surface_consumer_collection([body], source_prefix="feature")

    body_record = validate_feature_surface_handoff("feature.body", body)
    collection_record = validate_feature_surface_handoff("feature.collection", collection)
    diagnostic = feature_surface_handoff_diagnostic("feature.mesh", make_box_mesh())

    assert isinstance(body_record, FeatureSurfaceHandoffRecord)
    assert body_record.output_type == "SurfaceBody"
    assert body_record.body_identities == (body.stable_identity,)
    assert collection_record.output_type == "SurfaceConsumerCollection"
    assert collection_record.body_identities == (body.stable_identity,)
    assert isinstance(diagnostic, FeatureSurfaceHandoffDiagnostic)
    assert diagnostic.explicit_mesh_api_required is True
    assert "SurfaceBody, SurfaceConsumerCollection" in diagnostic.message
    with pytest.raises(TypeError, match="explicitly mesh-named API"):
        validate_feature_surface_handoff("feature.mesh", make_box_mesh())


def test_surface_composition_public_type_groups_surface_bodies_without_tessellating() -> None:
    body_a = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar")])])
    body_b = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar", origin=(2.0, 0.0, 0.0))])])
    translate = np.eye(4)
    translate[:3, 3] = [4.0, 0.0, 0.0]

    nested = make_surface_composition([body_b], composition_id="nested")
    composition = surface_group(
        [body_a, nested],
        group_id="authored",
        transform_matrix=translate,
        metadata={"purpose": "public-grouping"},
    )

    assert isinstance(composition, SurfaceComposition)
    assert composition.composition_id == "authored"
    assert composition.iter_children() == (body_a, nested)
    assert composition.metadata == {"purpose": "public-grouping"}
    assert np.allclose(composition.transform_matrix, translate)
    assert composition.stable_identity == surface_group(
        [body_a, nested],
        group_id="authored",
        transform_matrix=translate,
        metadata={"purpose": "public-grouping"},
    ).stable_identity


def test_surface_composition_transform_attachment_returns_new_composition() -> None:
    body = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar")])])
    composition = make_surface_composition([body], composition_id="move-me")
    translate = np.eye(4)
    translate[:3, 3] = [1.0, 2.0, 3.0]

    moved = composition.with_transform(translate)

    assert moved is not composition
    assert isinstance(moved, SurfaceComposition)
    assert np.allclose(composition.transform_matrix, np.eye(4))
    assert np.allclose(moved.transform_matrix, translate)
    assert moved.stable_identity != composition.stable_identity


def test_surface_composition_rejects_mesh_inputs_with_diagnostic() -> None:
    mesh = make_box_mesh(size=(1.0, 1.0, 1.0))

    with pytest.raises(SurfaceCompositionError) as exc:
        make_surface_composition([mesh], composition_id="bad")

    diagnostic = exc.value.diagnostic
    assert diagnostic.boundary == "surface-composition"
    assert diagnostic.target_type == "Mesh"
    assert diagnostic.child_index == 0
    assert "compatibility boundary" in diagnostic.reason


def test_surface_composition_traversal_is_preorder_and_transform_aware() -> None:
    body_a = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar")])])
    body_b = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar", origin=(2.0, 0.0, 0.0))])])
    root_translate = np.eye(4)
    root_translate[:3, 3] = [10.0, 0.0, 0.0]
    nested_translate = np.eye(4)
    nested_translate[:3, 3] = [0.0, 5.0, 0.0]

    nested = make_surface_composition([body_b], composition_id="nested", transform_matrix=nested_translate)
    root = surface_group([body_a, nested], group_id="root", transform_matrix=root_translate)

    records = traverse_surface_composition(root)
    collection = surface_composition_to_consumer_collection(root)

    assert all(isinstance(record, SurfaceCompositionTraversalRecord) for record in records)
    assert [record.source_id for record in records] == ["root/0", "root/1/nested/0"]
    assert [item.source_id for item in collection.items] == ["root/0", "root/1/nested/0"]
    assert collection.items[1].metadata["traversal"]["order"] == 1
    assert np.allclose(collection.items[0].body.bounds_estimate(), (10.0, 11.0, 0.0, 1.0, 0.0, 0.0))
    assert np.allclose(collection.items[1].body.bounds_estimate(), (12.0, 13.0, 5.0, 6.0, 0.0, 0.0))
    assert collection.metadata["surface_composition_traversal"]["traversal_order"] == (
        "root/0",
        "root/1/nested/0",
    )


def test_surface_composition_traversal_preserves_all_patch_families_without_tessellation() -> None:
    body = _make_all_patch_family_body()
    composition = surface_group([body], group_id="all-families")

    records = traverse_surface_composition(composition)
    collection = surface_composition_to_consumer_collection(composition)

    assert len(records) == 1
    assert isinstance(collection, SurfaceConsumerCollection)
    assert [item.source_id for item in collection.items] == ["all-families/0"]
    assert not isinstance(collection.items[0].body, Mesh)
    assert collection.items[0].body.stable_identity == body.stable_identity
    assert [patch.family for patch in collection.items[0].body.iter_patches(world=False)] == [
        "planar",
        "ruled",
        "revolution",
        "bspline",
        "nurbs",
        "sweep",
        "subdivision",
        "implicit",
        "heightmap",
        "displacement",
    ]
    assert collection.metadata["surface_composition_traversal"]["traversal_order"] == ("all-families/0",)


def test_surface_composition_transform_preserves_all_family_payload_identity_without_tessellation() -> None:
    body = _make_all_patch_family_body()
    translate = np.eye(4)
    translate[:3, 3] = [3.0, 4.0, 5.0]
    composition = surface_group([body], group_id="moved-families", transform_matrix=translate)

    records = traverse_surface_composition(composition)
    collection = surface_composition_to_consumer_collection(composition)
    transformed_body = collection.items[0].body

    assert not isinstance(transformed_body, Mesh)
    assert np.allclose(records[0].transform_matrix, translate)
    assert np.allclose(transformed_body.transform_matrix, translate)
    assert np.allclose(body.transform_matrix, np.eye(4))
    assert [patch.family for patch in transformed_body.shells[0].patches] == [
        patch.family for patch in body.shells[0].patches
    ]
    assert [patch.stable_identity for patch in transformed_body.shells[0].patches] == [
        patch.stable_identity for patch in body.shells[0].patches
    ]
    world_planar = transformed_body.iter_patches(world=True)[0]
    assert np.allclose(world_planar.point_at(0.0, 0.0), np.array([3.0, 4.0, 5.0]))
    assert collection.items[0].metadata["traversal"]["source_id"] == "moved-families/0"


def test_surface_composition_tessellation_handoff_is_explicit_boundary() -> None:
    body_a = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar")])])
    body_b = make_surface_body([make_surface_shell([PlanarSurfacePatch(family="planar", origin=(2.0, 0.0, 0.0))])])
    composition = surface_group([body_a, body_b], group_id="render-me")

    collection = surface_composition_to_consumer_collection(composition)
    direct = tessellate_surface_consumer_collection(collection, preview_tessellation_request())
    from_composition = tessellate_surface_composition(composition, preview_tessellation_request())

    assert isinstance(direct, SurfaceCollectionTessellationResult)
    assert isinstance(from_composition, SurfaceCollectionTessellationResult)
    assert direct.body_identities == collection.body_identities
    assert from_composition.body_identities == collection.body_identities
    assert direct.mesh.vertices.shape[0] == from_composition.mesh.vertices.shape[0]
    assert direct.mesh.metadata["surface_collection_body_identities"] == collection.body_identities
