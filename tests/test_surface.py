from __future__ import annotations

import numpy as np
import pytest

import impression.modeling as modeling
from impression.mesh import Mesh
from impression.modeling import (
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
from impression.modeling._surface_ops import make_surface_linear_extrude, make_surface_rotate_extrude
from impression.modeling import (
    PATCH_FAMILY_CAPABILITY_MATRIX,
    PATCH_FAMILY_FEATURE_COVERAGE,
    REQUIRED_V1_PATCH_FAMILIES,
    SUPPORTED_SURFACE_PATCH_FAMILIES,
    SURFACE_SPEC_66_RETIREMENT_NOTE,
    AdapterLossiness,
    analysis_tessellation_request,
    BSplineSurfacePatch,
    boolean_union,
    build_surface_boolean_unsupported_family_diagnostic,
    compare_tessellation_modes,
    DisplacementSurfacePatch,
    export_tessellation_request,
    flatten_surface_scene,
    handoff_surface_scene,
    HeightmapSurfacePatch,
    IMPLICIT_FIELD_NODE_KINDS,
    ImplicitApproximationMetadata,
    ImplicitFieldEvaluationDomain,
    ImplicitFieldEvaluationResult,
    ImplicitFieldNode,
    ImplicitFieldSafetyPolicy,
    ImplicitFieldValidationDiagnostic,
    ImplicitSurfacePatch,
    ImplicitTessellationBoundsDiagnostic,
    make_surface_consumer_collection,
    make_surface_composition,
    make_surface_mesh_adapter,
    make_surface_scene_group,
    make_surface_scene_node,
    make_surface_to_mesh_adapter_record,
    mesh_from_surface_body,
    NURBSSurfacePatch,
    normalize_tessellation_request,
    ParameterDomain,
    Path3D,
    PlanarSurfacePatch,
    preview_tessellation_request,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SurfaceAdjacencyRecord,
    SurfaceBody,
    SurfaceBoundaryDescriptor,
    SurfaceBoundaryRef,
    SurfaceComposition,
    SurfaceCompositionError,
    SurfaceCompositionTraversalRecord,
    SurfaceContinuityMetadata,
    SurfaceConsumerCollection,
    SurfaceCollectionTessellationResult,
    SurfaceFamilyTessellationAdapter,
    SurfaceFamilyTessellationAdapterCoverageRecord,
    SURFACE_FAMILY_TESSELLATION_ADAPTERS,
    SurfaceBooleanFamilyEligibilityResult,
    SurfaceBooleanFamilyPairSupport,
    SurfaceBooleanUnsupportedFamilyDiagnostic,
    SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX,
    SurfaceSeamParticipationRecord,
    SurfaceSeamValidationResult,
    SubdivisionCrease,
    SubdivisionRefinementResult,
    SubdivisionSurfacePatch,
    SurfaceMeshAdapter,
    SurfaceSceneGroup,
    SurfaceSceneNode,
    SurfaceSeam,
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
    classify_surface_seam_continuity,
    evaluate_implicit_field,
    evaluate_implicit_field_domain,
    extract_surface_boundary_descriptor,
    inspect_surface_family_tessellation_adapter_coverage,
    make_surface_body,
    make_surface_shell,
    make_implicit_surface,
    make_subdivision_surface,
    make_implicit_field_node,
    prepare_surface_boolean_operands,
    assess_implicit_field_security,
    refine_subdivision_control_cage,
    surface_adjacency_from_seams,
    surface_boolean_family_eligibility,
    surface_boolean_result,
    surface_composition_to_consumer_collection,
    surface_group,
    validate_implicit_field_security,
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
    assert PATCH_FAMILY_CAPABILITY_MATRIX["nurbs"].support_phase == "planned"
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


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"family": "bspline"}, "family must be 'nurbs'"),
        ({"family": "nurbs", "weights": [[1.0, 1.0]]}, "weights must match"),
        ({"family": "nurbs", "weights": [[1.0, 0.0], [1.0, 1.0]]}, "finite and positive"),
        ({"family": "nurbs", "weights": [[1.0, float("nan")], [1.0, 1.0]]}, "finite and positive"),
    ],
)
def test_nurbs_surface_patch_rejects_invalid_weight_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NURBSSurfacePatch(**kwargs)


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


def test_subdivision_boundary_descriptor_records_approximation_and_implicit_seams_refuse() -> None:
    descriptor = extract_surface_boundary_descriptor(SubdivisionSurfacePatch(family="subdivision"), "left")

    assert descriptor.exact is False
    assert descriptor.approximation_metadata["method"] == "finite_subdivision_boundary"

    with pytest.raises(ValueError, match="cannot participate in parametric seams"):
        SurfaceShell(
            patches=(ImplicitSurfacePatch(family="implicit"), PlanarSurfacePatch(family="planar")),
            seams=(SurfaceSeam("implicit-planar", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left"))),),
        )


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


def test_surface_boolean_family_eligibility_reports_unsupported_mixed_family_without_mesh_fallback() -> None:
    box = make_surface_box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0))
    sphere = make_surface_sphere(radius=0.5, center=(2.0, 0.0, 0.0))
    operands = prepare_surface_boolean_operands("union", (box, sphere))

    eligibility = surface_boolean_family_eligibility(operands)
    result = surface_boolean_result("union", operands)

    assert eligibility.supported is False
    assert eligibility.required_future_capabilities
    assert all(isinstance(diagnostic, SurfaceBooleanUnsupportedFamilyDiagnostic) for diagnostic in eligibility.diagnostics)
    assert result.status == "unsupported"
    assert result.body is None
    assert result.failure_reason is not None
    assert "unsupported surface boolean family pair" in result.failure_reason
    assert "revolution" in result.failure_reason


def test_surface_boolean_unsupported_family_diagnostic_builder_refuses_supported_pair() -> None:
    supported = SURFACE_BOOLEAN_FAMILY_PAIR_SUPPORT_MATRIX[("union", "planar", "planar")]

    with pytest.raises(ValueError, match="Supported surface boolean family pairs"):
        build_surface_boolean_unsupported_family_diagnostic(supported)


def test_surface_backend_boolean_api_uses_family_diagnostic_result_for_unsupported_pairs() -> None:
    box = make_surface_box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0))
    sphere = make_surface_sphere(radius=0.5, center=(2.0, 0.0, 0.0))

    result = boolean_union((box, sphere), backend="surface")

    assert result.status == "unsupported"
    assert result.failure_reason is not None
    assert "operand-family-eligibility" in result.failure_reason


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
