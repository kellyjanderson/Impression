from __future__ import annotations

import numpy as np
import pytest

import impression.modeling as modeling
from impression.modeling import (
    make_box,
    make_cone,
    make_cylinder,
    make_ngon,
    make_nhedron,
    make_polyhedron,
    make_prism,
    make_rect,
    make_sphere,
    make_torus,
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
    DEFERRED_V1_PATCH_FAMILIES,
    PATCH_FAMILY_FEATURE_COVERAGE,
    REQUIRED_V1_PATCH_FAMILIES,
    AdapterLossiness,
    analysis_tessellation_request,
    compare_tessellation_modes,
    export_tessellation_request,
    flatten_surface_scene,
    handoff_surface_scene,
    make_surface_consumer_collection,
    make_surface_mesh_adapter,
    make_surface_scene_group,
    make_surface_scene_node,
    mesh_from_surface_body,
    normalize_tessellation_request,
    ParameterDomain,
    PlanarSurfacePatch,
    preview_tessellation_request,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SurfaceAdjacencyRecord,
    SurfaceBody,
    SurfaceBoundaryRef,
    SurfaceConsumerCollection,
    SurfaceMeshAdapter,
    SurfaceSceneGroup,
    SurfaceSceneNode,
    SurfaceSeam,
    SurfaceShell,
    TessellationRequest,
    TrimLoop,
    make_surface_body,
    make_surface_shell,
    tessellate_surface_body,
    tessellate_surface_patch,
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


def test_internal_surface_box_builder_is_closed_and_public_box_stays_mesh() -> None:
    surface_box = make_surface_box(size=(2.0, 4.0, 6.0), center=(1.0, 2.0, 3.0))
    mesh_box = make_box(size=(2.0, 4.0, 6.0), center=(1.0, 2.0, 3.0))

    assert isinstance(surface_box, SurfaceBody)
    assert surface_box.shell_count == 1
    assert surface_box.patch_count == 6
    assert surface_box.bounds_estimate() == (0.0, 2.0, 0.0, 4.0, 0.0, 6.0)
    assert mesh_box.n_faces > 0

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


def test_internal_surface_cylinder_returns_surface_body_and_public_api_stays_mesh() -> None:
    surface_cylinder = make_surface_cylinder(radius=1.0, height=2.0, resolution=32)
    mesh_cylinder = make_cylinder(radius=1.0, height=2.0, resolution=32)

    assert isinstance(surface_cylinder, SurfaceBody)
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


def test_internal_surface_cone_returns_surface_body_and_public_api_stays_mesh() -> None:
    surface_cone = make_surface_cone(bottom_diameter=2.0, top_diameter=0.0, height=2.0, resolution=32)
    mesh_cone = make_cone(bottom_diameter=2.0, top_diameter=0.0, height=2.0, resolution=32)

    assert isinstance(surface_cone, SurfaceBody)
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


def test_internal_surface_ngon_returns_surface_body_and_public_api_stays_mesh() -> None:
    surface_ngon = make_surface_ngon(sides=6, radius=1.0, height=2.0)
    mesh_ngon = make_ngon(sides=6, radius=1.0, height=2.0)

    assert isinstance(surface_ngon, SurfaceBody)
    assert surface_ngon.shell_count == 1
    assert surface_ngon.patch_count == 8
    assert [patch.family for patch in surface_ngon.iter_patches(world=False)].count("ruled") == 6
    assert [patch.family for patch in surface_ngon.iter_patches(world=False)].count("planar") == 2
    assert mesh_ngon.n_faces > 0

    result = tessellate_surface_body(surface_ngon, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_internal_surface_prism_returns_surface_body_and_public_api_stays_mesh() -> None:
    surface_prism = make_surface_prism(base_size=(2.0, 1.0), top_size=(1.0, 0.5), height=2.0)
    mesh_prism = make_prism(base_size=(2.0, 1.0), top_size=(1.0, 0.5), height=2.0)

    assert isinstance(surface_prism, SurfaceBody)
    assert surface_prism.shell_count == 1
    assert surface_prism.patch_count == 6
    assert [patch.family for patch in surface_prism.iter_patches(world=False)].count("ruled") == 4
    assert [patch.family for patch in surface_prism.iter_patches(world=False)].count("planar") == 2
    assert mesh_prism.n_faces > 0

    result = tessellate_surface_body(surface_prism, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_internal_surface_torus_returns_closed_surface_body_and_public_api_stays_mesh() -> None:
    surface_torus = make_surface_torus(major_radius=2.0, minor_radius=0.5, n_theta=32, n_phi=16)
    mesh_torus = make_torus(major_radius=2.0, minor_radius=0.5, n_theta=32, n_phi=16)

    assert isinstance(surface_torus, SurfaceBody)
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


def test_internal_surface_sphere_returns_closed_surface_body_and_public_api_stays_mesh() -> None:
    surface_sphere = make_surface_sphere(radius=1.0, center=(0.0, 0.0, 0.0), theta_resolution=32, phi_resolution=16)
    mesh_sphere = make_sphere(radius=1.0, center=(0.0, 0.0, 0.0), theta_resolution=32, phi_resolution=16)

    assert isinstance(surface_sphere, SurfaceBody)
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


def test_internal_surface_polyhedron_returns_surface_body_and_public_api_stays_mesh() -> None:
    surface_polyhedron = make_surface_polyhedron(faces=6, radius=1.0)
    mesh_polyhedron = make_polyhedron(faces=6, radius=1.0)

    assert isinstance(surface_polyhedron, SurfaceBody)
    assert surface_polyhedron.shell_count == 1
    assert surface_polyhedron.patch_count == 6
    assert [patch.family for patch in surface_polyhedron.iter_patches(world=False)].count("planar") == 6
    assert mesh_polyhedron.n_faces > 0

    result = tessellate_surface_body(surface_polyhedron, export_tessellation_request(require_watertight=False))
    assert result.classification == "open"
    assert result.mesh.n_faces > 0


def test_internal_surface_nhedron_wraps_polyhedron_and_public_api_stays_mesh() -> None:
    surface_nhedron = make_surface_nhedron(faces=8, radius=1.0)
    mesh_nhedron = make_nhedron(faces=8, radius=1.0)

    assert isinstance(surface_nhedron, SurfaceBody)
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
    assert "nurbs" in DEFERRED_V1_PATCH_FAMILIES
    assert PATCH_FAMILY_FEATURE_COVERAGE["planar"] == ("caps", "planar-primitives", "trimmed-faces")


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
