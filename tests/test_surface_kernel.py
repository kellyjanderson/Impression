from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from impression.modeling import (
    ParameterDomain,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SurfaceAdjacencyRecord,
    SurfaceBoundaryRef,
    SurfacePatch,
    SurfaceSeam,
    SurfaceShell,
    TessellationRequest,
    TrimLoop,
    make_surface_body,
    make_surface_shell,
    normalize_tessellation_request,
    tessellate_surface_body,
)
from impression.modeling._surface_ops import (
    _as_vec3 as _ops_as_vec3,
    _attached_transform as _ops_attached_transform,
    _closed_loop_points,
    _domain_for_region,
    _normalize as _ops_normalize,
    _rotation_transform,
    make_surface_linear_extrude,
    make_surface_rotate_extrude,
)
from impression.modeling._surface_primitives import (
    _planar_patch_from_face,
    make_surface_box,
    make_surface_cone,
    make_surface_cylinder,
    make_surface_polyhedron,
    make_surface_prism,
    make_surface_sphere,
    make_surface_torus,
)
from impression.modeling.drawing2d import PlanarShape2D, make_rect
from impression.modeling.topology import Loop, Region, Section
from impression.modeling.surface import (
    _as_matrix4,
    _as_points3,
    _as_vec2,
    _as_vec3,
    _canonicalize,
    _ensure_loop_orientation,
    _normalize_axis,
    _normalize_loop_points_2d,
    _polyline_point_at,
    _signed_area_2d,
    _split_metadata,
    _transform_bounds,
)
from impression.modeling.tessellation import (
    _boundary_is_collapsed,
    _boundary_samples,
    _clamp_patch_parameters,
    _patch_requires_shell_grid_tessellation,
    _rectangular_boundary_indices,
    _rectangular_grid_counts,
    _rectangular_grid_uv_mesh_data,
    _seam_vertex_assignments,
    _shell_grid_counts_for_patch,
    _validate_positive_or_none,
)


@dataclass(frozen=True)
class _DegeneratePatch(SurfacePatch):
    def point_at(self, u: float, v: float) -> np.ndarray:
        self.validate_parameters(u, v)
        return np.array([float(u), float(v), 0.0], dtype=float)

    def derivatives_at(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        self.validate_parameters(u, v)
        return (np.zeros(3, dtype=float), np.zeros(3, dtype=float))

    def geometry_payload(self) -> dict[str, object]:
        return {"mode": "degenerate"}


def test_surface_boundary_and_seam_records_validate_open_and_shared_forms() -> None:
    open_ref = SurfaceBoundaryRef(0, "left")
    adjacency = SurfaceAdjacencyRecord(source=open_ref, target=None)
    open_seam = SurfaceSeam("open-left", (open_ref,))

    assert adjacency.is_open is True
    assert open_seam.is_open is True
    assert open_seam.canonical_payload()["boundaries"][0]["boundary_id"] == "left"

    with pytest.raises(ValueError, match="must be >= 0"):
        SurfaceBoundaryRef(-1, "left")

    with pytest.raises(ValueError, match="must be non-empty"):
        SurfaceBoundaryRef(0, "   ")

    with pytest.raises(ValueError, match="must be non-empty"):
        SurfaceAdjacencyRecord(source=open_ref, target=None, continuity="   ")

    with pytest.raises(ValueError, match="must be non-empty"):
        SurfaceAdjacencyRecord(source=open_ref, target=None, seam_id="   ")

    with pytest.raises(ValueError, match="must be unique"):
        SurfaceSeam("dup", (open_ref, open_ref))

    with pytest.raises(ValueError, match="must be non-empty"):
        SurfaceSeam("   ", (open_ref,))

    with pytest.raises(ValueError, match="one open boundary or two shared boundaries"):
        SurfaceSeam("bad-count", ())

    with pytest.raises(ValueError, match="one open boundary or two shared boundaries"):
        SurfaceSeam("bad-count", (open_ref, SurfaceBoundaryRef(1, "left"), SurfaceBoundaryRef(2, "left")))

    with pytest.raises(ValueError, match="continuity must be non-empty"):
        SurfaceSeam("bad-continuity", (open_ref,), continuity="   ")


def test_surface_shell_validates_duplicate_seams_and_out_of_range_patch_refs() -> None:
    patch = PlanarSurfacePatch(family="planar")
    seam = SurfaceSeam("s0", (SurfaceBoundaryRef(0, "left"),))

    with pytest.raises(ValueError, match="seam IDs must be unique"):
        SurfaceShell(patches=(patch,), seams=(seam, seam))

    with pytest.raises(ValueError, match="outside the shell"):
        SurfaceShell(
            patches=(patch,),
            seams=(SurfaceSeam("bad", (SurfaceBoundaryRef(1, "left"),)),),
        )

    with pytest.raises(ValueError, match="target references a patch index outside the shell"):
        SurfaceShell(
            patches=(patch,),
            adjacency=(
                SurfaceAdjacencyRecord(
                    source=SurfaceBoundaryRef(0, "left"),
                    target=SurfaceBoundaryRef(2, "right"),
                ),
            ),
        )

    with pytest.raises(ValueError, match="source references a patch index outside the shell"):
        SurfaceShell(
            patches=(patch,),
            adjacency=(
                SurfaceAdjacencyRecord(
                    source=SurfaceBoundaryRef(3, "left"),
                    target=None,
                ),
            ),
        )


def test_surface_patch_sample_grid_and_normal_fail_cleanly_for_degenerate_patch() -> None:
    patch = _DegeneratePatch(family="degenerate", domain=ParameterDomain((0.0, 1.0), (0.0, 1.0)))

    with pytest.raises(ValueError, match="at least 2 samples on each axis"):
        patch.sample_grid(1, 2)

    with pytest.raises(ValueError, match="normal is undefined"):
        patch.normal_at(0.5, 0.5)


def test_surface_identity_world_iteration_and_metadata_helpers_are_stable() -> None:
    patch = PlanarSurfacePatch(family="planar", metadata={"preview_label": "panel"})
    shell = make_surface_shell([patch], metadata=None)
    body = make_surface_body([shell], metadata=None)

    assert shell.iter_patches(world=True) == shell.patches
    assert body.iter_shells(world=True) == body.shells
    assert patch.kernel_metadata() == {"preview_label": "panel"}
    assert patch.consumer_metadata() == {}
    assert shell.kernel_metadata() == {}
    assert body.consumer_metadata() == {}
    assert body.cache_key == body.stable_identity


def test_trim_boundary_seams_validate_boundary_ids_during_tessellation() -> None:
    patch_a = PlanarSurfacePatch(
        family="planar",
        trim_loops=(
            TrimLoop([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], category="outer"),
            TrimLoop([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)], category="inner"),
        ),
    )
    patch_b = PlanarSurfacePatch(
        family="planar",
        origin=(2.0, 0.0, 0.0),
        trim_loops=(
            TrimLoop([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], category="outer"),
        ),
    )

    invalid_index_body = make_surface_body(
        [
            make_surface_shell(
                [patch_a, patch_b],
                seams=(
                    SurfaceSeam(
                        "bad-inner",
                        (SurfaceBoundaryRef(0, "trim:inner:9"), SurfaceBoundaryRef(1, "trim:outer")),
                    ),
                ),
            )
        ]
    )
    invalid_parse_body = make_surface_body(
        [
            make_surface_shell(
                [patch_a, patch_b],
                seams=(
                    SurfaceSeam(
                        "bad-parse",
                        (SurfaceBoundaryRef(0, "trim:inner:not-an-int"), SurfaceBoundaryRef(1, "trim:outer")),
                    ),
                ),
            )
        ]
    )

    request = normalize_tessellation_request(TessellationRequest(intent="preview"))

    with pytest.raises(ValueError, match="out of range"):
        tessellate_surface_body(invalid_index_body, request)

    with pytest.raises(ValueError, match="Invalid trim boundary_id"):
        tessellate_surface_body(invalid_parse_body, request)


def test_surface_helper_normalizers_and_bounds_cover_low_level_branches() -> None:
    assert np.allclose(_as_vec2((1.0, 2.0), name="uv"), np.array([1.0, 2.0]))
    assert np.allclose(_as_vec3((1.0, 2.0, 3.0), name="xyz"), np.array([1.0, 2.0, 3.0]))
    assert np.allclose(
        _as_points3(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), name="points"),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="must contain only finite values"):
        _as_vec2((1.0, np.inf), name="uv")

    with pytest.raises(ValueError, match="must contain only finite values"):
        _as_vec3((1.0, 2.0, np.nan), name="xyz")

    with pytest.raises(ValueError, match="at least two 3D points"):
        _as_points3(((0.0, 0.0, 0.0),), name="points")

    with pytest.raises(ValueError, match="must contain only finite values"):
        _as_points3(((0.0, 0.0, 0.0), (np.inf, 0.0, 0.0)), name="points")

    with pytest.raises(ValueError, match="must contain only finite values"):
        _as_matrix4(np.full((4, 4), np.nan), name="matrix")

    with pytest.raises(ValueError, match="must be non-zero"):
        _normalize_axis((0.0, 0.0, 0.0), name="axis")

    with pytest.raises(ValueError, match="ranges must be finite"):
        ParameterDomain((0.0, float("inf")), (0.0, 1.0))

    assert _normalize_loop_points_2d([], name="loop").shape == (0, 2)
    assert _normalize_loop_points_2d([(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)], name="loop").shape == (2, 2)
    assert _ensure_loop_orientation(np.array([(0.0, 0.0), (1.0, 0.0)]), clockwise=False).shape == (2, 2)
    assert _signed_area_2d(np.array([(0.0, 0.0), (1.0, 0.0)])) == 0.0

    with pytest.raises(ValueError, match="must contain only finite values"):
        _normalize_loop_points_2d([(0.0, 0.0), (np.nan, 0.0)], name="loop")

    translated = _transform_bounds((0.0, 1.0, 0.0, 2.0, 0.0, 3.0), np.array(
        [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ))
    assert translated == (2.0, 3.0, -1.0, 1.0, 4.0, 7.0)
    assert _canonicalize(np.float64(3.5)) == 3.5
    assert _canonicalize({3, 1, 2}) == [1.0, 2.0, 3.0]
    assert _canonicalize(object()).startswith("<object object at")
    assert np.allclose(
        _polyline_point_at(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float), 0.5),
        np.array([1.0, 0.0, 0.0]),
    )
    assert np.allclose(
        _polyline_point_at(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=float), 0.75),
        np.array([1.0, 1.0, 1.0]),
    )

    with pytest.raises(ValueError, match="finite numeric values"):
        _canonicalize(float("inf"))


def test_surface_metadata_trim_and_container_validation_cover_error_paths() -> None:
    assert _split_metadata({"kernel": {"a": 1}, "consumer": {"b": 2}}) == ({"a": 1}, {"b": 2})

    with pytest.raises(ValueError, match="may only use 'kernel' and 'consumer'"):
        _split_metadata({"kernel": {}, "consumer": {}, "extra": True})

    with pytest.raises(ValueError, match="must be dictionaries"):
        _split_metadata({"kernel": [], "consumer": {}})

    with pytest.raises(ValueError, match="category must be 'outer' or 'inner'"):
        TrimLoop([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], category="mystery")

    with pytest.raises(ValueError, match="at least three distinct points"):
        TrimLoop([(0.0, 0.0), (1.0, 0.0)], category="outer")

    trim = TrimLoop([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], category="outer")
    with pytest.raises(ValueError, match="outside the patch domain"):
        trim.validate_against_domain(ParameterDomain((2.0, 3.0), (2.0, 3.0)))

    with pytest.raises(ValueError, match="family must be non-empty"):
        PlanarSurfacePatch(family="   ")

    outer = TrimLoop([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], category="outer")
    with pytest.raises(ValueError, match="more than one outer trim loop"):
        PlanarSurfacePatch(family="planar", trim_loops=(outer, outer))

    with pytest.raises(ValueError, match="transform_matrix must contain only finite values"):
        PlanarSurfacePatch(family="planar", transform_matrix=np.full((4, 4), np.nan))

    with pytest.raises(ValueError, match="angles must be finite"):
        RevolutionSurfacePatch(family="revolution", start_angle_deg=float("nan"))

    patch = PlanarSurfacePatch(family="planar", metadata={"kernel": {"k": 1}, "consumer": {"c": 2}})
    assert patch.outer_trim is None
    assert patch.inner_trims == ()
    with pytest.raises(ValueError, match="outside patch domain"):
        patch.validate_parameters(2.0, 2.0)
    assert patch.merged_kernel_metadata({"base": 0}) == {"base": 0, "k": 1}
    assert patch.merged_consumer_metadata({"base": 0}) == {"base": 0, "c": 2}

    seam = SurfaceSeam("known", (SurfaceBoundaryRef(0, "left"),))
    with pytest.raises(ValueError, match="unknown seam_id"):
        SurfaceShell(
            patches=(patch,),
            seams=(seam,),
            adjacency=(SurfaceAdjacencyRecord(source=SurfaceBoundaryRef(0, "left"), target=None, seam_id="missing"),),
        )

    with pytest.raises(ValueError, match="requires at least one patch"):
        SurfaceShell(patches=())

    with pytest.raises(TypeError, match="must all be SurfacePatch instances"):
        SurfaceShell(patches=("not-a-patch",))

    shell = make_surface_shell([patch], metadata={"kernel": {"shell": 1}, "consumer": {"view": "demo"}})
    assert shell.merged_kernel_metadata({"root": 1}) == {"root": 1, "shell": 1}
    assert shell.merged_consumer_metadata({"root": 1}) == {"root": 1, "view": "demo"}

    with pytest.raises(ValueError, match="requires at least one shell"):
        make_surface_body([])

    with pytest.raises(TypeError, match="must all be SurfaceShell instances"):
        make_surface_body(["not-a-shell"])  # type: ignore[list-item]

    body = make_surface_body([shell], metadata={"kernel": {"body": 1}, "consumer": {"label": "demo"}})
    assert body.merged_kernel_metadata({"root": 1}) == {"root": 1, "body": 1}
    assert body.merged_consumer_metadata({"root": 1}) == {"root": 1, "label": "demo"}


def test_tessellation_private_helper_branches_are_exercised() -> None:
    patch = PlanarSurfacePatch(family="planar")

    assert _validate_positive_or_none(None, name="tol") is None
    assert _validate_positive_or_none(1.5, name="tol") == 1.5

    with pytest.raises(ValueError, match="must be finite"):
        _validate_positive_or_none(np.inf, name="tol")

    with pytest.raises(ValueError, match="must be > 0"):
        _validate_positive_or_none(0.0, name="tol")

    assert _clamp_patch_parameters(patch, -1e-7, 1.0 + 1e-7) == (0.0, 1.0)

    with pytest.raises(ValueError, match="outside patch domain"):
        _clamp_patch_parameters(patch, -0.01, 0.5)

    with pytest.raises(ValueError, match="at least 2 samples on each axis"):
        _rectangular_grid_uv_mesh_data(patch, u_count=1, v_count=2)

    with pytest.raises(ValueError, match="Unsupported boundary_id for boundary sampling"):
        _boundary_samples(patch, "diagonal")

    uv_vertices, _faces = _rectangular_grid_uv_mesh_data(patch, u_count=3, v_count=3)
    with pytest.raises(ValueError, match="Unsupported boundary_id for seam-aware tessellation"):
        _rectangular_boundary_indices(patch, uv_vertices, "diagonal")

    sphere_patch = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        sweep_angle_deg=360.0,
    )
    assert _boundary_is_collapsed(sphere_patch, "bottom") is True
    assert _boundary_is_collapsed(sphere_patch, "top") is True


def test_tessellation_request_and_shell_grid_helpers_cover_remaining_branches() -> None:
    with pytest.raises(ValueError, match="intent must be"):
        TessellationRequest(intent="draft")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Only 'mesh' tessellation output"):
        TessellationRequest(output="points")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="quality_preset must be one of"):
        TessellationRequest(quality_preset="ultra")  # type: ignore[arg-type]

    request = TessellationRequest(intent="analysis")
    assert request.canonical_payload()["intent"] == "analysis"
    assert _rectangular_grid_counts(normalize_tessellation_request(request)) == (48, 24)

    ruled_patch = RuledSurfacePatch(family="ruled")
    ruled_shell = make_surface_shell(
        [ruled_patch],
        seams=(SurfaceSeam("wrap", (SurfaceBoundaryRef(0, "bottom"), SurfaceBoundaryRef(0, "top"))),),
    )
    assert _patch_requires_shell_grid_tessellation(ruled_shell, 0, ruled_patch) is True
    assert _shell_grid_counts_for_patch(
        ruled_shell,
        0,
        ruled_patch,
        normalize_tessellation_request(TessellationRequest(intent="preview")),
    ) == (16, 2)

    revolution_patch = RevolutionSurfacePatch(family="revolution")
    revolution_shell = make_surface_shell([revolution_patch])
    assert _patch_requires_shell_grid_tessellation(revolution_shell, 0, revolution_patch) is True


def test_tessellation_boundary_index_and_seam_assignment_failures_are_exercised() -> None:
    trimmed_patch = PlanarSurfacePatch(
        family="planar",
        trim_loops=(TrimLoop([(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)], category="outer"),),
    )
    with pytest.raises(ValueError, match="does not expose enough sampled vertices"):
        _rectangular_boundary_indices(
            trimmed_patch,
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            "trim:outer",
        )

    with pytest.raises(ValueError, match="does not expose enough sampled vertices"):
        _rectangular_boundary_indices(
            PlanarSurfacePatch(family="planar"),
            np.array([[0.0, 0.0]], dtype=float),
            "left",
        )

    open_shell = make_surface_shell(
        [PlanarSurfacePatch(family="planar")],
        seams=(SurfaceSeam("open", (SurfaceBoundaryRef(0, "left"),)),),
    )
    uv_vertices, _faces = _rectangular_grid_uv_mesh_data(open_shell.patches[0], u_count=3, v_count=3)
    assert _seam_vertex_assignments(open_shell, open_shell.iter_patches(world=True), (uv_vertices,)) == {}

    shell = make_surface_shell(
        [PlanarSurfacePatch(family="planar"), PlanarSurfacePatch(family="planar", origin=(2.0, 0.0, 0.0))],
        seams=(SurfaceSeam("bad", (SurfaceBoundaryRef(0, "left"), SurfaceBoundaryRef(1, "right"))),),
    )
    owner_uv = np.array([[0.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=float)
    peer_uv = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]], dtype=float)
    with pytest.raises(ValueError, match="boundary sample counts do not match"):
        _seam_vertex_assignments(shell, shell.iter_patches(world=True), (owner_uv, peer_uv))


def test_private_surface_builders_and_ops_cover_validation_branches() -> None:
    transform = _ops_attached_transform((1.0, 2.0, 3.0), (0.0, 0.0, 4.0))
    assert np.allclose(transform[:3, 3], np.array([1.0, 2.0, 3.0]))
    assert np.isclose(np.linalg.det(transform[:3, :3]), 1.0)

    rotation = _rotation_transform((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    assert np.allclose(rotation[:3, 2], np.array([0.0, 0.0, 1.0]))

    with pytest.raises(ValueError, match="perpendicular"):
        _rotation_transform((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="must contain only finite values"):
        _ops_as_vec3((0.0, 0.0, np.inf), name="center")

    with pytest.raises(ValueError, match="must be non-zero"):
        _ops_normalize((0.0, 0.0, 0.0), name="direction")

    with pytest.raises(ValueError, match="at least three points"):
        _closed_loop_points(Loop(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)))

    with pytest.raises(ValueError, match="positive planar span"):
        _domain_for_region(
            Region(
                outer=Loop(np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)),
            )
        )

    with pytest.raises(ValueError, match="requires a single connected region"):
        make_surface_rotate_extrude(
            Section(
                regions=(
                    Region(outer=Loop(np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=float))),
                    Region(outer=Loop(np.array([[1.5, -0.5], [2.5, -0.5], [2.5, 0.5], [1.5, 0.5]], dtype=float))),
                )
            )
        )

    shape = make_rect(size=(1.0, 1.0))
    with pytest.raises(ValueError, match="height must be positive"):
        make_surface_linear_extrude(shape, height=0.0)

    with pytest.raises(ValueError, match="at least one non-empty region"):
        make_surface_linear_extrude(Section(regions=()))

    with pytest.raises(ValueError, match="angle_deg must be non-zero"):
        make_surface_rotate_extrude(shape, angle_deg=0.0)

    with pytest.raises(ValueError, match="segments must be >= 3"):
        make_surface_rotate_extrude(shape, segments=2)

    holed = PlanarShape2D(
        outer=make_rect(size=(2.0, 2.0)).outer,
        holes=[make_rect(size=(0.5, 0.5)).outer],
    )
    extruded = make_surface_linear_extrude(holed, height=1.0)
    assert any(seam.seam_id.startswith("hole-0-") for seam in extruded.shells[0].seams)

    with pytest.raises(ValueError, match="size components must all be > 0"):
        make_surface_box(size=(0.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="radius must be positive"):
        make_surface_cylinder(radius=0.0)

    with pytest.raises(ValueError, match="resolution must be >= 8"):
        make_surface_cylinder(resolution=4)
    assert make_surface_cylinder(capping=False).shells[0].connected is True

    with pytest.raises(ValueError, match="Specify either bottom_diameter or radius, not both"):
        make_surface_cone(bottom_diameter=2.0, radius=2.0)

    with pytest.raises(ValueError, match="At least one of bottom_diameter or top_diameter must be > 0"):
        make_surface_cone(bottom_diameter=0.0, top_diameter=0.0)
    assert make_surface_cone(bottom_diameter=0.0, top_diameter=1.0).shells[0].connected is True

    with pytest.raises(ValueError, match="base_size must be positive"):
        make_surface_prism(base_size=(0.0, 1.0))

    with pytest.raises(ValueError, match="top_size must be non-negative"):
        make_surface_prism(base_size=(1.0, 1.0), top_size=(-1.0, 1.0))

    with pytest.raises(ValueError, match="major_radius must be positive"):
        make_surface_torus(major_radius=0.0)

    with pytest.raises(ValueError, match="n_theta and n_phi must both be >= 8"):
        make_surface_torus(n_theta=4, n_phi=4)

    with pytest.raises(ValueError, match="radius must be positive"):
        make_surface_sphere(radius=0.0)

    with pytest.raises(ValueError, match="theta_resolution and phi_resolution must both be >= 8"):
        make_surface_sphere(theta_resolution=4, phi_resolution=4)

    with pytest.raises(ValueError, match="radius must be positive"):
        make_surface_polyhedron(radius=0.0)

    with pytest.raises(ValueError, match="at least three points"):
        _planar_patch_from_face(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float), metadata={})

    with pytest.raises(ValueError, match="degenerate first edge"):
        _planar_patch_from_face(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
            metadata={},
        )

    with pytest.raises(ValueError, match="must not be collinear"):
        _planar_patch_from_face(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
            metadata={},
        )
