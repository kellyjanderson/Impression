from __future__ import annotations

import numpy as np

from impression.mesh import section_mesh_with_plane
from impression.modeling import (
    BSplineSurfacePatch,
    ImplicitFieldNode,
    ImplicitSurfacePatch,
    NURBSSurfacePatch,
    Path3D,
    SubdivisionSurfacePatch,
    SurfaceBody,
    SweepSurfacePatch,
    build_sampled_implicit_promotion_provenance_record,
    boolean_difference,
    boolean_union,
    evaluate_sampled_implicit_reconstruction_feasibility,
    make_box,
    make_surface_body,
    make_surface_shell,
    tessellate_surface_body,
    verify_sampled_implicit_promotion_matrix,
)


def _surface_boolean_body(result) -> object:
    assert result is not None
    assert getattr(result, "status", None) == "succeeded"
    body = getattr(result, "body", None)
    assert body is not None
    return body


def surface_body_section_loops(body, z_height: float) -> list[np.ndarray]:
    mesh = tessellate_surface_body(body).mesh
    result = section_mesh_with_plane(
        mesh,
        origin=(0.0, 0.0, float(z_height)),
        normal=(0.0, 0.0, 1.0),
        stitch_epsilon=1e-5,
    )
    return [polyline.points[:, :2] for polyline in result.polylines if polyline.closed]


def _fixture_metadata(
    *,
    fixture_name: str,
    orientation_cue: str,
) -> dict[str, object]:
    return {
        "fixture_name": fixture_name,
        "presentation": "triptych",
        "evidence_scope": "overlap",
        "orientation_cue": orientation_cue,
    }


def build_csg_union_box_post_fixture() -> dict[str, object]:
    base = make_box(size=(2.4, 2.0, 1.2), center=(0.0, 0.0, 0.0))
    post = make_box(size=(0.50, 0.38, 0.95), center=(1.15, -0.48, 0.28))
    result = _surface_boolean_body(boolean_union([base, post]))
    expected_slice = np.asarray(
        [
            (-1.2, -1.0),
            (1.2, -1.0),
            (1.2, -0.67),
            (1.4, -0.67),
            (1.4, -0.29),
            (1.2, -0.29),
            (1.2, 1.0),
            (-1.2, 1.0),
        ],
        dtype=float,
    )
    fixture = {
        "left_operand": base,
        "result_body": result,
        "right_operand": post,
        "slice_z": 0.0,
        "expected_slice_loops": [expected_slice],
        "orientation_required": True,
    }
    fixture.update(_fixture_metadata(fixture_name="surfacebody/csg_union_box_post", orientation_cue="edge_protrusion"))
    return fixture


def build_csg_difference_slot_fixture() -> dict[str, object]:
    base = make_box(size=(2.4, 2.0, 1.2), center=(0.0, 0.0, 0.0))
    cutter = make_box(size=(1.90, 1.40, 1.60), center=(-0.25, 0.50, 0.0))
    result = _surface_boolean_body(boolean_difference(base, [cutter]))
    expected_slice = np.asarray(
        [
            (-1.2, -1.0),
            (1.2, -1.0),
            (1.2, 1.0),
            (0.70, 1.0),
            (0.70, -0.20),
            (-1.2, -0.20),
        ],
        dtype=float,
    )
    fixture = {
        "left_operand": base,
        "result_body": result,
        "right_operand": cutter,
        "slice_z": 0.0,
        "expected_slice_loops": [expected_slice],
        "orientation_required": True,
    }
    fixture.update(_fixture_metadata(fixture_name="surfacebody/csg_difference_slot", orientation_cue="edge_protrusion"))
    return fixture


def make_box_with_higher_order_front_wall(family: str) -> SurfaceBody:
    box = make_box(size=(1.0, 1.0, 1.0))
    shell = box.iter_shells(world=True)[0]
    front = shell.patches[0]
    control_net = np.asarray(
        [
            [front.point_at(0.0, 0.0), front.point_at(0.0, 1.0)],
            [front.point_at(1.0, 0.0), front.point_at(1.0, 1.0)],
        ],
        dtype=float,
    )
    metadata = {"kernel": {"authored_surface_family": family, "surface_role": "front-wall"}}
    if family == "bspline":
        front_patch = BSplineSurfacePatch(family="bspline", control_net=control_net, metadata=metadata)
    elif family == "nurbs":
        front_patch = NURBSSurfacePatch(
            family="nurbs",
            control_net=control_net,
            weights=np.ones((2, 2), dtype=float),
            metadata=metadata,
        )
    else:
        raise ValueError(f"unsupported higher-order fixture family {family!r}")
    return make_surface_body(
        (
            make_surface_shell(
                (front_patch, *shell.patches[1:]),
                connected=True,
                seams=shell.seams,
                adjacency=shell.adjacency,
                metadata={"kernel": {"operation": "box", "authored_surface_family": family}},
            ),
        ),
        metadata={"kernel": {"operation": "box", "authored_surface_family": family}},
    )


def make_box_with_sweep_front_wall() -> SurfaceBody:
    box = make_box(size=(1.0, 1.0, 1.0))
    shell = box.iter_shells(world=True)[0]
    front_patch = SweepSurfacePatch(
        family="sweep",
        path=Path3D.from_points([(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5)]),
        profile_points_uv=np.asarray([(0.0, 0.0), (1.0, 0.0)], dtype=float),
        metadata={"kernel": {"authored_surface_family": "sweep", "surface_role": "front-wall"}},
    )
    return make_surface_body(
        (
            make_surface_shell(
                (front_patch, *shell.patches[1:]),
                connected=True,
                seams=shell.seams,
                adjacency=shell.adjacency,
                metadata={"kernel": {"authored_surface_family": "sweep"}},
            ),
        ),
        metadata={"kernel": {"authored_surface_family": "sweep"}},
    )


def make_box_with_subdivision_front_wall() -> SurfaceBody:
    box = make_box(size=(1.0, 1.0, 1.0))
    shell = box.iter_shells(world=True)[0]
    front = shell.patches[0]
    front_patch = SubdivisionSurfacePatch(
        family="subdivision",
        control_points=(
            tuple(float(value) for value in front.point_at(0.0, 0.0)),
            tuple(float(value) for value in front.point_at(1.0, 0.0)),
            tuple(float(value) for value in front.point_at(1.0, 1.0)),
            tuple(float(value) for value in front.point_at(0.0, 1.0)),
        ),
        metadata={"kernel": {"authored_surface_family": "subdivision", "surface_role": "front-wall"}},
    )
    return make_surface_body(
        (
            make_surface_shell(
                (front_patch, *shell.patches[1:]),
                connected=True,
                seams=shell.seams,
                adjacency=shell.adjacency,
                metadata={"kernel": {"authored_surface_family": "subdivision"}},
            ),
        ),
        metadata={"kernel": {"authored_surface_family": "subdivision"}},
    )


def make_sampled_implicit_promotion_target_body(target_family: str) -> SurfaceBody:
    matrix = verify_sampled_implicit_promotion_matrix(operations=("union",))
    row = next((candidate for candidate in matrix.rows if candidate.target_family == target_family), None)
    if row is None:
        raise ValueError(f"No sampled/implicit promotion row targets {target_family!r}.")
    provenance = build_sampled_implicit_promotion_provenance_record(
        row,
        operand_ids=(f"{row.left_family}:fixture-left", f"{row.right_family}:fixture-right"),
    )
    feasibility = evaluate_sampled_implicit_reconstruction_feasibility(
        provenance,
        estimated_sample_count=16,
        residual=0.0,
    )
    if not provenance.supported or not feasibility.supported:
        raise ValueError(f"Sampled/implicit promotion target {target_family!r} is not fixture-ready.")
    metadata = {
        "kernel": {
            "sampled_implicit_promotion": provenance.canonical_payload(),
            "fixture_target_family": target_family,
            "no_mesh_fallback": True,
        }
    }
    if target_family == "implicit":
        patch = ImplicitSurfacePatch(
            family="implicit",
            field=ImplicitFieldNode("sphere"),
            bounds=(-0.75, 0.75, -0.75, 0.75, -0.75, 0.75),
            metadata=metadata,
        )
    elif target_family == "subdivision":
        patch = SubdivisionSurfacePatch(family="subdivision", metadata=metadata)
    elif target_family == "nurbs":
        patch = NURBSSurfacePatch(family="nurbs", metadata=metadata)
    elif target_family == "bspline":
        patch = BSplineSurfacePatch(family="bspline", metadata=metadata)
    else:
        raise ValueError(f"Unsupported sampled/implicit promotion target {target_family!r}.")
    return make_surface_body(
        (make_surface_shell((patch,), connected=False, metadata=metadata),),
        metadata=metadata,
    )
