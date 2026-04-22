from __future__ import annotations

import numpy as np

from impression.mesh import section_mesh_with_plane
from impression.modeling import boolean_difference, boolean_union, make_box, tessellate_surface_body


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
    base = make_box(size=(2.4, 2.0, 1.2), center=(0.0, 0.0, 0.0), backend="surface")
    post = make_box(size=(0.50, 0.38, 0.95), center=(1.15, -0.48, 0.28), backend="surface")
    result = _surface_boolean_body(boolean_union([base, post], backend="surface"))
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
    base = make_box(size=(2.4, 2.0, 1.2), center=(0.0, 0.0, 0.0), backend="surface")
    cutter = make_box(size=(1.90, 1.40, 1.60), center=(-0.25, 0.50, 0.0), backend="surface")
    result = _surface_boolean_body(boolean_difference(base, [cutter], backend="surface"))
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
