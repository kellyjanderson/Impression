from __future__ import annotations

import numpy as np
import pytest

from impression.modeling import (
    MESH_GROUP_COMPATIBILITY_BOUNDARY,
    MESH_GROUP_COMPATIBILITY_CLASSIFICATION,
    MeshGroupCompatibilityError,
    SurfaceBody,
    TransformMeshCompatibilityResult,
    group,
    make_box,
    translate,
    rotate,
    rotate_euler,
    scale,
    resize,
    mirror,
    multmatrix,
    tessellate_surface_body,
)


def _bounds(mesh):
    return np.array(mesh.bounds, dtype=float)


def test_translate_bounds():
    base = make_box(size=(2.0, 4.0, 6.0), backend="mesh")
    moved = translate(base.copy(), (1.0, 2.0, 3.0))
    assert np.allclose(_bounds(moved), np.array([0.0, 2.0, 0.0, 4.0, 0.0, 6.0]))


def test_rotate_axis_z_90():
    base = make_box(size=(2.0, 4.0, 1.0), backend="mesh")
    turned = rotate(base.copy(), axis=(0.0, 0.0, 1.0), angle_deg=90.0)
    bounds = _bounds(turned)
    assert np.allclose(bounds[[0, 1]], [-2.0, 2.0], atol=1e-6)
    assert np.allclose(bounds[[2, 3]], [-1.0, 1.0], atol=1e-6)


def test_rotate_euler_matches_sequential():
    base = make_box(size=(1.0, 2.0, 3.0), backend="mesh")
    sequential = rotate(base.copy(), axis=(1.0, 0.0, 0.0), angle_deg=10.0)
    sequential = rotate(sequential, axis=(0.0, 1.0, 0.0), angle_deg=20.0)
    sequential = rotate(sequential, axis=(0.0, 0.0, 1.0), angle_deg=30.0)

    euler = rotate_euler(base.copy(), angles_deg=(10.0, 20.0, 30.0))
    assert np.allclose(sequential.vertices, euler.vertices, atol=1e-6)


def test_scale_bounds():
    base = make_box(size=(2.0, 4.0, 6.0), backend="mesh")
    scaled = scale(base.copy(), (2.0, 0.5, 1.0))
    assert np.allclose(_bounds(scaled), np.array([-2.0, 2.0, -1.0, 1.0, -3.0, 3.0]))


def test_resize_auto_axes():
    base = make_box(size=(2.0, 4.0, 6.0), backend="mesh")
    resized = resize(base.copy(), (4.0, 0.0, 0.0), auto=[False, True, True])
    assert np.allclose(_bounds(resized), np.array([-2.0, 2.0, -4.0, 4.0, -6.0, 6.0]))


def test_mirror_flips_axis():
    base = make_box(size=(2.0, 2.0, 2.0), center=(2.0, 0.0, 0.0), backend="mesh")
    flipped = mirror(base.copy(), (1.0, 0.0, 0.0))
    assert np.allclose(_bounds(flipped)[[0, 1]], [-3.0, -1.0])


def test_multmatrix_translation():
    base = make_box(size=(2.0, 2.0, 2.0), backend="mesh")
    mat = np.eye(4)
    mat[:3, 3] = [1.0, 2.0, 3.0]
    moved = multmatrix(base.copy(), mat)
    assert np.allclose(_bounds(moved), np.array([0.0, 2.0, 1.0, 3.0, 2.0, 4.0]))


def test_translate_surface_body_preserves_native_body_until_tessellation():
    base = make_box(size=(2.0, 2.0, 2.0))
    moved = translate(base, (1.0, 2.0, 3.0))

    assert isinstance(base, SurfaceBody)
    assert isinstance(moved, SurfaceBody)
    assert moved is not base
    assert np.allclose(base.transform_matrix, np.eye(4))
    assert np.allclose(moved.transform_matrix[:3, 3], [1.0, 2.0, 3.0])

    mesh = tessellate_surface_body(moved).mesh
    assert np.allclose(_bounds(mesh), np.array([0.0, 2.0, 1.0, 3.0, 2.0, 4.0]))


def test_surface_transform_family_returns_surface_bodies():
    base = make_box(size=(2.0, 2.0, 2.0))
    mat = np.eye(4)
    mat[:3, 3] = [0.5, 0.0, 0.0]

    transformed = [
        rotate(base, axis=(0.0, 0.0, 1.0), angle_deg=45.0),
        rotate_euler(base, angles_deg=(5.0, 10.0, 15.0)),
        scale(base, (2.0, 1.0, 1.0)),
        resize(base, (4.0, 0.0, 0.0), auto=[False, True, True]),
        mirror(base, (1.0, 0.0, 0.0)),
        multmatrix(base, mat),
    ]

    assert all(isinstance(body, SurfaceBody) for body in transformed)
    assert all(not np.allclose(body.transform_matrix, np.eye(4)) for body in transformed)


def test_mesh_transform_records_explicit_compatibility_boundary():
    mesh = make_box(size=(1.0, 1.0, 1.0), backend="mesh")

    moved = translate(mesh, (1.0, 0.0, 0.0))

    assert moved is mesh
    assert mesh.metadata["transform_mesh_compatibility"] == [
        TransformMeshCompatibilityResult("Mesh", "translate").canonical_payload()
    ]


def test_mesh_group_is_explicit_compatibility_api():
    mesh = make_box(size=(1.0, 1.0, 1.0), backend="mesh")

    grp = group([mesh])

    assert grp.classification == MESH_GROUP_COMPATIBILITY_CLASSIFICATION
    assert grp.metadata["mesh_group_compatibility"]["boundary"] == MESH_GROUP_COMPATIBILITY_BOUNDARY
    assert grp.metadata["mesh_group_compatibility"]["classification"] == MESH_GROUP_COMPATIBILITY_CLASSIFICATION
    assert grp.to_mesh().vertices.shape[0] == mesh.vertices.shape[0]


def test_mesh_group_rejects_surface_body_inputs_with_diagnostic():
    body = make_box(size=(1.0, 1.0, 1.0))

    with pytest.raises(MeshGroupCompatibilityError) as exc:
        group([body])

    diagnostic = exc.value.diagnostic
    assert diagnostic.boundary == MESH_GROUP_COMPATIBILITY_BOUNDARY
    assert diagnostic.target_type == "SurfaceBody"
    assert "SurfaceComposition" in diagnostic.reason
