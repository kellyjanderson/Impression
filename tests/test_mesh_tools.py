from __future__ import annotations

import numpy as np
import pytest

from impression.mesh import Mesh, MeshSectionResult, analyze_mesh, repair_mesh, section_mesh_with_plane
from impression.modeling import make_box, make_cylinder


def test_section_mesh_with_plane_returns_closed_box_slice() -> None:
    mesh = make_box(size=(2.0, 2.0, 2.0), backend="mesh")

    result = section_mesh_with_plane(mesh, origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))

    assert isinstance(result, MeshSectionResult)
    assert result.polyline_count == 1
    assert result.closed_count == 1
    polyline = result.polylines[0]
    assert polyline.closed is True
    assert polyline.points.shape[0] >= 4
    assert np.allclose(polyline.points[:, 2], 0.0)
    xmin, xmax, ymin, ymax, zmin, zmax = polyline.bounds
    assert xmin == pytest.approx(-1.0)
    assert xmax == pytest.approx(1.0)
    assert ymin == pytest.approx(-1.0)
    assert ymax == pytest.approx(1.0)
    assert zmin == pytest.approx(0.0)
    assert zmax == pytest.approx(0.0)


def test_section_mesh_with_plane_returns_closed_cylinder_slice() -> None:
    mesh = make_cylinder(radius=1.0, height=2.0, backend="mesh")

    result = section_mesh_with_plane(mesh, origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))

    assert result.polyline_count == 1
    assert result.closed_count == 1
    polyline = result.polylines[0]
    assert polyline.closed is True
    assert polyline.points.shape[0] >= 16
    radii = np.linalg.norm(polyline.points[:, :2], axis=1)
    assert np.all(radii <= 1.05)
    assert np.allclose(polyline.points[:, 2], 0.0, atol=1e-8)


def test_section_mesh_with_plane_returns_empty_for_disjoint_plane() -> None:
    mesh = make_box(size=(1.0, 1.0, 1.0), backend="mesh")

    result = section_mesh_with_plane(mesh, origin=(0.0, 0.0, 5.0), normal=(0.0, 0.0, 1.0))

    assert result.polylines == ()
    assert result.polyline_count == 0


def test_section_mesh_with_plane_validates_inputs() -> None:
    mesh = Mesh(vertices=np.zeros((0, 3), dtype=float), faces=np.zeros((0, 3), dtype=int))

    with pytest.raises(ValueError, match="normal must be non-zero"):
        section_mesh_with_plane(mesh, normal=(0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="epsilon must be positive"):
        section_mesh_with_plane(mesh, epsilon=0.0)

    with pytest.raises(ValueError, match="stitch_epsilon must be positive"):
        section_mesh_with_plane(mesh, stitch_epsilon=0.0)


def test_mesh_analysis_and_sectioning_can_be_used_together_for_tooling() -> None:
    mesh = make_box(size=(1.0, 1.0, 1.0), backend="mesh")

    analysis = analyze_mesh(mesh)
    section = section_mesh_with_plane(mesh, origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))

    assert analysis.is_watertight is True
    assert section.closed_count == 1


def test_repair_mesh_removes_invalid_and_degenerate_faces() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [np.nan, 0.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 0, 1],
            [0, 2, 3],
        ],
        dtype=int,
    )
    mesh = Mesh(vertices=vertices, faces=faces)

    repaired, report = repair_mesh(mesh)

    assert report.removed_invalid_faces == 1
    assert report.removed_degenerate_faces == 1
    assert report.changed is True
    assert repaired.n_faces == 1
    assert repaired.analysis is not None
    assert repaired.analysis.has_invalid_vertices is False
    assert repaired.analysis.has_degenerate_faces is False


def test_repair_mesh_removes_unreferenced_vertices_and_preserves_face_colors() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2]], dtype=int)
    face_colors = np.asarray([[1.0, 0.0, 0.0, 1.0]], dtype=float)
    mesh = Mesh(vertices=vertices, faces=faces, face_colors=face_colors)

    repaired, report = repair_mesh(mesh)

    assert report.removed_unreferenced_vertices == 1
    assert repaired.n_vertices == 3
    assert repaired.face_colors is not None
    assert repaired.face_colors.shape == (1, 4)


def test_repair_mesh_validates_area_epsilon() -> None:
    mesh = make_box(size=(1.0, 1.0, 1.0), backend="mesh")

    with pytest.raises(ValueError, match="area_epsilon must be positive"):
        repair_mesh(mesh, area_epsilon=0.0)
