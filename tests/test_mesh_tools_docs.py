from __future__ import annotations

from pathlib import Path


def test_mesh_tools_docs_describe_analysis_and_repair_boundary(project_root: Path) -> None:
    doc = (project_root / "docs" / "modeling" / "mesh-tools.md").read_text()
    assert "analyze_mesh" in doc
    assert "section_mesh_with_plane" in doc
    assert "repair_mesh" in doc
    assert "not canonical modeling truth" in doc or "not canonical modeling" in doc
    assert "union_meshes(...)" in doc
    assert "hull([...])" in doc


def test_mesh_capability_retention_matrix_uses_explicit_inventory_fields(project_root: Path) -> None:
    matrix = (project_root / "project" / "research" / "2026-04-19-mesh-capability-retention-matrix.md").read_text()
    assert "## Inventory Fields" in matrix
    assert "`retain_or_delete`" in matrix
    assert "| area | file | symbols | role | retain_or_delete | target_state | notes |" in matrix
    assert "mesh-primitives" in matrix
    assert "mesh-loft" in matrix


def test_mesh_utility_docs_keep_standalone_tool_posture_explicit(project_root: Path) -> None:
    csg_doc = (project_root / "docs" / "modeling" / "csg.md").read_text()
    transform_doc = (project_root / "docs" / "modeling" / "transforms.md").read_text()
    assert "standalone mesh tool" in csg_doc
    assert "retained as an explicit standalone utility" in transform_doc
    assert "not canonical surfaced modeling truth" in transform_doc
