from __future__ import annotations

from pathlib import Path

from impression.modeling import inventory_legacy_primitive_mesh_assumptions


def test_csg_docs_explain_surface_migration_posture(project_root: Path) -> None:
    doc = (project_root / "docs" / "modeling" / "csg.md").read_text()
    assert "surfacebody-result APIs" in doc
    assert "SurfaceBooleanResult" in doc
    assert 'status="unsupported"' not in doc or "remains explicitly unsupported" in doc
    assert "succeeds for a very small bounded initial scope" in doc
    assert "exact no-cut disjoint, touching, equal, and exact-containment cases" in doc
    assert "split-selection records" in doc
    assert "surface_boolean_overlap_fragments" in doc
    assert 'status="invalid"' in doc
    assert "bounded deterministic cleanup" in doc
    assert "does not fall back to mesh" in doc


def test_csg_docs_define_required_reference_fixtures(project_root: Path) -> None:
    doc = (project_root / "docs" / "modeling" / "csg.md").read_text()
    assert "surfacebody/csg_union_box_post" in doc
    assert "surfacebody/csg_difference_slot" in doc
    assert "surfacebody/csg_intersection_box_sphere" in doc
    assert "dirty and clean reference images" in doc
    assert "dirty and clean reference STL files" in doc
    assert "triptych-style operand/result presentation" in doc
    assert "edge-protrusion cue" in doc
    assert "does not yet own a meaningful partial-overlap box/sphere result" in doc
    assert "expected section bitmap" in doc
    assert "same shape but rotated" in doc


def test_tutorials_keep_mesh_boolean_lane_explicit_while_surface_lane_migrates(project_root: Path) -> None:
    getting_started = (project_root / "docs" / "tutorials" / "getting-started.md").read_text()
    serious_modeling = (project_root / "docs" / "tutorials" / "serious-modeling.md").read_text()
    assert "surfaced CSG lane" in getting_started
    assert "executable mesh boolean lane" in serious_modeling


def test_mesh_compatibility_docs_and_examples_use_explicit_mesh_inputs(project_root: Path) -> None:
    paths = (
        project_root / "README.md",
        project_root / "docs" / "modeling" / "csg.md",
        project_root / "docs" / "examples" / "csg" / "union_example.py",
        project_root / "docs" / "examples" / "csg" / "difference_example.py",
        project_root / "docs" / "examples" / "csg" / "intersection_example.py",
        project_root / "docs" / "examples" / "csg" / "union_meshes_example.py",
        project_root / "docs" / "examples" / "csg" / "teeth_union_example.py",
        project_root / "docs" / "examples" / "csg" / "tooth_union_example.py",
        project_root / "docs" / "examples" / "csg" / "tooth_example.py",
        project_root / "docs" / "examples" / "csg" / "tooth_parts_example.py",
    )
    report = inventory_legacy_primitive_mesh_assumptions(
        {str(path.relative_to(project_root)): path.read_text(encoding="utf-8") for path in paths}
    )
    csg_doc = (project_root / "docs" / "modeling" / "csg.md").read_text(encoding="utf-8")
    union_meshes_example = (project_root / "docs" / "examples" / "csg" / "union_meshes_example.py").read_text(
        encoding="utf-8"
    )

    assert report.stale_findings == ()
    assert "not treat public `make_*` primitives as mesh constructors" in csg_doc
    assert "make_box_mesh" in union_meshes_example
    assert "make_cylinder_mesh" in union_meshes_example
