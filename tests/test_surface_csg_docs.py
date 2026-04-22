from __future__ import annotations

from pathlib import Path


def test_csg_docs_explain_surface_migration_posture(project_root: Path) -> None:
    doc = (project_root / "docs" / "modeling" / "csg.md").read_text()
    assert 'backend="surface"' in doc
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
