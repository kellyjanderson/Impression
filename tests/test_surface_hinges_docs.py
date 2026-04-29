from __future__ import annotations


def test_hinges_docs_explain_surface_migration_layers(project_root) -> None:
    doc = (project_root / "docs" / "modeling" / "hinges.md").read_text()
    assert "HingeSurfaceAssembly" in doc
    assert "handoff_hinge_surface" in doc
    assert "legacy mesh builders still exist" in doc
    assert "do not silently collapse back to mesh-first truth" in doc
    assert "consumer-facing color metadata" in doc


def test_hinges_docs_define_required_reference_fixtures(project_root) -> None:
    doc = (project_root / "docs" / "modeling" / "hinges.md").read_text()
    assert "surfacebody/hinge_traditional_pair" in doc
    assert "surfacebody/hinge_living_panel" in doc
    assert "surfacebody/hinge_bistable_blank" in doc
    assert "dirty and clean reference images" in doc
    assert "dirty and clean reference STL files" in doc


def test_hinges_docs_define_surface_public_handoff(project_root) -> None:
    doc = (project_root / "docs" / "modeling" / "hinges.md").read_text()
    assert "SurfaceConsumerCollection" in doc
    assert "standard surface handoff boundary" in doc
    assert "backend=\"surface\"" in doc
