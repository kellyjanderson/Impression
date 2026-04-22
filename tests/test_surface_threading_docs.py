from __future__ import annotations


def test_threading_docs_explain_surface_migration_layers(project_root) -> None:
    doc = (project_root / "docs" / "modeling" / "threading.md").read_text()
    assert "ThreadSurfaceRepresentation" in doc
    assert "ThreadSurfaceAssembly" in doc
    assert "legacy executable mesh generation" in doc
    assert "do not silently collapse back to mesh-first assembly" in doc


def test_threading_docs_define_fit_and_quality_posture(project_root) -> None:
    doc = (project_root / "docs" / "modeling" / "threading.md").read_text()
    assert "Fit presets still change canonical geometry" in doc
    assert "Mesh-quality controls do not" in doc


def test_threading_docs_define_required_future_reference_fixtures(project_root) -> None:
    doc = (project_root / "docs" / "modeling" / "threading.md").read_text()
    assert "surfacebody/thread_external_metric_m6" in doc
    assert "surfacebody/thread_hex_nut_m6" in doc
    assert "surfacebody/thread_runout_relief_metric" in doc
    assert "dirty and clean reference images" in doc
    assert "dirty and clean reference STL files" in doc
