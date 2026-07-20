from __future__ import annotations

from pathlib import Path


def test_loft_spec_60_is_retired_as_canonical_mesh_execution(project_root) -> None:
    spec = project_root / "project/release-0.1.0a/specifications/loft-60-mesh-executor-correspondence-consumption-v1_0.md"
    text = spec.read_text()
    normalized = " ".join(text.split())

    assert "Status: Retired as canonical modeled execution" in text
    assert "`LoftPlan -> SurfaceBody` is the canonical modeled" in normalized
    assert "legacy compatibility, debug, or tessellation-boundary" in normalized
    assert "Status: Proposed" not in text
    assert "mesh executor owns emitted mesh geometry" not in text


def test_reference_review_architecture_index_lists_child_domains(project_root: Path) -> None:
    architecture = project_root / "project/release-0.1.0a/architecture"
    parent = architecture / "reference-review-workbench-architecture.md"
    text = parent.read_text()
    child_docs = (
        "reference-review-fixture-source-contract.md",
        "reference-review-qt-workbench-ui.md",
        "reference-review-async-concurrency.md",
        "reference-review-promotion-and-notes-lifecycle.md",
        "reference-review-codex-sandbox.md",
    )

    assert "## Domain Ownership Index" in text
    assert "## Cross-Document Commitments" in text
    for child in child_docs:
        assert child in text
        assert (architecture / child).is_file()
    assert "Codex sandbox owns sidecar authority" in text
    assert "Qt workbench UI owns visible state" in text
