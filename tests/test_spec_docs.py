from __future__ import annotations


def test_loft_spec_60_is_retired_as_canonical_mesh_execution(project_root) -> None:
    spec = project_root / "project/release-0.1.0a/specifications/loft-60-mesh-executor-correspondence-consumption-v1_0.md"
    text = spec.read_text()
    normalized = " ".join(text.split())

    assert "Status: Retired as canonical modeled execution" in text
    assert "`LoftPlan -> SurfaceBody` is the canonical modeled" in normalized
    assert "legacy compatibility, debug, or tessellation-boundary" in normalized
    assert "Status: Proposed" not in text
    assert "mesh executor owns emitted mesh geometry" not in text
