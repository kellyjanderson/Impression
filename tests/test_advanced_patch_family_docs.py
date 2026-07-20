from __future__ import annotations

from pathlib import Path

import impression.modeling as modeling


def test_advanced_patch_family_docs_cover_public_api_inventory() -> None:
    project_root = Path(__file__).resolve().parents[1]
    doc = (project_root / "docs" / "modeling" / "advanced-patch-families.md").read_text(encoding="utf-8")
    index = (project_root / "docs" / "index.md").read_text(encoding="utf-8")
    required_names = (
        "BSplineSurfacePatch",
        "NURBSSurfacePatch",
        "SweepSurfacePatch",
        "SubdivisionSurfacePatch",
        "ImplicitSurfacePatch",
        "HeightmapSurfacePatch",
        "DisplacementSurfacePatch",
        "make_subdivision_surface",
        "make_implicit_surface",
        "make_implicit_field_node",
        "loft_plan_sections",
        "loft_execute_plan",
        "tessellate_surface_body",
    )

    assert "modeling/advanced-patch-families.md" in index
    for name in required_names:
        assert hasattr(modeling, name), name
        assert name in doc, name
    assert "safe declarative field nodes" in doc
    assert "do not fall back to mesh" in doc
    assert "explicit tessellation" in doc
