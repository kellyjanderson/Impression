from __future__ import annotations

from pathlib import Path


def test_shared_and_project_documentation_rules_exist(project_root: Path) -> None:
    shared_docs = project_root / "agents" / "documentation.md"
    project_docs = project_root / "project" / "agents" / "reference-images.md"
    assert shared_docs.exists()
    assert project_docs.exists()


def test_spec_guidance_requires_documentation_for_completion(project_root: Path) -> None:
    spec_guidance = (project_root / "agents" / "specifications.md").read_text()
    assert "not fully complete without durable documentation" in spec_guidance
    assert "Final features should not ship with documentation left implied" in spec_guidance


def test_project_reference_rules_cover_images_and_stl(project_root: Path) -> None:
    project_rules = (project_root / "project" / "agents" / "reference-images.md").read_text()
    assert "Reference STL files live under" in project_rules
    assert "Agents must not silently promote dirty references to clean." in project_rules
    assert "rendered images and STL files" in (
        project_root / "project" / "specifications" / "surface-106-reference-artifact-regression-suite-v1_0.md"
    ).read_text()
