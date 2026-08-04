from __future__ import annotations

from pathlib import Path

import pytest


DOCUMENTATION_AUTHORITY = Path(".agents/skills/documentation/SKILL.md")
REFERENCE_AUTHORITY = Path(".agents/skills/reference-artifact-lifecycle/SKILL.md")
RELEASE_LIFECYCLE_AUTHORITY = Path("project/releases/README.md")


def _assert_semantic_obligations(path: Path, obligations: tuple[tuple[str, ...], ...]) -> None:
    text = path.read_text().casefold()
    for obligation in obligations:
        assert all(term.casefold() in text for term in obligation), (
            f"{path} is missing the semantic obligation expressed by {obligation!r}"
        )


def _assert_documentation_completion(path: Path) -> None:
    _assert_semantic_obligations(
        path,
        (
            ("durable documentation", "missing or stale"),
            ("documentation completion", "delivery", "not optional"),
        ),
    )


def _assert_reference_artifact_lifecycle(path: Path) -> None:
    _assert_semantic_obligations(
        path,
        (
            ("rendered reference image", "exported reference stl"),
            ("dirty references", "clean references", "explicit human review"),
            ("must not silently promote dirty references",),
        ),
    )


def _assert_release_lifecycle(path: Path) -> None:
    _assert_semantic_obligations(
        path,
        (
            ("only active release work", "top-level"),
            ("move it under", "project/releases/"),
            ("archived release folders", "historical records", "not", "active planning"),
        ),
    )


def test_current_documentation_authorities_exist(project_root: Path) -> None:
    for relative_path in (
        DOCUMENTATION_AUTHORITY,
        REFERENCE_AUTHORITY,
        RELEASE_LIFECYCLE_AUTHORITY,
    ):
        assert (project_root / relative_path).is_file(), f"missing authority: {relative_path}"


def test_documentation_guidance_requires_durable_completion(project_root: Path) -> None:
    _assert_documentation_completion(project_root / DOCUMENTATION_AUTHORITY)


def test_reference_guidance_covers_image_and_stl_lifecycle(project_root: Path) -> None:
    _assert_reference_artifact_lifecycle(project_root / REFERENCE_AUTHORITY)


def test_release_lifecycle_separates_active_work_from_archives(project_root: Path) -> None:
    _assert_release_lifecycle(project_root / RELEASE_LIFECYCLE_AUTHORITY)


@pytest.mark.parametrize(
    ("authority", "validator", "required_phrase"),
    (
        (DOCUMENTATION_AUTHORITY, _assert_documentation_completion, "durable documentation"),
        (REFERENCE_AUTHORITY, _assert_reference_artifact_lifecycle, "exported reference STL"),
        (RELEASE_LIFECYCLE_AUTHORITY, _assert_release_lifecycle, "move it under"),
    ),
)
def test_removing_a_required_obligation_fails_clearly(
    project_root: Path,
    tmp_path: Path,
    authority: Path,
    validator,
    required_phrase: str,
) -> None:
    mutated = tmp_path / authority.name
    original = (project_root / authority).read_text()
    mutated.write_text(original.replace(required_phrase, "", 1))

    with pytest.raises(AssertionError, match="semantic obligation"):
        validator(mutated)


def test_unrelated_prose_does_not_destabilize_semantic_checks(
    project_root: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "documentation-skill.md"
    copied.write_text(
        (project_root / DOCUMENTATION_AUTHORITY).read_text()
        + "\nAdditional explanatory prose that changes no obligation.\n"
    )

    _assert_documentation_completion(copied)
