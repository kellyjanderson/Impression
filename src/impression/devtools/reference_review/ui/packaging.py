"""Optional dependency and QML resource policy checks for the workbench UI."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata, resources
from pathlib import Path
from typing import Iterable

PREVIEW_DISPLAY_CONTROL_ICON_FILES: tuple[str, ...] = (
    "icons/preview-display/authored-colors.svg",
    "icons/preview-display/inspection-color.svg",
    "icons/preview-display/lighting-flat.svg",
    "icons/preview-display/lighting-face-normals.svg",
    "icons/preview-display/lighting-camera.svg",
    "icons/preview-display/object-fill.svg",
    "icons/preview-display/object-edges.svg",
    "icons/preview-display/triangle-wireframe.svg",
    "icons/preview-display/bounds-grid.svg",
    "icons/preview-display/axis-triad.svg",
    "icons/preview-display/gradient-background.svg",
    "icons/preview-display/polylines.svg",
)


@dataclass(frozen=True)
class DependencyPolicyRecord:
    extra_name: str = "reference-review-ui"
    required_dependencies: tuple[str, ...] = ("PySide6",)
    webengine_optional: bool = True


@dataclass(frozen=True)
class PackageResourceManifest:
    qml_root: Path
    required_files: tuple[str, ...] = (
        "Main.qml",
        "ComponentGallery.qml",
        "qtquickcontrols2.conf",
        "components/ArtifactPanel.qml",
        "components/CodexPanel.qml",
        "components/MarkdownPanel.qml",
        "components/NotesPanel.qml",
        "components/StatusBadge.qml",
    ) + PREVIEW_DISPLAY_CONTROL_ICON_FILES


@dataclass(frozen=True)
class DependencyPolicyReport:
    valid: bool
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackagingSmokeResult:
    valid: bool
    diagnostics: tuple[str, ...] = ()


def _declared_extras(distribution: str = "impression") -> set[str]:
    try:
        return set(metadata.metadata(distribution).get_all("Provides-Extra") or ())
    except metadata.PackageNotFoundError:
        return set()


def build_dependency_policy_report(
    policy: DependencyPolicyRecord = DependencyPolicyRecord(),
    *,
    declared_extras: Iterable[str] | None = None,
    core_dependencies: Iterable[str] = (),
) -> DependencyPolicyReport:
    extras = set(declared_extras) if declared_extras is not None else _declared_extras()
    diagnostics: list[str] = []
    if policy.extra_name not in extras:
        diagnostics.append("missing-reference-review-ui-extra")
    core_dependency_names = {dep.split(">=")[0] for dep in core_dependencies}
    for dependency in policy.required_dependencies:
        if dependency in core_dependency_names:
            diagnostics.append(f"{dependency.lower()}-leaked-into-core-dependencies")
    return DependencyPolicyReport(valid=not diagnostics, diagnostics=tuple(diagnostics))


def qml_resource_root() -> Path:
    return Path(str(resources.files(__package__) / "qml"))


def verify_qml_resource_layout(
    manifest: PackageResourceManifest | None = None,
) -> PackagingSmokeResult:
    manifest = manifest or PackageResourceManifest(qml_root=qml_resource_root())
    diagnostics = [
        f"missing-qml-resource:{relative}"
        for relative in manifest.required_files
        if not (manifest.qml_root / relative).is_file()
    ]
    return PackagingSmokeResult(valid=not diagnostics, diagnostics=tuple(diagnostics))
