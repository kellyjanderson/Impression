"""Optional dependency and QML resource policy checks for the workbench UI."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata, resources
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DependencyPolicyRecord:
    extra_name: str = "reference-review-ui"
    required_dependency: str = "PySide6"
    webengine_optional: bool = True


@dataclass(frozen=True)
class PackageResourceManifest:
    qml_root: Path
    required_files: tuple[str, ...] = (
        "Main.qml",
        "qtquickcontrols2.conf",
        "components/StatusBadge.qml",
    )


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
    if any(dep.split(">=")[0] == policy.required_dependency for dep in core_dependencies):
        diagnostics.append("pyside6-leaked-into-core-dependencies")
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

