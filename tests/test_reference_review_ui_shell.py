from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

from impression.devtools.reference_review.ui import (
    BridgeRecord,
    BridgeRegistry,
    DependencyPolicyRecord,
    PackageResourceManifest,
    build_dependency_policy_report,
    launch_workbench,
    load_style_tokens,
    verify_qml_resource_layout,
)
from impression.devtools.reference_review.ui.style import component_contracts


def test_reference_review_ui_dependency_is_optional_extra() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    core_dependencies = pyproject["project"]["dependencies"]

    report = build_dependency_policy_report(
        DependencyPolicyRecord(),
        declared_extras=extras.keys(),
        core_dependencies=core_dependencies,
    )

    assert report.valid
    assert "reference-review-ui" in extras
    assert any(dep.startswith("PySide6") for dep in extras["reference-review-ui"])
    assert not any(dep.startswith("PySide6") for dep in core_dependencies)


def test_qml_resource_layout_contains_shell_and_component_files() -> None:
    result = verify_qml_resource_layout()

    assert result.valid
    assert not result.diagnostics


def test_qml_resource_layout_reports_missing_files(tmp_path: Path) -> None:
    result = verify_qml_resource_layout(PackageResourceManifest(qml_root=tmp_path))

    assert not result.valid
    assert "missing-qml-resource:Main.qml" in result.diagnostics


def test_bridge_registry_allows_only_named_non_authority_bridges() -> None:
    registry = BridgeRegistry().register(BridgeRecord("queueBridge", object()))

    assert "queueBridge" in registry.records
    with pytest.raises(ValueError, match="bridge-not-allowlisted"):
        registry.register(BridgeRecord("fileDialogBridge", object()))
    with pytest.raises(ValueError, match="bridge-authority-forbidden"):
        registry.register(BridgeRecord("notesBridge", object(), authority=("promotion",)))


def test_bridge_registry_reports_missing_required_bridges() -> None:
    diagnostics = BridgeRegistry().diagnostics(("queueBridge", "notesBridge"))

    assert [item.bridge_name for item in diagnostics] == ["queueBridge", "notesBridge"]


def test_component_style_contracts_are_stable_and_named() -> None:
    tokens = load_style_tokens()
    contracts = component_contracts()

    assert {token.name for token in tokens} >= {"surface", "panel", "accent"}
    assert {contract.name for contract in contracts} >= {
        "IconButton",
        "TextField",
        "StatusBadge",
        "SplitPane",
    }
    assert all(contract.stable_size for contract in contracts)


def test_qml_shell_launches_offscreen_without_loading_fixture_on_ui_thread() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    result = launch_workbench(offscreen=True)

    assert result.launched
    assert result.engine is not None
