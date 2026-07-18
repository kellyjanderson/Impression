from __future__ import annotations

import os
import json
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from PySide6.QtCore import QObject, QSize, Qt
from PySide6.QtGui import QIcon, QImage
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QListWidget, QPushButton, QTabWidget, QTextEdit, QToolButton, QWidget

from impression.mesh import Mesh, Polyline
from impression.devtools.reference_review import (
    ReferenceEvidenceArtifactRecord,
    ReferenceEvidenceBundleRecord,
    ReferenceReviewStatus,
    ReviewSourceModelRecord,
)
from impression.devtools.reference_review.ui import (
    ArtifactEvidenceRow,
    BridgeRecord,
    BridgeRegistry,
    ContextEvidenceSummary,
    DependencyPolicyRecord,
    PackageResourceManifest,
    PREVIEW_DISPLAY_CONTROL_ICON_FILES,
    PREVIEW_DISPLAY_CONTROL_ICON_RECORDS,
    ExclusiveIconGroupState,
    ExclusiveIconOptionGroup,
    ExclusiveIconOptionRecord,
    PreviewDisplayControlBar,
    PreviewDisplayOptions,
    PreviewRenderCommand,
    PreviewRenderCommandKind,
    PreviewRenderCommandQueue,
    WorkbenchIconToggleButton,
    build_dependency_policy_report,
    format_missing_artifact_status,
    launch_workbench,
    load_style_tokens,
    map_artifact_evidence_rows,
    map_context_evidence_summary,
    PreviewRendererLifecycleWidget,
    PreviewWidgetPayloadState,
    preview_display_control_icon_record,
    preview_display_control_icon_records,
    route_preview_display_command,
    select_exclusive_icon_option,
    verify_qml_resource_layout,
)
from impression.devtools.reference_review import (
    LoadedPreviewDataset,
    PreviewPayload,
    PreviewPayloadDiagnostic,
    PreviewPayloadRequest,
    build_impress_preview_result,
    write_preview_payload_file,
)
import impression.devtools.reference_review.ui.preview_widget as preview_widget
from impression.devtools.reference_review.ui import artifact_preview
from impression.devtools.reference_review.ui.preview_widget import (
    _apply_pyvistaqt_scene,
)
from impression.devtools.reference_review.ui import shell
from impression.devtools.reference_review.ui.shell import InteractiveStlPreviewLabel
from impression.devtools.reference_review.ui.style import component_contracts


def test_reference_review_ui_dependency_is_optional_extra() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    core_dependencies = pyproject["project"]["dependencies"]
    package_data = pyproject["tool"]["setuptools"]["package-data"][
        "impression.devtools.reference_review.ui"
    ]

    report = build_dependency_policy_report(
        DependencyPolicyRecord(),
        declared_extras=extras.keys(),
        core_dependencies=core_dependencies,
    )

    assert report.valid
    assert "reference-review-ui" in extras
    assert any(dep.startswith("PySide6") for dep in extras["reference-review-ui"])
    assert not any(dep.startswith("PySide6") for dep in core_dependencies)
    assert "qml/icons/preview-display/*.svg" in package_data


def test_review_ui_maps_context_evidence_summary_without_absolute_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "section.json"
    artifact.write_text("{}\n")
    record = ReviewSourceModelRecord(
        "fixture/evidence",
        "Evidence Fixture",
        tmp_path / "source.py",
        evidence_bundles=(
            ReferenceEvidenceBundleRecord(
                bundle_id="bundle-a",
                evidence_kind="loft-section",
                artifacts=(
                    ReferenceEvidenceArtifactRecord("section", "application/json", artifact),
                    ReferenceEvidenceArtifactRecord("stl", "model/stl", tmp_path / "model.stl"),
                ),
            ),
        ),
    )

    summary = map_context_evidence_summary(record)

    assert all(isinstance(item, ContextEvidenceSummary) for item in summary)
    assert summary[0].display_text() == "bundle-a (loft-section): section, stl"
    assert tmp_path.as_posix() not in summary[0].display_text()


def test_review_ui_maps_artifact_evidence_rows_with_missing_status(tmp_path: Path) -> None:
    available = tmp_path / "available.stl"
    missing_optional = tmp_path / "optional.json"
    available.write_text("solid demo\nendsolid demo\n")
    record = ReviewSourceModelRecord(
        "fixture/artifacts",
        "Artifact Fixture",
        tmp_path / "source.py",
        evidence_bundles=(
            ReferenceEvidenceBundleRecord(
                bundle_id="bundle-a",
                evidence_kind="loft-section",
                artifacts=(
                    ReferenceEvidenceArtifactRecord("stl", "model/stl", available),
                    ReferenceEvidenceArtifactRecord(
                        "section",
                        "application/json",
                        missing_optional,
                        required=False,
                    ),
                ),
            ),
        ),
    )

    rows = map_artifact_evidence_rows(record)

    assert all(isinstance(row, ArtifactEvidenceRow) for row in rows)
    assert rows[0].display_text() == "stl [model/stl] available.stl - available"
    assert rows[1].status == "missing optional"
    assert format_missing_artifact_status(missing_optional, required=True) == "missing required"
    assert tmp_path.as_posix() not in rows[0].display_text()


def test_live_shell_does_not_force_pyvista_offscreen_mode() -> None:
    env = dict(os.environ)
    env.pop("PYVISTA_OFF_SCREEN", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "import impression.devtools.reference_review.ui.shell; "
                "print(os.environ.get('PYVISTA_OFF_SCREEN'))"
            ),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "None"


def test_qml_resource_layout_contains_shell_and_component_files() -> None:
    result = verify_qml_resource_layout()

    assert result.valid
    assert not result.diagnostics


def test_preview_display_control_icons_are_packaged_and_valid_xml() -> None:
    qml_root = Path("src/impression/devtools/reference_review/ui/qml")

    assert len(PREVIEW_DISPLAY_CONTROL_ICON_FILES) == 12
    for relative in PREVIEW_DISPLAY_CONTROL_ICON_FILES:
        path = qml_root / relative
        assert path.is_file(), relative
        assert "currentColor" in path.read_text()
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg")
        assert root.attrib["viewBox"] == "0 0 24 24"


def test_preview_display_control_icon_metadata_registry_resolves_packaged_paths() -> None:
    qml_root = Path("src/impression/devtools/reference_review/ui/qml")
    records = preview_display_control_icon_records()

    assert records == PREVIEW_DISPLAY_CONTROL_ICON_RECORDS
    assert {record.id for record in records} == {
        "authored-colors",
        "inspection-color",
        "lighting-flat",
        "lighting-face-normals",
        "lighting-camera",
        "object-fill",
        "object-edges",
        "triangle-wireframe",
        "bounds-grid",
        "axis-triad",
        "gradient-background",
        "polylines",
    }
    for record in records:
        assert record.tooltip
        assert record.accessible_name
        assert (qml_root / record.resource_path).is_file()
        assert preview_display_control_icon_record(record.id) == record
    with pytest.raises(KeyError, match="unknown-preview-display-control-icon"):
        preview_display_control_icon_record("missing")


def test_workbench_icon_toggle_button_states_and_command_contract() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication.instance() or QApplication([])
    icon = preview_display_control_icon_record("object-fill")
    button = WorkbenchIconToggleButton(icon)
    emitted = []
    button.commandTriggered.connect(emitted.append)
    original_size = button.size()

    assert isinstance(button, QToolButton)
    assert button.text() == ""
    assert button.toolTip() == icon.tooltip
    assert button.accessibleName() == icon.accessible_name
    style = button.styleSheet()
    assert "background: #1b2430" in style
    assert "border: 1px solid #65788e" in style
    assert "color: #dbe8f5" in style
    assert "QToolButton:checked" in style
    assert "color: #ffffff" in style
    inactive_colors = _visible_icon_colors(
        button.icon().pixmap(QSize(18, 18), QIcon.Mode.Normal, QIcon.State.Off).toImage()
    )
    active_colors = _visible_icon_colors(
        button.icon().pixmap(QSize(18, 18), QIcon.Mode.Normal, QIcon.State.On).toImage()
    )
    disabled_colors = _visible_icon_colors(
        button.icon().pixmap(QSize(18, 18), QIcon.Mode.Disabled, QIcon.State.Off).toImage()
    )
    assert inactive_colors
    assert min(color.red() + color.green() + color.blue() for color in inactive_colors) > 520
    assert min(color.red() + color.green() + color.blue() for color in active_colors) > 700
    assert min(color.red() + color.green() + color.blue() for color in disabled_colors) > 350
    button.setChecked(True)
    assert button.size() == original_size
    button.click()
    assert emitted[-1].command == "object-fill"
    button.setEnabled(False)
    button.click()
    assert len(emitted) == 1


def _visible_icon_colors(image: QImage) -> list:
    return [
        image.pixelColor(x, y)
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() > 32
    ]


def test_exclusive_icon_group_selection_model_and_component() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication.instance() or QApplication([])
    options = (
        ExclusiveIconOptionRecord("flat", "lighting-flat", "lighting-flat"),
        ExclusiveIconOptionRecord("face_normals", "lighting-face-normals", "lighting-face-normals"),
        ExclusiveIconOptionRecord("camera", "lighting-camera", "lighting-camera"),
    )
    state = ExclusiveIconGroupState(options, "face_normals")
    group = ExclusiveIconOptionGroup(state)
    selected = []
    group.optionSelected.connect(selected.append)

    assert group.option_ids() == ("flat", "face_normals", "camera")
    assert select_exclusive_icon_option(state, "camera").selected_id == "camera"
    with pytest.raises(ValueError, match="unknown-exclusive-icon-option"):
        select_exclusive_icon_option(state, "missing")
    group.select_option("camera")
    assert selected == ["camera"]
    assert group.state.selected_id == "camera"
    group.set_state(ExclusiveIconGroupState(options, "camera", enabled=False))
    group.select_option("flat")
    assert group.state.selected_id == "camera"


def test_preview_display_options_and_command_routing_are_deterministic() -> None:
    options = PreviewDisplayOptions()

    assert options.color_mode == "inspection"
    assert options.lighting_mode == "face_normals"
    assert options.show_object_fill
    assert not options.show_triangle_wireframe
    assert options.updated(show_bounds_grid=False).show_bounds_grid is False
    with pytest.raises(ValueError, match="unsupported-preview-color-mode"):
        PreviewDisplayOptions(color_mode="neon")
    with pytest.raises(ValueError, match="unsupported-preview-lighting-mode"):
        PreviewDisplayOptions(lighting_mode="studio")

    authored = route_preview_display_command(options, "authored-colors", ready=True)
    assert authored.executed
    assert authored.options.color_mode == "authored"
    lit = route_preview_display_command(authored.options, "lighting-camera", ready=True)
    assert lit.options.lighting_mode == "camera"
    toggled = route_preview_display_command(lit.options, "object-fill", ready=True)
    assert toggled.options.show_object_fill is False
    assert toggled.options.color_mode == "authored"
    disabled = route_preview_display_command(options, "object-fill", ready=False)
    assert not disabled.executed
    assert disabled.diagnostic == "preview-display-controls-disabled"
    unknown = route_preview_display_command(options, "explode", ready=True)
    assert unknown.diagnostic == "unsupported-preview-display-command"


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


def test_empty_shell_commands_give_immediate_visible_feedback() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    result = launch_workbench(offscreen=True)
    assert result.launched
    root = result.engine.rootObjects()[0]
    refresh = root.findChild(QObject, "refreshQueueButton")
    approve = root.findChild(QObject, "approveFixtureButton")
    decline = root.findChild(QObject, "declineFixtureButton")

    assert refresh is not None
    assert approve is not None
    assert decline is not None
    assert isinstance(refresh, QPushButton)
    assert isinstance(approve, QPushButton)
    assert isinstance(decline, QPushButton)
    assert approve.text() == "Approve"
    refresh.click()
    assert root.property("queueStatusText") == "No fixtures shown"
    assert root.property("selectedMessageText") == "No fixture selected."
    assert root.findChild(QObject, "sendPromptButton") is None


def test_shell_loads_fixture_file_into_selectable_queue(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    fixture_file = tmp_path / "fixtures.json"
    fixture_file.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "fixture_id": "demo/selectable",
                        "feature_name": "demo",
                        "source_path": source.name,
                        "expected_output": "demo.png",
                    }
                ]
            }
        )
    )

    records, diagnostics = shell.load_fixture_records(fixture_files=(fixture_file,))
    result = launch_workbench(
        fixture_records=records,
        fixture_diagnostics=diagnostics,
        offscreen=True,
    )
    root = result.engine.rootObjects()[0]

    assert result.launched
    assert diagnostics == ()
    assert root.property("queueStatusText") == "1 fixture shown"
    assert root.property("selectedMessageText") == "No fixture selected."
    assert not root.property("hasFixture")
    queue = root.findChild(QObject, "fixtureQueueList")
    assert isinstance(queue, QListWidget)
    queue.setCurrentRow(0)
    assert root.property("selectedMessageText") == "demo/selectable"
    assert root.property("hasFixture")


def test_shell_startup_does_not_launch_preview_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    artifact = tmp_path / "part.impress"
    artifact.write_text('{"format": "impress"}\n')
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    record = ReviewSourceModelRecord(
        "demo/no-startup-preview",
        "demo",
        source,
        artifact_paths=(artifact,),
    )
    launches = []

    class StartupProbePreviewController:
        active_identity = None

        def __init__(self, *args, **kwargs) -> None:
            pass

        def launch(self, record):
            launches.append(record)
            return SimpleNamespace(accepted=False, future=None, diagnostic="unexpected-launch")

        def close(self) -> None:
            pass

    monkeypatch.setattr(shell, "PreviewPayloadProcessController", StartupProbePreviewController)

    result = launch_workbench(fixture_records=(record,), offscreen=True)
    root = result.engine.rootObjects()[0]

    assert result.launched
    assert launches == []
    assert root.property("selectedMessageText") == "No fixture selected."
    assert not root.property("hasFixture")


def test_context_tab_shows_fixture_purpose_methodology_and_render_description(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    record = ReviewSourceModelRecord(
        fixture_id="demo/context",
        feature_name="demo",
        source_path=source,
        purpose="Validate bevel edge visibility.",
        methodology="Render with object edges enabled and inspect the silhouette.",
        render_description="A compact orange part with four clean outer vertical edges.",
    )

    result = launch_workbench(fixture_records=(record,), offscreen=True)
    root = result.engine.rootObjects()[0]
    queue = root.findChild(QObject, "fixtureQueueList")
    context = root.findChild(QObject, "selectedFixtureContextText")

    assert isinstance(queue, QListWidget)
    queue.setCurrentRow(0)
    assert isinstance(context, QLabel)
    assert "Purpose: Validate bevel edge visibility." in context.text()
    assert "Methodology: Render with object edges enabled and inspect the silhouette." in context.text()
    assert (
        "Rendered result: A compact orange part with four clean outer vertical edges."
        in context.text()
    )
    fixture = root.property("reviewFixtures")[0]
    assert fixture["purpose"] == "Validate bevel edge visibility."
    assert fixture["methodology"] == "Render with object edges enabled and inspect the silhouette."
    assert fixture["render_description"] == "A compact orange part with four clean outer vertical edges."


def test_fixture_queue_hides_approved_until_checkbox_is_checked(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    approved_source = tmp_path / "approved.py"
    declined_source = tmp_path / "declined.py"
    unreviewed_source = tmp_path / "unreviewed.py"
    for source in (approved_source, declined_source, unreviewed_source):
        source.write_text("def build():\n    return None\n")
    records = (
        ReviewSourceModelRecord(
            fixture_id="demo/approved",
            feature_name="demo",
            source_path=approved_source,
            review_status=ReferenceReviewStatus.APPROVED,
        ),
        ReviewSourceModelRecord(
            fixture_id="demo/declined",
            feature_name="demo",
            source_path=declined_source,
            review_status=ReferenceReviewStatus.DECLINED,
        ),
        ReviewSourceModelRecord(
            fixture_id="demo/unreviewed",
            feature_name="demo",
            source_path=unreviewed_source,
            review_status=ReferenceReviewStatus.UNREVIEWED,
        ),
    )

    result = launch_workbench(fixture_records=records, offscreen=True)
    root = result.engine.rootObjects()[0]
    show_approved = root.findChild(QObject, "showApprovedCheckBox")
    queue = root.findChild(QObject, "fixtureQueueList")
    badge = root.findChild(QObject, "reviewStatusBadge")

    assert isinstance(show_approved, QCheckBox)
    assert not show_approved.isChecked()
    assert isinstance(queue, QListWidget)
    assert isinstance(badge, QLabel)
    assert queue.count() == 2
    assert root.property("queueStatusText") == "2 fixtures shown"
    assert "demo/approved" not in "\n".join(queue.item(index).text() for index in range(queue.count()))
    assert root.property("reviewFixtures")[0]["status"] == "declined"
    assert root.property("reviewFixtures")[1]["status"] == "unreviewed"
    assert root.property("selectedMessageText") == "No fixture selected."
    assert badge.text() == "UNREVIEWED"
    assert "#5f6368" in badge.styleSheet()

    show_approved.setChecked(True)
    queue.setCurrentRow(0)

    assert queue.count() == 3
    labels = "\n".join(queue.item(index).text() for index in range(queue.count()))
    assert "demo/approved [approved]" in labels
    assert root.property("showApproved")
    assert root.property("selectedMessageText") == "demo/approved"
    assert badge.text() == "APPROVED"
    assert "#1f7a4d" in badge.styleSheet()


def test_decline_button_marks_fixture_file_declined(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    artifact = tmp_path / "reference-stl" / "dirty" / "demo" / "fixture.stl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("solid demo\nendsolid demo\n")
    fixture_file = tmp_path / "fixtures.json"
    fixture_file.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "fixture_id": "demo/selectable",
                        "feature_name": "demo",
                        "source_path": source.name,
                        "expected_output": "dirty STL",
                        "artifact_paths": [artifact.relative_to(tmp_path).as_posix()],
                    }
                ]
            }
        )
    )
    records, diagnostics = shell.load_fixture_records(fixture_files=(fixture_file,))
    result = launch_workbench(
        fixture_records=records,
        fixture_diagnostics=diagnostics,
        fixture_files=(fixture_file,),
        offscreen=True,
    )
    root = result.engine.rootObjects()[0]
    queue = root.findChild(QObject, "fixtureQueueList")
    decline = root.findChild(QObject, "declineFixtureButton")

    assert isinstance(queue, QListWidget)
    queue.setCurrentRow(0)
    assert isinstance(decline, QPushButton)
    decline.click()

    payload = json.loads(fixture_file.read_text())
    assert payload["fixtures"][0]["review_status"] == "declined"
    assert root.property("queueStatusText") == "demo/selectable declined"
    badge = root.findChild(QObject, "reviewStatusBadge")
    assert isinstance(badge, QLabel)
    assert badge.text() == "DECLINED"
    assert "#b42318" in badge.styleSheet()


def test_notes_editor_persists_to_selected_fixture_file_and_loads_on_selection(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    first_source = tmp_path / "first.py"
    second_source = tmp_path / "second.py"
    first_source.write_text("def build():\n    return None\n")
    second_source.write_text("def build():\n    return None\n")
    fixture_file = tmp_path / "fixtures.json"
    fixture_file.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "fixture_id": "demo/first",
                        "feature_name": "demo",
                        "source_path": first_source.name,
                        "notes": "Existing first note.",
                    },
                    {
                        "fixture_id": "demo/second",
                        "feature_name": "demo",
                        "source_path": second_source.name,
                        "notes": "Existing second note.",
                    },
                ]
            }
        )
    )
    records, diagnostics = shell.load_fixture_records(fixture_files=(fixture_file,))
    result = launch_workbench(
        fixture_records=records,
        fixture_diagnostics=diagnostics,
        fixture_files=(fixture_file,),
        offscreen=True,
    )
    root = result.engine.rootObjects()[0]
    queue = root.findChild(QObject, "fixtureQueueList")
    notes = root.findChild(QObject, "reviewNotesTextEdit")

    assert isinstance(queue, QListWidget)
    assert isinstance(notes, QTextEdit)
    assert notes.toPlainText() == ""
    queue.setCurrentRow(0)
    assert notes.toPlainText() == "Existing first note."

    notes.setPlainText("Edited first note.")
    payload = json.loads(fixture_file.read_text())
    assert payload["fixtures"][0]["notes"] == "Edited first note."
    assert payload["fixtures"][1]["notes"] == "Existing second note."

    queue.setCurrentRow(1)
    assert notes.toPlainText() == "Existing second note."
    notes.setPlainText("Edited second note.")
    payload = json.loads(fixture_file.read_text())
    assert payload["fixtures"][0]["notes"] == "Edited first note."
    assert payload["fixtures"][1]["notes"] == "Edited second note."

    queue.setCurrentRow(0)
    assert notes.toPlainText() == "Edited first note."


def test_impress_preview_edge_overlay_uses_object_edges_not_triangle_wireframe(project_root: Path) -> None:
    import pyvista as pv

    artifact = project_root / "tests/reference_review_fixtures/reference-impress/dirty/surfacebody/box.impress"
    mesh = artifact_preview._preview_mesh_for_artifact(artifact, pv)

    object_edges = artifact_preview._object_feature_edges(mesh)
    triangle_edges = mesh.extract_all_edges()

    assert object_edges.n_cells == 12
    assert triangle_edges.n_cells > object_edges.n_cells


def test_dirty_impress_fixture_launch_exposes_artifact_without_startup_render(project_root: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    fixture_file = project_root / "tests/reference_review_fixtures/dirty-impress-fixtures.json"
    records, diagnostics = shell.load_fixture_records(fixture_files=(fixture_file,))

    result = launch_workbench(
        fixture_records=records[:1],
        fixture_diagnostics=diagnostics,
        offscreen=True,
    )
    root = result.engine.rootObjects()[0]
    fixtures = root.property("reviewFixtures")

    assert result.launched
    assert fixtures[0]["artifact_display_path"] == "box.impress"
    assert fixtures[0]["artifact_preview_url"] == ""
    assert fixtures[0]["artifact_preview_status"] == "ready"


def test_diagnostic_fixture_queue_item_is_marked_non_renderable(project_root: Path) -> None:
    source = project_root / "tests/reference_review_fixtures/stl_review_sources.py"
    record = ReviewSourceModelRecord(
        "loft/csg/diagnostic",
        "loft",
        source,
        expected_output="diagnostic evidence",
        purpose="Explain a refusal.",
        methodology="Build diagnostic evidence only.",
        render_description="No STL should render.",
        artifact_paths=(),
    )
    queue = shell.FixtureQueueViewModel((record,))

    item = shell._fixture_items_for_qml(queue)[0]

    assert item["artifact_display_path"] == ""
    assert item["artifact_kind_label"] == "diagnostic evidence"
    assert item["renderable_artifact"] is False
    assert "no STL or .impress artifact to render" in item["preview_empty_message"]


def test_dirty_impress_fixture_selects_embedded_preview_surface(project_root: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    source = project_root / "tests/reference_review_fixtures/stl_review_sources.py"
    artifact = project_root / "tests/reference_review_fixtures/reference-impress/dirty/surfacebody/box.impress"
    records = (
        ReviewSourceModelRecord(
            "surfacebody/box",
            "surfacebody",
            source,
            expected_output="dirty .impress",
            artifact_paths=(artifact,),
        ),
    )

    result = launch_workbench(
        fixture_records=records,
        offscreen=True,
    )
    root = result.engine.rootObjects()[0]
    queue = root.findChild(QObject, "fixtureQueueList")

    assert result.launched
    assert isinstance(queue, QListWidget)
    assert root.property("selectedMessageText") == "No fixture selected."
    assert not root.property("hasFixture")
    queue.setCurrentRow(0)
    assert root.property("selectedMessageText") == "surfacebody/box"
    assert root.property("hasFixture")
    assert root.findChild(QObject, "openPreviewButton") is None
    assert root.findChild(QObject, "embeddedPreviewSurface") is not None
    assert root.findChild(QObject, "resetPreviewButton") is not None
    assert root.findChild(QObject, "approveFixtureButton") is not None
    assert root.findChild(QObject, "declineFixtureButton") is not None
    assert root.findChild(QObject, "sendPromptButton") is None
    assert root.findChild(QObject, "previewDisplayControlBar") is not None
    assert root.findChild(QObject, "previewDisplayColorGroup") is not None
    assert root.findChild(QObject, "previewDisplayLightingGroup") is not None
    detail_tabs = root.findChild(QObject, "reviewDetailTabs")
    assert isinstance(detail_tabs, QTabWidget)
    assert [detail_tabs.tabText(index) for index in range(detail_tabs.count())] == [
        "Context",
        "Notes",
        "Artifacts",
    ]
    assert root.property("interactivePreviewReady")


def test_preview_display_control_bar_order_and_ready_state() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication.instance() or QApplication([])
    bar = PreviewDisplayControlBar()

    assert bar.control_ids() == (
        "authored-colors",
        "inspection-color",
        "separator",
        "lighting-flat",
        "lighting-face-normals",
        "lighting-camera",
        "separator",
        "object-fill",
        "object-edges",
        "triangle-wireframe",
        "bounds-grid",
        "axis-triad",
        "gradient-background",
        "polylines",
    )
    bar.set_ready(False)
    assert not bar.findChild(QObject, "previewDisplayControl-object-fill").isEnabled()
    bar.set_ready(True)
    object_fill = bar.findChild(QObject, "previewDisplayControl-object-fill")
    assert object_fill.isEnabled()
    assert object_fill.isChecked()
    bar.set_options(PreviewDisplayOptions(show_object_fill=False, lighting_mode="camera"))
    assert not object_fill.isChecked()
    assert bar.findChild(QObject, "previewDisplayControl-lighting-camera").isChecked()


def test_live_preview_loads_impress_artifact_into_plotter(
    monkeypatch: pytest.MonkeyPatch,
    project_root: Path,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    preview = InteractiveStlPreviewLabel()
    preview.resize(360, 260)
    artifact = project_root / "tests/reference_review_fixtures/reference-impress/dirty/surfacebody/box.impress"

    class FakePlotter:
        def __init__(self) -> None:
            self.cleared = 0
            self.rendered = 0
            self.clipping_reset = 0
            self.reset = 0
            self.datasets = []

        def clear(self) -> None:
            self.cleared += 1

        def set_datasets(self, datasets) -> None:
            self.datasets.append(tuple(datasets))

        def reset_camera(self) -> None:
            self.reset += 1

        def reset_camera_clipping_range(self) -> None:
            self.clipping_reset += 1

        def render(self) -> None:
            self.rendered += 1

    fake_plotter = FakePlotter()
    preview._plotter = fake_plotter
    monkeypatch.setattr(preview, "_ensure_plotter", lambda: fake_plotter)
    preview._artifact_path = artifact
    preview._load_generation = 1
    datasets = shell._load_impress_preview_datasets(artifact)

    preview._apply_build_result(shell._LivePreviewBuildResult(1, artifact, datasets=datasets))

    assert len(preview._current_datasets) == 1
    assert fake_plotter.datasets == [tuple(preview._current_datasets)]
    assert fake_plotter.reset == 1
    assert fake_plotter.clipping_reset == 1
    assert fake_plotter.rendered == 1


def test_live_preview_schedules_impress_load_without_entering_vtk(
    tmp_path: Path,
    project_root: Path,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    preview = InteractiveStlPreviewLabel()
    artifact = project_root / "tests/reference_review_fixtures/reference-impress/dirty/surfacebody/box.impress"

    preview.set_artifact(artifact)

    assert not hasattr(preview, "_load_live_artifact")
    assert preview._status.text() == "Preview payload pending."
    stl_artifact = tmp_path / "part.stl"
    stl_artifact.write_text("solid empty\nendsolid empty\n")
    preview.set_artifact(stl_artifact)
    assert preview._status.text() == "Preview payload pending."


def test_live_preview_widget_does_not_own_worker_state() -> None:
    preview = InteractiveStlPreviewLabel()

    assert not hasattr(preview, "_executor")
    assert not hasattr(preview, "_pending_future")
    assert not hasattr(preview, "_poll_timer")


def test_preview_renderer_lifecycle_widget_reuses_single_render_surface() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widget = PreviewRendererLifecycleWidget()
    created = []

    class FakeRenderSurface(QLabel):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.closed = 0
            self.cleared = 0
            self.rendered = 0

        def close(self) -> bool:
            self.closed += 1
            return True

        def clear(self) -> None:
            self.cleared += 1

        def render(self) -> None:
            self.rendered += 1

    def renderer_factory(parent):
        surface = FakeRenderSurface(parent)
        created.append(surface)
        return surface

    first = widget.ensure_renderer(
        renderer_factory=renderer_factory,
    )
    second = widget.ensure_renderer(
        renderer_factory=renderer_factory,
    )

    assert first is second
    assert len(created) == 1
    assert widget.renderer_state.created
    assert widget._previewer is None
    widget.clear_renderer_scene("Cleared")
    assert first.cleared == 1
    assert first.rendered == 0
    assert widget._status.text() == "Cleared"
    widget.dispose_renderer()
    assert first.closed == 1
    assert widget.renderer_state.disposed


def test_preview_renderer_lifecycle_widget_default_previewer_is_disabled() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widget = PreviewRendererLifecycleWidget()

    previewer = widget._default_previewer_factory()

    assert previewer is None


def test_preview_renderer_lifecycle_widget_applies_payload_without_recreating_renderer(
    tmp_path: Path,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widget = PreviewRendererLifecycleWidget()
    record = ReviewSourceModelRecord(
        fixture_id="fixture/payload",
        feature_name="Payload",
        source_path=tmp_path / "model.py",
    )
    request = PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-payload",
        request_id=1,
        generation=7,
    )
    loaded = LoadedPreviewDataset(
        request=request,
        datasets=(
            Mesh(
                vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
                faces=np.asarray(((0, 1, 2),)),
            ),
        ),
        source_type="SurfaceBody",
    )
    payload = write_preview_payload_file(loaded, payload_dir=tmp_path)
    created = []

    class FakeRenderSurface(QLabel):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.cleared = 0
            self.datasets = []

        def clear(self) -> None:
            self.cleared += 1

        def set_datasets(self, datasets) -> None:
            self.datasets.append(tuple(datasets))

        def render(self) -> None:
            pass

    def renderer_factory(parent):
        surface = FakeRenderSurface(parent)
        created.append(surface)
        return surface

    first_state = widget.set_preview_payload(
        payload,
        renderer_factory=renderer_factory,
    )
    second_state = widget.set_preview_payload(
        payload,
        renderer_factory=renderer_factory,
    )

    assert first_state.ready
    assert second_state.ready
    assert second_state.generation == 7
    assert len(created) == 1
    assert len(created[0].datasets) == 2
    assert created[0].datasets[-1][0].n_faces == 1


def test_preview_renderer_lifecycle_widget_failure_payload_shows_payload_diagnostic() -> None:
    widget = PreviewRendererLifecycleWidget()
    request = PreviewPayloadRequest(
        owner="test",
        request_id=1,
        fixture_id="fixture",
        generation=1,
        source_path=Path("model.py"),
        entrypoint="build",
    )
    payload = PreviewPayload.failure(
        request,
        PreviewPayloadDiagnostic(
            "preview-source-load-failed",
            "difference base must be a SurfaceBody.",
            "fixture",
            "test",
            1,
            1,
        ),
    )

    state = widget.set_preview_payload(payload)

    assert not state.ready
    assert state.diagnostic == "difference base must be a SurfaceBody."
    assert widget._status.text() == "Preview unavailable: difference base must be a SurfaceBody."


def test_preview_renderer_lifecycle_widget_preserves_last_good_same_fixture_failure(
    tmp_path: Path,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widget = PreviewRendererLifecycleWidget()
    record = ReviewSourceModelRecord(
        fixture_id="fixture/last-good",
        feature_name="Last Good",
        source_path=tmp_path / "model.py",
    )
    request = PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-payload",
        request_id=1,
        generation=1,
    )
    loaded = LoadedPreviewDataset(
        request=request,
        datasets=(
            Mesh(
                vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
                faces=np.asarray(((0, 1, 2),)),
            ),
        ),
        source_type="SurfaceBody",
    )
    payload = write_preview_payload_file(loaded, payload_dir=tmp_path)

    class FakePreviewSurface(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.datasets = ()

        def clear(self) -> None:
            self.datasets = ()

        def set_display_options(self, options) -> None:
            self.display_options = options

        def set_datasets(self, datasets) -> None:
            self.datasets = tuple(datasets)

    ready_state = widget.set_preview_payload(
        payload,
        renderer_factory=lambda parent: FakePreviewSurface(parent),
    )
    failure = PreviewPayload.failure(
        PreviewPayloadRequest.from_source_record(
            record,
            owner="preview-payload",
            request_id=2,
            generation=2,
        ),
        PreviewPayloadDiagnostic(
            "preview-source-load-failed",
            "temporary build failure",
            record.fixture_id,
            "preview-payload",
            2,
            2,
        ),
    )

    failed_state = widget.set_preview_payload(failure)

    assert ready_state.ready
    assert failed_state.ready
    assert failed_state.generation == 1
    assert failed_state.diagnostic == "temporary build failure"
    assert len(widget._current_datasets) == 1
    assert widget._status.text() == "Preview stale: temporary build failure"


def test_preview_image_capture_reports_missing_render(tmp_path: Path) -> None:
    widget = PreviewRendererLifecycleWidget()

    result = widget.capture_visible_preview_image(tmp_path)

    assert not result.ok
    assert result.diagnostic == "preview-capture-missing-render"


def test_preview_image_capture_reports_busy_render_queue(tmp_path: Path) -> None:
    widget = PreviewRendererLifecycleWidget()
    widget.ensure_renderer(renderer_factory=lambda parent: QLabel("rendered", parent))
    widget._payload_state = PreviewWidgetPayloadState(generation=1, fixture_id="fixture", ready=True)
    widget._render_drain_scheduled = True

    result = widget.capture_visible_preview_image(tmp_path)

    assert not result.ok
    assert result.diagnostic == "preview-capture-busy"


def test_preview_image_capture_writes_local_png_with_metadata(tmp_path: Path) -> None:
    widget = PreviewRendererLifecycleWidget()
    widget.resize(320, 240)
    widget.ensure_renderer(renderer_factory=lambda parent: QLabel("rendered", parent))
    widget._payload_state = PreviewWidgetPayloadState(generation=3, fixture_id="fixture", ready=True)
    widget.show()

    result = widget.capture_visible_preview_image(tmp_path, basename="test preview")

    assert result.ok
    assert result.file_reference is not None
    assert result.metadata is not None
    assert result.file_reference.path.is_file()
    assert result.file_reference.mime_type == "image/png"
    assert result.file_reference.byte_count > 0
    assert result.file_reference.path.name == "test-preview.png"
    assert result.metadata.fixture_id == "fixture"
    assert result.metadata.payload_generation == 3


def test_preview_renderer_lifecycle_widget_applies_payload_to_injected_surface(
    tmp_path: Path,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widget = PreviewRendererLifecycleWidget()
    record = ReviewSourceModelRecord(
        fixture_id="fixture/software-payload",
        feature_name="Software Payload",
        source_path=tmp_path / "model.py",
    )
    request = PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-payload",
        request_id=1,
        generation=4,
    )
    loaded = LoadedPreviewDataset(
        request=request,
        datasets=(
            Mesh(
                vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
                faces=np.asarray(((0, 1, 2),)),
            ),
        ),
        source_type="SurfaceBody",
    )
    payload = write_preview_payload_file(loaded, payload_dir=tmp_path)

    class FakePreviewSurface(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.datasets = ()

        def clear(self) -> None:
            self.datasets = ()

        def set_display_options(self, options) -> None:
            self.display_options = options

        def set_datasets(self, datasets) -> None:
            self.datasets = tuple(datasets)

    state = widget.set_preview_payload(
        payload,
        renderer_factory=lambda parent: FakePreviewSurface(parent),
    )

    assert state.ready
    assert isinstance(widget._plotter, FakePreviewSurface)
    assert widget._previewer is None
    assert len(widget._plotter.datasets) == 1


def test_preview_renderer_lifecycle_widget_reports_invalid_payload(
    tmp_path: Path,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widget = PreviewRendererLifecycleWidget()
    record = ReviewSourceModelRecord(
        fixture_id="fixture/bad-payload",
        feature_name="Bad Payload",
        source_path=tmp_path / "model.py",
    )
    request = PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-payload",
        request_id=1,
        generation=1,
    )
    bad_path = tmp_path / "bad.preview-payload.json"
    bad_path.write_text('{"format":"wrong","datasets":[]}')
    from impression.devtools.reference_review import PreviewPayload

    state = widget.set_preview_payload(PreviewPayload.success(request, payload_path=bad_path))

    assert not state.ready
    assert state.diagnostic == "unsupported-preview-payload-format"
    assert widget._status.text() == "Preview unavailable: unsupported-preview-payload-format"


def test_default_renderer_policy_requires_pyvistaqt_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widget = PreviewRendererLifecycleWidget()
    monkeypatch.setattr(preview_widget, "qt_preview_supported_environment", lambda: False)

    with pytest.raises(RuntimeError, match="pyvistaqt-preview-unavailable"):
        widget.ensure_renderer()


def test_default_renderer_policy_uses_pyvistaqt_surface_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widget = PreviewRendererLifecycleWidget()
    created = []

    class FakePyVistaQtPreviewSurface(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            created.append(parent)

        def set_display_options(self, options) -> None:
            self.display_options = options

    monkeypatch.setattr(preview_widget, "qt_preview_supported_environment", lambda: True)
    monkeypatch.setattr(preview_widget, "PyVistaQtPreviewSurface", FakePyVistaQtPreviewSurface)

    renderer = widget.ensure_renderer()

    assert isinstance(renderer, FakePyVistaQtPreviewSurface)
    assert created == [widget]


def test_pyvistaqt_preview_surface_uses_lightweight_scene_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = (
        Mesh(
            vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
            faces=np.asarray(((0, 1, 2),)),
        ),
    )
    calls = []

    class FakeSceneController:
        def apply_scene(self, plotter, datasets, **kwargs):
            calls.append(("apply", tuple(datasets), kwargs))

    _apply_pyvistaqt_scene(
        FakeSceneController(),
        object(),
        datasets,
        PreviewDisplayOptions(),
        align_camera=True,
    )

    assert calls[0][2] == {
        "show_edges": False,
        "face_edges": True,
        "show_bounds": True,
        "show_axes": True,
        "align_camera": True,
        "show_object_fill": True,
        "show_polylines": True,
        "smooth_shading": False,
        "lighting": True,
        "lighting_profile": "face_normals",
        "specular": 0.0,
        "background": "#07111f",
        "background_top": "#10223a",
    }


def test_pyvistaqt_preview_scene_options_map_display_controls() -> None:
    options = PreviewDisplayOptions(
        lighting_mode="camera",
        show_object_fill=False,
        show_triangle_wireframe=True,
        show_object_edges=False,
        show_bounds_grid=False,
        show_axis_triad=False,
        show_gradient_background=False,
        show_polylines=False,
    )

    apply_options = preview_widget._preview_scene_apply_options(options, align_camera=False)

    assert apply_options.show_edges is True
    assert apply_options.face_edges is False
    assert apply_options.show_bounds is False
    assert apply_options.show_axes is False
    assert apply_options.align_camera is False
    assert apply_options.show_object_fill is False
    assert apply_options.show_polylines is False
    assert apply_options.smooth_shading is True
    assert apply_options.lighting is True
    assert apply_options.lighting_profile == "camera"
    assert apply_options.specular == 0.2
    assert apply_options.background == "#07111f"
    assert apply_options.background_top is None


def test_pyvistaqt_preview_dataset_filtering_hides_polylines() -> None:
    mesh = Mesh(
        vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        faces=np.asarray(((0, 1, 2),)),
    )
    polyline = Polyline(np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 0.0))))

    filtered = preview_widget._datasets_for_display_options(
        (mesh, polyline),
        PreviewDisplayOptions(show_polylines=False),
    )

    assert len(filtered) == 1
    assert filtered[0] is not polyline
    assert isinstance(filtered[0], Mesh)


def test_preview_render_command_records_and_queue_coalesce_by_lane(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    record = ReviewSourceModelRecord("fixture/queue", "feature", source)
    request = PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-payload",
        request_id=4,
        generation=2,
    )
    payload = PreviewPayload.success(request, payload_path=tmp_path / "payload.json")
    first_display = PreviewDisplayOptions(show_bounds_grid=True)
    second_display = PreviewDisplayOptions(show_bounds_grid=False)
    queue = PreviewRenderCommandQueue()

    first = queue.enqueue(PreviewRenderCommand.display(first_display))
    second = queue.enqueue(PreviewRenderCommand.display(second_display))
    queue.enqueue(PreviewRenderCommand.payload_ready(payload))

    assert not first.replaced
    assert second.replaced
    assert queue.state.pending_lanes == ("payload", "display")
    commands = queue.drain()
    assert [command.kind for command in commands] == [
        PreviewRenderCommandKind.PAYLOAD,
        PreviewRenderCommandKind.DISPLAY,
    ]
    assert commands[0].identity == payload.identity
    assert commands[1].display_options == second_display
    assert queue.state.pending_count == 0


def test_preview_widget_drains_latest_display_command_once() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widget = PreviewRendererLifecycleWidget()
    applied = []

    class FakePlotter(QWidget):
        def clear(self) -> None:
            pass

        def set_display_options(self, options) -> None:
            applied.append(options)

    widget.ensure_renderer(renderer_factory=lambda parent: FakePlotter(parent))
    applied.clear()

    widget.enqueue_render_command(
        PreviewRenderCommand.display(PreviewDisplayOptions(show_axis_triad=False))
    )
    widget.enqueue_render_command(
        PreviewRenderCommand.display(PreviewDisplayOptions(show_axis_triad=True))
    )
    widget._drain_preview_render_queue()

    assert applied == [PreviewDisplayOptions(show_axis_triad=True)]


def test_preview_widget_drains_one_render_command_per_turn() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widget = PreviewRendererLifecycleWidget()
    applied = []

    class FakePlotter(QWidget):
        def clear(self) -> None:
            pass

        def set_display_options(self, options) -> None:
            applied.append(options)

    widget.ensure_renderer(renderer_factory=lambda parent: FakePlotter(parent))
    applied.clear()

    widget.enqueue_render_command(PreviewRenderCommand.loading())
    widget.enqueue_render_command(
        PreviewRenderCommand.display(PreviewDisplayOptions(show_axis_triad=False))
    )

    widget._drain_preview_render_queue()

    assert widget._status.text() == "Loading preview..."
    assert applied == []

    widget._drain_preview_render_queue()

    assert applied == [PreviewDisplayOptions(show_axis_triad=False)]


def test_pyvistaqt_display_options_replace_scene_once(monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication.instance() or QApplication([])
    mesh = Mesh(
        vertices=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        faces=np.asarray(((0, 1, 2),)),
    )

    class FakeQtPreviewSurface(QWidget):
        def __init__(self, parent=None, *, config=None):
            super().__init__(parent)
            self.calls = []

        @property
        def plotter(self):
            return object()

        def replace_scene(self, datasets, *, apply_options, align_camera=True):
            self.calls.append(("replace", tuple(datasets), apply_options, align_camera))

        def set_apply_options(self, options):
            self.calls.append(("set_apply_options", options))

        def set_datasets(self, datasets, *, align_camera=True):
            self.calls.append(("set_datasets", tuple(datasets), align_camera))

        def clear(self):
            pass

    monkeypatch.setattr(preview_widget, "QtPreviewSurface", FakeQtPreviewSurface)
    surface = preview_widget.PyVistaQtPreviewSurface()

    surface.set_datasets((mesh,))
    fake_surface = surface._surface
    assert [call[0] for call in fake_surface.calls] == ["replace"]
    fake_surface.calls.clear()

    surface.set_display_options(PreviewDisplayOptions(show_triangle_wireframe=True))

    assert [call[0] for call in fake_surface.calls] == ["replace"]
    assert fake_surface.calls[0][3] is False


def test_window_preview_controller_launches_and_polls_payload(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    artifact = tmp_path / "part.impress"
    artifact.write_text('{"format": "impress"}\n')
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    record = ReviewSourceModelRecord(
        "demo/part",
        "demo",
        source,
        artifact_paths=(artifact,),
    )
    result = launch_workbench(fixture_records=(record,), offscreen=True)
    window = result.engine.rootObjects()[0]
    future: Future = Future()
    request = PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-payload",
        request_id=1,
        generation=1,
    )
    payload = PreviewPayload.success(request, payload_path=tmp_path / "payload.json")
    launched = []
    applied = []
    cleaned = []

    class FakePreviewController:
        def launch(self, record):
            launched.append(record)
            return SimpleNamespace(accepted=True, future=future, diagnostic=None)

        def handle_completion(self, envelope, handoff, diagnostic_handoff=None):
            handoff(SimpleNamespace(payload=payload))

        def cleanup_payload(self, payload, *, reason: str):
            cleaned.append((payload, reason))

        def close(self):
            pass

    window._preview_controller.close()
    window._preview_futures = []
    window._preview_controller = FakePreviewController()
    window.preview_surface.set_preview_payload = lambda payload: (
        applied.append(payload) or SimpleNamespace(ready=True, diagnostic=None)
    )

    window._load_artifact_preview(window._fixture_items[0])
    future.set_result(SimpleNamespace(ok=True))
    window._poll_preview_payloads()
    window.preview_surface._drain_preview_render_queue()
    window.preview_surface._drain_preview_render_queue()

    assert launched == [record]
    assert applied == [payload]
    assert cleaned == [(payload, "completed")]
    assert window._preview_futures == []


def test_window_preview_display_button_updates_live_surface(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    artifact = tmp_path / "part.impress"
    artifact.write_text('{"format": "impress"}\n')
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    record = ReviewSourceModelRecord(
        "demo/part",
        "demo",
        source,
        artifact_paths=(artifact,),
    )
    result = launch_workbench(fixture_records=(record,), offscreen=True)
    window = result.engine.rootObjects()[0]
    future: Future = Future()
    request = PreviewPayloadRequest.from_source_record(
        record,
        owner="preview-payload",
        request_id=7,
        generation=1,
    )
    payload = PreviewPayload.success(request, payload_path=tmp_path / "payload.json")
    applied_options = []

    class FakePreviewController:
        active_identity = payload.identity

        def launch(self, record):
            return SimpleNamespace(accepted=True, future=future, diagnostic=None, request=request)

        def handle_completion(self, envelope, handoff, diagnostic_handoff=None):
            handoff(SimpleNamespace(payload=payload))

        def cleanup_payload(self, payload, *, reason: str):
            return None

        def close(self):
            pass

    window._preview_controller.close()
    window._preview_futures = []
    window._preview_controller = FakePreviewController()

    def apply_payload(payload):
        window.preview_surface._payload_state = preview_widget.PreviewWidgetPayloadState(
            generation=payload.request.generation,
            fixture_id=payload.request.fixture_id,
            ready=True,
        )
        return window.preview_surface._payload_state

    window.preview_surface.set_preview_payload = apply_payload
    window.preview_surface.set_display_options = lambda options: applied_options.append(options)

    window._load_artifact_preview(window._fixture_items[0])
    future.set_result(SimpleNamespace(ok=True))
    window._poll_preview_payloads()
    window.preview_surface._drain_preview_render_queue()
    window.preview_surface._drain_preview_render_queue()

    button = window.findChild(QToolButton, "previewDisplayControl-triangle-wireframe")
    assert button is not None
    button.click()
    window.preview_surface._drain_preview_render_queue()

    assert applied_options[-1].show_triangle_wireframe is True
    assert window.preview_display_controls.findChild(
        QToolButton,
        "previewDisplayControl-triangle-wireframe",
    ).isChecked()


def test_window_preview_controller_uses_thread_dispatcher_not_process_pool(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    artifact = tmp_path / "part.impress"
    artifact.write_text('{"format": "impress"}\n')
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    record = ReviewSourceModelRecord(
        "demo/part",
        "demo",
        source,
        artifact_paths=(artifact,),
    )

    result = launch_workbench(fixture_records=(record,), offscreen=True)
    window = result.engine.rootObjects()[0]

    assert window._preview_controller._dispatcher is not None
    assert window._preview_controller._owns_dispatcher
    assert window._preview_controller._process_executor is None


def test_window_preview_ignores_stale_future_exception(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    artifact = tmp_path / "part.impress"
    artifact.write_text('{"format": "impress"}\n')
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    record = ReviewSourceModelRecord(
        "demo/part",
        "demo",
        source,
        artifact_paths=(artifact,),
    )
    result = launch_workbench(fixture_records=(record,), offscreen=True)
    window = result.engine.rootObjects()[0]
    stale_future: Future = Future()
    current_future: Future = Future()
    futures = [stale_future, current_future]
    enqueued = []

    class FakePreviewController:
        def __init__(self) -> None:
            self.active_identity = None
            self._request_id = 0

        def launch(self, record):
            self._request_id += 1
            generation = self._request_id
            request = SimpleNamespace(
                owner="preview-payload",
                request_id=self._request_id,
                fixture_id=record.fixture_id,
                payload={"generation": generation},
            )
            self.active_identity = (
                request.owner,
                request.request_id,
                request.fixture_id,
                generation,
            )
            return SimpleNamespace(
                accepted=True,
                future=futures[self._request_id - 1],
                request=request,
                diagnostic=None,
            )

        def handle_completion(self, envelope, handoff, diagnostic_handoff=None):
            raise AssertionError("stale exception should not reach completion handling")

        def close(self):
            pass

    window._preview_controller.close()
    window._preview_futures = []
    window._preview_future_identities = {}
    window._preview_controller = FakePreviewController()
    window.preview_surface.enqueue_render_command = lambda command: enqueued.append(command)

    window._load_artifact_preview(window._fixture_items[0])
    window._load_artifact_preview(window._fixture_items[0])
    enqueued.clear()
    stale_future.set_exception(RuntimeError("stale failure"))
    window._poll_preview_payloads()

    assert enqueued == []
    assert window._preview_futures == [current_future]


def test_window_preview_retries_latest_coalesced_request(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    artifact = tmp_path / "part.impress"
    artifact.write_text('{"format": "impress"}\n')
    source = tmp_path / "model.py"
    source.write_text("def build():\n    return None\n")
    record = ReviewSourceModelRecord(
        "demo/part",
        "demo",
        source,
        artifact_paths=(artifact,),
    )
    result = launch_workbench(fixture_records=(record,), offscreen=True)
    window = result.engine.rootObjects()[0]
    first_future: Future = Future()
    second_future: Future = Future()
    launched = []

    class CoalescingPreviewController:
        def __init__(self) -> None:
            self.active_identity = None
            self._calls = 0

        def launch(self, record):
            self._calls += 1
            launched.append(record)
            request = SimpleNamespace(
                owner="preview-payload",
                request_id=self._calls,
                fixture_id=record.fixture_id,
                payload={"generation": self._calls},
            )
            self.active_identity = (
                request.owner,
                request.request_id,
                request.fixture_id,
                self._calls,
            )
            if self._calls == 1:
                return SimpleNamespace(
                    accepted=True,
                    future=first_future,
                    request=request,
                    diagnostic=None,
                )
            if self._calls == 2:
                return SimpleNamespace(
                    accepted=False,
                    future=None,
                    request=request,
                    diagnostic="coalesced",
                )
            return SimpleNamespace(
                accepted=True,
                future=second_future,
                request=request,
                diagnostic=None,
            )

        def handle_completion(self, envelope, handoff, diagnostic_handoff=None):
            return None

        def close(self):
            pass

    window._preview_controller.close()
    window._preview_futures = []
    window._preview_future_identities = {}
    window._pending_preview_record = None
    window._preview_controller = CoalescingPreviewController()
    window._selected_index = 0

    window._load_artifact_preview(window._fixture_items[0])
    window._load_artifact_preview(window._fixture_items[0])

    assert window._preview_futures == [first_future]
    assert window._pending_preview_record is record

    first_future.set_result(SimpleNamespace(ok=False, result=None, error="stale"))
    window._poll_preview_payloads()

    assert launched == [record, record, record]
    assert window._preview_futures == [second_future]
    assert window._pending_preview_record is None


def test_live_preview_process_target_is_non_ui_module() -> None:
    assert build_impress_preview_result.__module__ == (
        "impression.devtools.reference_review.preview_payload_builder"
    )


def test_live_preview_process_builder_returns_mesh_dataset(project_root: Path) -> None:
    from concurrent.futures import ProcessPoolExecutor

    artifact = project_root / "tests/reference_review_fixtures/reference-impress/dirty/surfacebody/box.impress"

    with ProcessPoolExecutor(max_workers=1) as executor:
        result = executor.submit(build_impress_preview_result, 3, artifact).result(timeout=10)

    assert result.generation == 3
    assert result.artifact_path == artifact
    assert result.diagnostic is None
    assert len(result.datasets) == 1


def test_live_preview_ignores_stale_build_result(project_root: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    preview = InteractiveStlPreviewLabel()
    artifact = project_root / "tests/reference_review_fixtures/reference-impress/dirty/surfacebody/box.impress"
    other_artifact = project_root / "tests/reference_review_fixtures/reference-impress/dirty/surfacebody/missing.impress"
    preview._artifact_path = other_artifact
    preview._load_generation = 2

    preview._apply_build_result(shell._LivePreviewBuildResult(1, artifact, datasets=(object(),)))

    assert preview._current_datasets == []
    assert preview._status.text() == "No fixture selected."


def test_live_preview_legacy_build_result_applies_to_existing_surface(project_root: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    preview = InteractiveStlPreviewLabel()
    artifact = project_root / "tests/reference_review_fixtures/reference-impress/dirty/surfacebody/box.impress"
    preview._artifact_path = artifact
    preview._load_generation = 1
    datasets = shell._load_impress_preview_datasets(artifact)
    applied = []

    class FakePreviewSurface:
        def set_datasets(self, datasets):
            applied.append(tuple(datasets))

        def reset_camera(self):
            pass

        def reset_camera_clipping_range(self):
            pass

        def render(self):
            pass

    preview._ensure_plotter = lambda: FakePreviewSurface()
    preview._apply_build_result(shell._LivePreviewBuildResult(1, artifact, datasets=datasets))

    assert len(applied) == 1
    assert len(applied[0]) == 1


def test_window_defers_preview_load_and_ignores_stale_loads(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    artifact = tmp_path / "part.impress"
    artifact.write_text('{"format": "impress"}\n')
    record = ReviewSourceModelRecord(
        "demo/part",
        "demo",
        tmp_path / "model.py",
        artifact_paths=(artifact,),
    )
    record.source_path.write_text("def build():\n    return None\n")
    result = launch_workbench(fixture_records=(record,), offscreen=True)
    window = result.engine.rootObjects()[0]
    prepared = []
    loaded = []
    window.preview_surface.prepare_artifact = lambda path: prepared.append(path)
    window.preview_surface.set_artifact = lambda path: loaded.append(path)

    window._load_artifact_preview(window._fixture_items[0])
    first_generation = window._preview_load_generation
    window._load_artifact_preview(window._fixture_items[0])

    window._apply_preview_load(first_generation, artifact)
    window._apply_preview_load(window._preview_load_generation, artifact)

    assert prepared == [artifact, artifact]
    assert loaded == [artifact]


def test_shell_next_button_selects_fixture_record(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    records = (
        ReviewSourceModelRecord("demo/first", "demo", tmp_path / "first.py"),
        ReviewSourceModelRecord("demo/second", "demo", tmp_path / "second.py"),
    )
    for record in records:
        record.source_path.write_text("def build():\n    return None\n")

    result = launch_workbench(fixture_records=records, offscreen=True)
    root = result.engine.rootObjects()[0]
    next_button = root.findChild(QObject, "nextFixtureButton")

    assert root.property("selectedMessageText") == "No fixture selected."
    assert next_button is not None
    assert isinstance(next_button, QPushButton)
    next_button.click()
    assert root.property("selectedMessageText") == "demo/first"
    next_button.click()
    assert root.property("selectedMessageText") == "demo/second"


def test_status_badge_is_display_only_not_a_button_like_control(project_root: Path) -> None:
    badge = (
        project_root
        / "src/impression/devtools/reference_review/ui/qml/components/StatusBadge.qml"
    ).read_text()

    assert "Rectangle {" in badge
    assert "Control {" not in badge
    assert "onClicked" not in badge


def test_console_entrypoint_supports_help_and_check(capsys) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    assert shell.main(("impression-reference-review", "--help")) == 0
    help_output = capsys.readouterr().out
    assert "usage: impression-reference-review" in help_output
    assert shell.main(("impression-reference-review", "--check", "--offscreen")) == 0
    check_output = capsys.readouterr().out
    assert "launch check passed" in check_output
    assert shell._ACTIVE_LAUNCH is not None
