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
from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QTabWidget, QToolButton

from impression.mesh import Mesh
from impression.devtools.reference_review import ReviewSourceModelRecord
from impression.devtools.reference_review.ui import (
    BridgeRecord,
    BridgeRegistry,
    DependencyPolicyRecord,
    PackageResourceManifest,
    PREVIEW_DISPLAY_CONTROL_ICON_FILES,
    PREVIEW_DISPLAY_CONTROL_ICON_RECORDS,
    ExclusiveIconGroupState,
    ExclusiveIconOptionGroup,
    ExclusiveIconOptionRecord,
    PreviewDisplayControlBar,
    PreviewDisplayOptions,
    WorkbenchIconToggleButton,
    build_dependency_policy_report,
    launch_workbench,
    load_style_tokens,
    PreviewRendererLifecycleWidget,
    SoftwarePreviewSurface,
    preview_display_control_icon_record,
    preview_display_control_icon_records,
    route_preview_display_command,
    select_exclusive_icon_option,
    verify_qml_resource_layout,
)
from impression.devtools.reference_review import (
    LoadedPreviewDataset,
    PreviewPayloadRequest,
    build_impress_preview_result,
    write_preview_payload_file,
)
import impression.devtools.reference_review.ui.preview_widget as preview_widget
from impression.devtools.reference_review.ui import artifact_preview
from impression.devtools.reference_review.ui.preview_widget import (
    _SOFTWARE_PREVIEW_DEFAULT_MESH,
    _object_edge_keys,
    _project_prepared_geometry,
    _project_datasets,
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
    button.setChecked(True)
    assert button.size() == original_size
    button.click()
    assert emitted[-1].command == "object-fill"
    button.setEnabled(False)
    button.click()
    assert len(emitted) == 1


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
    send = root.findChild(QObject, "sendPromptButton")

    assert refresh is not None
    assert send is not None
    assert isinstance(refresh, QPushButton)
    refresh.click()
    assert root.property("queueStatusText") == "No fixtures loaded"
    assert root.property("selectedMessageText") == "No fixture selected."
    assert isinstance(send, QPushButton)
    send.click()
    assert root.property("codexStreamText") == "No fixture selected."


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
    assert root.property("queueStatusText") == "1 fixture loaded"
    assert root.property("selectedMessageText") == "demo/selectable"
    assert root.property("hasFixture")


def test_impress_preview_edge_overlay_uses_object_edges_not_triangle_wireframe(project_root: Path) -> None:
    import pyvista as pv

    artifact = project_root / "tests/reference_review_fixtures/reference-impress/dirty/surfacebody/box.impress"
    mesh = artifact_preview._preview_mesh_for_artifact(artifact, pv)

    object_edges = artifact_preview._object_feature_edges(mesh)
    triangle_edges = mesh.extract_all_edges()

    assert object_edges.n_cells == 12
    assert triangle_edges.n_cells > object_edges.n_cells


def test_software_preview_projects_object_edges_not_triangle_wireframe() -> None:
    mesh = Mesh(
        vertices=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3))),
    )

    scene = _project_datasets(
        (mesh,),
        width=200,
        height=200,
        rotation_x=0.0,
        rotation_y=0.0,
        zoom=1.0,
    )

    assert len(scene.faces) == 2
    assert _object_edge_keys(mesh) == {(0, 1), (1, 2), (2, 3), (0, 3)}
    assert sum(len(face.edge_segments) for face in scene.faces) == 4
    assert sum(len(face.triangle_segments) for face in scene.faces) == 0
    assert len(scene.grid_lines) == 10
    assert len(scene.axis_lines) == 3
    assert any(
        face.color.name() != _SOFTWARE_PREVIEW_DEFAULT_MESH.name()
        for face in scene.faces
    )


def test_software_preview_layer_options_are_projected_without_topology_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = Mesh(
        vertices=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3))),
    )
    calls = 0
    original_object_edge_keys = preview_widget._object_edge_keys

    def counting_object_edge_keys(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_object_edge_keys(*args, **kwargs)

    monkeypatch.setattr(preview_widget, "_object_edge_keys", counting_object_edge_keys)
    surface = SoftwarePreviewSurface()
    surface.set_datasets((mesh,))
    assert calls == 1
    assert surface._geometry is not None

    overlays_off = _project_prepared_geometry(
        surface._geometry,
        width=200,
        height=200,
        rotation_x=0.0,
        rotation_y=0.0,
        zoom=1.0,
        options=PreviewDisplayOptions(
            show_object_edges=False,
            show_bounds_grid=False,
            show_axis_triad=False,
            show_polylines=False,
        ),
    )
    wireframe = _project_prepared_geometry(
        surface._geometry,
        width=200,
        height=200,
        rotation_x=0.0,
        rotation_y=0.0,
        zoom=1.0,
        options=PreviewDisplayOptions(show_triangle_wireframe=True),
    )

    assert overlays_off.grid_lines == ()
    assert overlays_off.axis_lines == ()
    assert sum(len(face.edge_segments) for face in overlays_off.faces) == 0
    assert sum(len(face.triangle_segments) for face in wireframe.faces) == 6
    assert calls == 1


def test_software_preview_authored_color_and_lighting_modes() -> None:
    mesh = Mesh(
        vertices=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3))),
        color=(0.0, 1.0, 0.0, 1.0),
        face_colors=np.asarray(((0.5, 0.0, 0.0, 1.0), (0.0, 0.0, 0.5, 1.0))),
    )

    authored_flat = _project_datasets(
        (mesh,),
        width=200,
        height=200,
        rotation_x=0.0,
        rotation_y=0.0,
        zoom=1.0,
        options=PreviewDisplayOptions(color_mode="authored", lighting_mode="flat"),
    )
    inspection_flat = _project_datasets(
        (mesh,),
        width=200,
        height=200,
        rotation_x=0.0,
        rotation_y=0.0,
        zoom=1.0,
        options=PreviewDisplayOptions(lighting_mode="flat"),
    )
    camera_lit = _project_datasets(
        (mesh,),
        width=200,
        height=200,
        rotation_x=0.0,
        rotation_y=0.0,
        zoom=1.0,
        options=PreviewDisplayOptions(color_mode="authored", lighting_mode="camera"),
    )

    assert {face.color.name() for face in authored_flat.faces} == {"#800000", "#000080"}
    assert {face.color.name() for face in inspection_flat.faces} == {
        _SOFTWARE_PREVIEW_DEFAULT_MESH.name()
    }
    assert {face.color.name() for face in camera_lit.faces} != {"#800000", "#000080"}


def test_software_preview_caches_object_edges_between_repaints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    mesh = Mesh(
        vertices=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3))),
    )
    calls = 0
    original_object_edge_keys = preview_widget._object_edge_keys

    def counting_object_edge_keys(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_object_edge_keys(*args, **kwargs)

    monkeypatch.setattr(preview_widget, "_object_edge_keys", counting_object_edge_keys)
    surface = SoftwarePreviewSurface()
    surface.set_datasets((mesh,))

    assert calls == 1
    assert surface._geometry is not None
    _project_prepared_geometry(
        surface._geometry,
        width=200,
        height=200,
        rotation_x=0.0,
        rotation_y=0.0,
        zoom=1.0,
    )
    _project_prepared_geometry(
        surface._geometry,
        width=300,
        height=300,
        rotation_x=0.2,
        rotation_y=0.3,
        zoom=1.1,
    )

    assert calls == 1


def test_software_preview_renders_to_qt_paint_device() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication.instance() or QApplication([])
    mesh = Mesh(
        vertices=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
                (0.0, 1.0, 1.0),
            )
        ),
        faces=np.asarray(
            (
                (0, 1, 2),
                (0, 2, 3),
                (4, 6, 5),
                (4, 7, 6),
                (0, 4, 5),
                (0, 5, 1),
                (1, 5, 6),
                (1, 6, 2),
                (2, 6, 7),
                (2, 7, 3),
                (3, 7, 4),
                (3, 4, 0),
            )
        ),
    )
    surface = SoftwarePreviewSurface()
    surface.resize(360, 260)
    surface.set_datasets((mesh,))
    image = QImage(surface.size(), QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)

    surface.render(painter)
    painter.end()

    sampled_colors = {
        image.pixelColor(x, y).name()
        for y in range(0, image.height(), 10)
        for x in range(0, image.width(), 10)
    }
    assert len(sampled_colors) > 8
    assert any(color.startswith("#") and int(color[1:3], 16) > 100 for color in sampled_colors)


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

    assert result.launched
    assert root.property("selectedMessageText") == "surfacebody/box"
    assert root.property("hasFixture")
    assert root.findChild(QObject, "openPreviewButton") is None
    assert root.findChild(QObject, "embeddedPreviewSurface") is not None
    assert root.findChild(QObject, "resetPreviewButton") is not None
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


def test_preview_renderer_lifecycle_widget_uses_software_surface_by_default() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widget = PreviewRendererLifecycleWidget()

    renderer = widget.ensure_renderer()

    assert isinstance(renderer, SoftwarePreviewSurface)
    assert widget.renderer_state.created
    assert widget._previewer is None


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


def test_preview_renderer_lifecycle_widget_applies_payload_to_software_surface(
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

    state = widget.set_preview_payload(payload)

    assert state.ready
    assert isinstance(widget._plotter, SoftwarePreviewSurface)
    assert widget._previewer is None
    assert len(widget._plotter._datasets) == 1


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
    launched = []
    applied = []
    cleaned = []

    class FakePreviewController:
        def launch(self, record):
            launched.append(record)
            return SimpleNamespace(accepted=True, future=future, diagnostic=None)

        def handle_completion(self, envelope, handoff, diagnostic_handoff=None):
            handoff(SimpleNamespace(payload="payload"))

        def cleanup_payload(self, payload, *, reason: str):
            cleaned.append((payload, reason))

        def close(self):
            pass

    window._preview_controller.close()
    window._preview_futures = []
    window._preview_controller = FakePreviewController()
    window.preview_surface.set_preview_payload = lambda payload: (
        applied.append(payload) or SimpleNamespace(ready=True)
    )

    window._load_artifact_preview(window._fixture_items[0])
    future.set_result(SimpleNamespace(ok=True))
    window._poll_preview_payloads()

    assert launched == [record]
    assert applied == ["payload"]
    assert cleaned == [("payload", "completed")]
    assert window._preview_futures == []


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


def test_live_preview_legacy_build_result_uses_software_surface(project_root: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    preview = InteractiveStlPreviewLabel()
    artifact = project_root / "tests/reference_review_fixtures/reference-impress/dirty/surfacebody/box.impress"
    preview._artifact_path = artifact
    preview._load_generation = 1
    datasets = shell._load_impress_preview_datasets(artifact)

    preview._apply_build_result(shell._LivePreviewBuildResult(1, artifact, datasets=datasets))

    assert isinstance(preview._plotter, SoftwarePreviewSurface)
    assert len(preview._plotter._datasets) == 1


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

    assert root.property("selectedMessageText") == "demo/first"
    assert next_button is not None
    assert isinstance(next_button, QPushButton)
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
