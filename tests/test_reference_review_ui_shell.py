from __future__ import annotations

import os
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QPushButton, QTabWidget

from impression.devtools.reference_review import ReviewSourceModelRecord
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
from impression.devtools.reference_review.ui import artifact_preview
from impression.devtools.reference_review.ui import shell
from impression.devtools.reference_review.ui.shell import InteractiveStlPreviewLabel
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
    detail_tabs = root.findChild(QObject, "reviewDetailTabs")
    assert isinstance(detail_tabs, QTabWidget)
    assert [detail_tabs.tabText(index) for index in range(detail_tabs.count())] == [
        "Context",
        "Notes",
        "Artifacts",
    ]
    assert root.property("interactivePreviewReady")


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
            self.backgrounds = []
            self.meshes = []
            self.cleared = 0
            self.rendered = 0
            self.clipping_reset = 0

        def clear(self) -> None:
            self.cleared += 1

        def set_background(self, color: str) -> None:
            self.backgrounds.append(color)

        def add_mesh(self, *args, **kwargs) -> None:
            self.meshes.append((args, kwargs))

        def reset_camera_clipping_range(self) -> None:
            self.clipping_reset += 1

        def render(self) -> None:
            self.rendered += 1

    class FakePreviewer:
        def __init__(self) -> None:
            self.reset_calls = []

        def _reset_camera(self, plotter, datasets) -> None:
            self.reset_calls.append((plotter, tuple(datasets)))

    fake_plotter = FakePlotter()
    fake_previewer = FakePreviewer()
    preview._plotter = fake_plotter
    preview._previewer = fake_previewer
    monkeypatch.setattr(preview, "_ensure_plotter", lambda: fake_plotter)
    preview._artifact_path = artifact
    preview._load_generation = 1
    datasets = shell._load_impress_preview_datasets(artifact)

    preview._apply_build_result(shell._LivePreviewBuildResult(1, artifact, datasets=datasets))

    assert fake_plotter.cleared == 1
    assert fake_plotter.backgrounds == ["#071426"]
    assert len(fake_plotter.meshes) == 2
    assert fake_plotter.meshes[0][1]["color"] == "#ffb56b"
    assert fake_plotter.meshes[1][1]["color"] == "#3d210f"
    assert len(preview._current_datasets) == 1
    assert fake_previewer.reset_calls[0][0] is fake_plotter
    assert fake_plotter.clipping_reset == 1
    assert fake_plotter.rendered == 1


def test_live_preview_schedules_impress_load_without_entering_vtk(
    monkeypatch: pytest.MonkeyPatch,
    project_root: Path,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    preview = InteractiveStlPreviewLabel()
    artifact = project_root / "tests/reference_review_fixtures/reference-impress/dirty/surfacebody/box.impress"
    submitted = []
    monkeypatch.setattr(preview, "_submit_impress_load", lambda path: submitted.append(path))

    preview.set_artifact(artifact)

    assert submitted == [artifact]
    assert preview._status.text() == "Loading preview..."


def test_live_preview_builder_executor_is_lazy() -> None:
    preview = InteractiveStlPreviewLabel()

    assert preview._executor is None


def test_live_preview_submit_uses_process_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = tmp_path / "part.impress"
    artifact.write_text('{"format": "impress"}\n')
    created = []

    class FakeFuture:
        def cancel(self) -> None:
            pass

    class FakeProcessPool:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers
            self.submissions = []
            created.append(self)

        def submit(self, fn, generation, artifact_path):
            self.submissions.append((fn, generation, artifact_path))
            return FakeFuture()

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            pass

    monkeypatch.setattr(shell, "ProcessPoolExecutor", FakeProcessPool)
    preview = InteractiveStlPreviewLabel()
    preview._load_generation = 7

    preview._submit_impress_load(artifact)

    assert len(created) == 1
    assert created[0].max_workers == 1
    assert created[0].submissions == [(shell._build_impress_preview_result, 7, artifact)]


def test_live_preview_process_builder_returns_mesh_dataset(project_root: Path) -> None:
    from concurrent.futures import ProcessPoolExecutor

    artifact = project_root / "tests/reference_review_fixtures/reference-impress/dirty/surfacebody/box.impress"

    with ProcessPoolExecutor(max_workers=1) as executor:
        result = executor.submit(shell._build_impress_preview_result, 3, artifact).result(timeout=10)

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


def test_live_preview_reports_unavailable_when_vtk_widget_cannot_start(project_root: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    preview = InteractiveStlPreviewLabel()
    artifact = project_root / "tests/reference_review_fixtures/reference-impress/dirty/surfacebody/box.impress"
    preview._artifact_path = artifact
    preview._load_generation = 1
    datasets = shell._load_impress_preview_datasets(artifact)

    preview._apply_build_result(shell._LivePreviewBuildResult(1, artifact, datasets=datasets))

    assert preview._status.text() == "Preview unavailable: RuntimeError"


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
