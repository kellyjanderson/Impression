from __future__ import annotations

import os
import json
import tomllib
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QObject, QPointF, Qt
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
    render_stl_preview,
    verify_qml_resource_layout,
)
from impression.devtools.reference_review.ui import shell
from impression.devtools.reference_review.ui.artifact_preview import PreviewCameraState
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


def test_stl_preview_renderer_writes_png_for_artifact(project_root: Path, tmp_path: Path) -> None:
    artifact = project_root / "project/release-0.1.0a/reference-stl/dirty/surfacebody/box.stl"

    preview = render_stl_preview(artifact, cache_root=tmp_path / "previews", window_size=(240, 180))

    assert preview.diagnostic is None
    assert preview.preview_path is not None
    assert preview.preview_path.exists()
    assert preview.preview_path.suffix == ".png"
    assert preview.preview_url.startswith("file://")
    image = Image.open(preview.preview_path).convert("RGB")
    corners = (
        image.getpixel((0, 0)),
        image.getpixel((image.width - 1, 0)),
        image.getpixel((0, image.height - 1)),
        image.getpixel((image.width - 1, image.height - 1)),
    )
    pixels = image.load()
    assert all(blue > red and blue > green and max((red, green, blue)) < 80 for red, green, blue in corners)
    assert any(
        (red := pixels[x, y][0]) > 200
        and 120 <= (green := pixels[x, y][1]) <= 210
        and (blue := pixels[x, y][2]) < 150
        for y in range(image.height)
        for x in range(image.width)
    )


def test_dirty_stl_fixture_launch_exposes_artifact_without_startup_render(project_root: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    fixture_file = project_root / "tests/reference_review_fixtures/dirty-stl-fixtures.json"
    records, diagnostics = shell.load_fixture_records(fixture_files=(fixture_file,))

    result = launch_workbench(
        fixture_records=records[:1],
        fixture_diagnostics=diagnostics,
        offscreen=True,
    )
    root = result.engine.rootObjects()[0]
    fixtures = root.property("reviewFixtures")

    assert result.launched
    assert fixtures[0]["artifact_display_path"] == "anchor_shift_rectangle.stl"
    assert fixtures[0]["artifact_preview_url"] == ""
    assert fixtures[0]["artifact_preview_status"] == "ready"


def test_dirty_stl_fixture_selects_embedded_preview_surface(project_root: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    source = project_root / "tests/reference_review_fixtures/stl_review_sources.py"
    artifact = project_root / "project/release-0.1.0a/reference-stl/dirty/surfacebody/box.stl"
    records = (
        ReviewSourceModelRecord(
            "surfacebody/box",
            "surfacebody",
            source,
            expected_output="dirty STL",
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


def test_embedded_preview_uses_vtk_trackball_left_drag_rotation_sign() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    preview = InteractiveStlPreviewLabel()
    preview.resize(360, 260)

    preview._apply_pointer_delta(
        QPointF(20.0, 10.0),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPointF(200.0, 140.0),
    )

    assert preview._camera.azimuth_deg == pytest.approx(36.0)
    assert preview._camera.elevation_deg == pytest.approx(23.5)


def test_embedded_preview_vertical_orbit_is_not_clamped_to_half_turn() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    preview = InteractiveStlPreviewLabel()
    preview.resize(360, 260)

    preview._apply_pointer_delta(
        QPointF(0.0, -360.0),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPointF(180.0, -100.0),
    )

    assert preview._camera.elevation_deg == pytest.approx(190.0)


def test_preview_camera_state_preserves_full_elevation_orbit() -> None:
    assert PreviewCameraState(elevation_deg=190.0).normalized().elevation_deg == pytest.approx(190.0)
    assert PreviewCameraState(elevation_deg=450.0).normalized().elevation_deg == pytest.approx(90.0)


def test_embedded_preview_matches_vtk_trackball_modifier_modes() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    preview = InteractiveStlPreviewLabel()
    preview.resize(360, 260)

    preview._apply_pointer_delta(
        QPointF(20.0, 10.0),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
        QPointF(200.0, 140.0),
    )
    assert preview._camera.pan_x == pytest.approx(-0.08)
    assert preview._camera.pan_y == pytest.approx(0.04)
    assert preview._camera.azimuth_deg == pytest.approx(45.0)

    preview.reset_view()
    preview._apply_pointer_delta(
        QPointF(0.0, 20.0),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        QPointF(200.0, 160.0),
    )
    assert preview._camera.zoom > 1.0
    assert preview._camera.azimuth_deg == pytest.approx(45.0)

    preview.reset_view()
    preview._apply_pointer_delta(
        QPointF(10.0, 0.0),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        QPointF(210.0, 140.0),
    )
    assert preview._camera.roll_deg != pytest.approx(0.0)
    assert preview._camera.azimuth_deg == pytest.approx(45.0)

    preview.reset_view()
    preview._apply_pointer_delta(
        QPointF(20.0, 10.0),
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
        QPointF(200.0, 140.0),
    )
    assert preview._camera.pan_x == pytest.approx(-0.08)
    assert preview._camera.pan_y == pytest.approx(0.04)

    preview.reset_view()
    preview._apply_pointer_delta(
        QPointF(0.0, 20.0),
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
        QPointF(200.0, 160.0),
    )
    assert preview._camera.zoom > 1.0
    assert preview._camera.pan_x == pytest.approx(0.0)


def test_embedded_preview_schedules_frames_on_background_render_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    loops = []

    class FakeRenderLoop:
        def __init__(self, *, cache_root: Path) -> None:
            self.cache_root = cache_root
            self.requests = []
            self.cleared = False
            loops.append(self)

        def request_frame(
            self,
            artifact_path: Path,
            *,
            generation: int,
            camera,
            window_size: tuple[int, int],
        ) -> int:
            self.requests.append((artifact_path, generation, camera, window_size))
            return len(self.requests)

        def clear(self) -> int:
            self.cleared = True
            return len(self.requests) + 1

        def take_results(self) -> list[object]:
            return []

        def stop(self) -> None:
            self.cleared = True

    monkeypatch.setattr(shell, "StlPreviewRenderLoop", FakeRenderLoop)
    preview = InteractiveStlPreviewLabel()
    preview.resize(240, 180)
    artifact = tmp_path / "part.stl"
    artifact.write_text("solid empty\nendsolid empty\n")

    preview.set_artifact(artifact)
    preview._camera = preview._camera.__class__(azimuth_deg=90.0)
    preview._schedule_render()
    render_generation = preview._render_generation
    preview.set_artifact(None)

    assert len(loops) == 1
    assert [request[0] for request in loops[0].requests] == [artifact, artifact]
    assert loops[0].requests[-1][1] == render_generation
    assert loops[0].requests[-1][2].azimuth_deg == 90.0
    assert loops[0].requests[-1][3] == (360, 260)
    assert loops[0].cleared


def test_embedded_preview_applies_completed_current_generation_frames(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    preview = InteractiveStlPreviewLabel()
    intermediate_record = shell.ArtifactPreviewRecord(tmp_path / "intermediate.stl", None, "intermediate")
    latest_record = shell.ArtifactPreviewRecord(tmp_path / "latest.stl", None, "latest")
    applied = []

    class FakeRenderLoop:
        def take_results(self) -> list[object]:
            return [
                shell._PreviewRenderResult(1, 4, record=intermediate_record),
                shell._PreviewRenderResult(2, 4, record=latest_record),
            ]

    preview._render_loop = FakeRenderLoop()
    preview._render_generation = 4
    preview._latest_request_id = 20
    preview._apply_render = lambda record: applied.append(record)

    preview._poll_render()

    assert applied == [latest_record]


def test_embedded_preview_discards_previous_artifact_generation_results(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    preview = InteractiveStlPreviewLabel()
    old_record = shell.ArtifactPreviewRecord(tmp_path / "old.stl", None, "old")
    applied = []

    class FakeRenderLoop:
        def take_results(self) -> list[object]:
            return [shell._PreviewRenderResult(1, 3, record=old_record)]

    preview._render_loop = FakeRenderLoop()
    preview._render_generation = 4
    preview._apply_render = lambda record: applied.append(record)

    preview._poll_render()

    assert applied == []


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
