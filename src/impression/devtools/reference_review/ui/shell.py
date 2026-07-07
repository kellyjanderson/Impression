"""Launcher/bootstrap for the Reference Review Workbench."""

from __future__ import annotations

import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .bridge import BridgeRecord, BridgeRegistry
from .packaging import qml_resource_root
from .queue_context import FixtureQueueViewModel
from ..source_registry import (
    DiscoverySummary,
    ReviewSourceModelRecord,
    discover_source_records,
    load_source_records_from_database,
    load_source_records_from_file,
)

_ACTIVE_LAUNCH: "WorkbenchLaunchResult | None" = None
_USAGE = """usage: impression-reference-review [--fixture-file PATH] [--fixture-root PATH] [--fixture-db PATH] [--check] [--offscreen]

Launch the Impression Reference Review Workbench.

options:
  --fixture-file PATH  load review fixture records from a JSON fixture file
  --fixture-root PATH  discover review-source.json records under a fixture root
  --fixture-db PATH    load review fixture records from a SQLite review_sources table
  --check       validate that the workbench shell can load, then exit
  --offscreen   use Qt's offscreen platform plugin
  -h, --help    show this help message and exit
"""


@dataclass(frozen=True)
class WorkbenchLaunchResult:
    launched: bool
    diagnostics: tuple[str, ...] = ()
    engine: object | None = None


def _ensure_qt_app(argv: Sequence[str], *, offscreen: bool, widgets: bool = False) -> object:
    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if widgets:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            return app
        return QApplication(list(argv))
    from PySide6.QtGui import QGuiApplication

    app = QGuiApplication.instance()
    if app is not None:
        return app
    return QGuiApplication(list(argv))


class _RootObjectAdapter:
    def __init__(self, root: object) -> None:
        self._root = root

    def rootObjects(self) -> list[object]:
        return [self._root]


def launch_workbench(
    argv: Sequence[str] | None = None,
    *,
    bridges: BridgeRegistry | None = None,
    fixture_records: tuple[ReviewSourceModelRecord, ...] = (),
    fixture_diagnostics: tuple[str, ...] = (),
    qml_path: Path | None = None,
    offscreen: bool = False,
) -> WorkbenchLaunchResult:
    argv = argv or ("impression-reference-review",)
    bridges = bridges or default_bridge_registry(fixture_records)
    diagnostics = [item.code for item in bridges.diagnostics()]
    if qml_path is not None:
        return _launch_qml_workbench(
            argv,
            bridges=bridges,
            fixture_records=fixture_records,
            fixture_diagnostics=fixture_diagnostics,
            qml_path=qml_path,
            offscreen=offscreen,
            diagnostics=diagnostics,
        )
    try:
        _ensure_qt_app(argv, offscreen=offscreen, widgets=True)
    except Exception as exc:
        return WorkbenchLaunchResult(False, (f"qt-unavailable:{exc}",))

    queue = FixtureQueueViewModel(fixture_records)
    artifact_previews: dict[str, object] = {}
    artifact_preview_diagnostics: tuple[str, ...] = ()
    try:
        window = ReferenceReviewWindow(
            queue,
            artifact_previews=artifact_previews,
            startup_diagnostics=tuple(diagnostics)
            + tuple(fixture_diagnostics)
            + artifact_preview_diagnostics,
            offscreen=offscreen,
        )
    except Exception as exc:
        return WorkbenchLaunchResult(False, (f"shell-unavailable:{exc}",))
    return WorkbenchLaunchResult(True, tuple(diagnostics), _RootObjectAdapter(window))


def _launch_qml_workbench(
    argv: Sequence[str],
    *,
    bridges: BridgeRegistry,
    fixture_records: tuple[ReviewSourceModelRecord, ...],
    fixture_diagnostics: tuple[str, ...],
    qml_path: Path,
    offscreen: bool,
    diagnostics: list[str],
) -> WorkbenchLaunchResult:
    try:
        _ensure_qt_app(argv, offscreen=offscreen)
        from PySide6.QtCore import QUrl
        from PySide6.QtQml import QQmlApplicationEngine
        from PySide6.QtQuickControls2 import QQuickStyle
    except Exception as exc:
        return WorkbenchLaunchResult(False, (f"qt-unavailable:{exc}",))

    QQuickStyle.setStyle("Basic")
    engine = QQmlApplicationEngine()
    context = engine.rootContext()
    for name, record in bridges.records.items():
        context.setContextProperty(name, record.bridge)
    queue = FixtureQueueViewModel(fixture_records)
    artifact_previews: dict[str, object] = {}
    artifact_preview_diagnostics: tuple[str, ...] = ()
    context.setContextProperty(
        "startupDiagnostics",
        diagnostics + list(fixture_diagnostics) + list(artifact_preview_diagnostics),
    )
    context.setContextProperty("fixtureItems", _fixture_items_for_qml(queue, artifact_previews))
    context.setContextProperty(
        "initialQueueStatus",
        _queue_status_text(queue, fixture_diagnostics + artifact_preview_diagnostics),
    )
    path = qml_path
    if not path.is_file():
        return WorkbenchLaunchResult(False, (f"missing-qml:{path.name}",), engine)
    engine.load(QUrl.fromLocalFile(str(path)))
    if not engine.rootObjects():
        return WorkbenchLaunchResult(False, ("qml-root-not-loaded",), engine)
    return WorkbenchLaunchResult(True, tuple(diagnostics), engine)


from PySide6.QtWidgets import QLabel, QWidget


@dataclass(frozen=True)
class _LivePreviewBuildResult:
    generation: int
    artifact_path: Path
    datasets: tuple[object, ...] = ()
    diagnostic: str | None = None


def _load_impress_preview_datasets(artifact_path: Path) -> tuple[object, ...]:
    from impression.io import load_impress
    from impression.modeling import preview_tessellation_request, tessellate_surface_body

    loaded = load_impress(artifact_path)
    return tuple(
        tessellate_surface_body(body, preview_tessellation_request(require_watertight=False)).mesh
        for body in loaded.bodies
    )


def _build_impress_preview_result(generation: int, artifact_path: Path) -> _LivePreviewBuildResult:
    try:
        datasets = _load_impress_preview_datasets(artifact_path)
    except Exception as exc:
        return _LivePreviewBuildResult(generation, artifact_path, diagnostic=exc.__class__.__name__)
    return _LivePreviewBuildResult(generation, artifact_path, datasets=datasets)


def _object_feature_edges(mesh):
    return mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=True,
        non_manifold_edges=True,
        manifold_edges=False,
        feature_angle=30.0,
    )


class LiveArtifactPreviewWidget(QWidget):
    """In-window live artifact preview using the same VTK interaction model as CLI preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QVBoxLayout

        super().__init__(parent)
        self.setObjectName("embeddedPreviewSurface")
        self.setMinimumSize(360, 260)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._status = QLabel("No fixture selected.", self)
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._status)
        self._artifact_path: Path | None = None
        self._plotter = None
        self._previewer = None
        self._current_datasets = []
        self._load_generation = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reference-preview-build")
        self._pending_future: Future[_LivePreviewBuildResult] | None = None
        self._poll_timer = None

    def set_artifact(self, artifact_path: Path | None) -> None:
        self._artifact_path = artifact_path
        self._current_datasets = []
        self._load_generation += 1
        if artifact_path is None:
            self._clear_scene("No fixture selected.")
            return
        self.prepare_artifact(artifact_path)
        if artifact_path.suffix.lower() == ".impress":
            self._submit_impress_load(artifact_path)
        else:
            self._load_live_artifact(artifact_path)

    def prepare_artifact(self, artifact_path: Path | None) -> None:
        self._artifact_path = artifact_path
        self._current_datasets = []
        if artifact_path is None:
            self._clear_scene("No fixture selected.")
            return
        self._status.setText("Loading preview...")
        self._status.show()

    def reset_view(self) -> None:
        if self._plotter is None:
            return
        if self._previewer is not None and self._current_datasets:
            self._previewer._reset_camera(self._plotter, self._current_datasets)
        else:
            self._plotter.reset_camera()
        self._plotter.reset_camera_clipping_range()
        self._plotter.render()

    def shutdown(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()
        if self._pending_future is not None:
            self._pending_future.cancel()
            self._pending_future = None
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None

    def _submit_impress_load(self, artifact_path: Path) -> None:
        from PySide6.QtCore import QTimer

        if self._pending_future is not None:
            self._pending_future.cancel()
        generation = self._load_generation
        self._pending_future = self._executor.submit(_build_impress_preview_result, generation, artifact_path)
        if self._poll_timer is None:
            self._poll_timer = QTimer(self)
            self._poll_timer.setInterval(30)
            self._poll_timer.timeout.connect(self._poll_pending_load)
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    def _poll_pending_load(self) -> None:
        future = self._pending_future
        if future is None:
            if self._poll_timer is not None:
                self._poll_timer.stop()
            return
        if not future.done():
            return
        self._pending_future = None
        if self._poll_timer is not None:
            self._poll_timer.stop()
        try:
            result = future.result()
        except Exception as exc:
            self._clear_scene(f"Preview unavailable: {exc.__class__.__name__}")
            return
        self._apply_build_result(result)

    def _apply_build_result(self, result: _LivePreviewBuildResult) -> None:
        if result.generation != self._load_generation or result.artifact_path != self._artifact_path:
            return
        if result.diagnostic is not None:
            self._clear_scene(f"Preview unavailable: {result.diagnostic}")
            return
        try:
            plotter = self._ensure_plotter()
            self._status.hide()
            plotter.clear()
            plotter.set_background("#071426")
            self._current_datasets = list(result.datasets)
            self._add_impress_datasets(plotter, self._current_datasets)
            self.reset_view()
        except Exception as exc:
            self._clear_scene(f"Preview unavailable: {exc.__class__.__name__}")

    def _load_live_artifact(self, artifact_path: Path) -> None:
        try:
            plotter = self._ensure_plotter()
            self._status.hide()
            plotter.clear()
            plotter.set_background("#071426")
            if artifact_path.suffix.lower() == ".impress":
                self._submit_impress_load(artifact_path)
            elif artifact_path.suffix.lower() == ".stl":
                import pyvista as pv

                mesh = pv.read(artifact_path)
                plotter.add_mesh(mesh, color="#ffb56b", smooth_shading=False, show_edges=False)
                edges = _object_feature_edges(mesh)
                if edges.n_cells > 0:
                    plotter.add_mesh(edges, color="#3d210f", line_width=2)
                plotter.reset_camera()
                plotter.reset_camera_clipping_range()
                plotter.render()
            else:
                self._clear_scene("Unsupported artifact type.")
        except Exception as exc:
            self._clear_scene(f"Preview unavailable: {exc.__class__.__name__}")

    def _ensure_plotter(self):
        if self._plotter is not None:
            return self._plotter
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            raise RuntimeError("live-preview-unavailable-offscreen")
        from pyvistaqt import QtInteractor
        from rich.console import Console
        from impression.preview import PyVistaPreviewer

        self._plotter = QtInteractor(self)
        self._layout.addWidget(self._plotter, 1)
        self._previewer = PyVistaPreviewer(console=Console())
        self._previewer._configure_plotter(self._plotter, show_bounds=False, show_axes=True)
        return self._plotter

    def _add_impress_datasets(self, plotter, datasets: list[object]) -> None:
        from impression.mesh import mesh_to_pyvista

        for index, mesh in enumerate(datasets):
            pv_mesh = mesh_to_pyvista(mesh)
            plotter.add_mesh(
                pv_mesh,
                name=f"artifact-{index}",
                color="#ffb56b",
                smooth_shading=False,
                show_edges=False,
            )
            edges = _object_feature_edges(pv_mesh)
            if edges.n_cells > 0:
                plotter.add_mesh(edges, name=f"artifact-{index}-edges", color="#3d210f", line_width=2)

    def _clear_scene(self, message: str) -> None:
        if self._plotter is not None:
            self._plotter.clear()
            self._plotter.render()
        self._status.setText(message)
        self._status.show()


InteractiveStlPreviewLabel = LiveArtifactPreviewWidget


class ReferenceReviewWindow(QWidget):
    """Widget-hosted review shell with an embedded PyVista interactor."""

    def __init__(
        self,
        queue: FixtureQueueViewModel,
        *,
        artifact_previews: dict[str, object],
        startup_diagnostics: tuple[str, ...],
        offscreen: bool,
    ) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QPushButton,
            QSplitter,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )

        super().__init__()
        self.setWindowTitle("Reference Review Workbench")
        self.resize(1180, 760)
        self.setMinimumSize(880, 560)
        self.queue = queue
        self.artifact_previews = artifact_previews
        self.startup_diagnostics = startup_diagnostics
        self._offscreen = offscreen
        self._interactive_preview_ready = True
        self._selected_index = queue.selected_index if queue.selected_index is not None else -1
        self._fixture_items = _fixture_items_for_qml(queue, artifact_previews)
        self._preview_load_generation = 0

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        sidebar = QFrame()
        sidebar.setMinimumWidth(220)
        sidebar.setMaximumWidth(360)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_header = QHBoxLayout()
        title = QLabel("Queue")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("refreshQueueButton")
        sidebar_header.addWidget(title, 1)
        sidebar_header.addWidget(self.refresh_button)
        sidebar_layout.addLayout(sidebar_header)
        self.queue_status = QLabel(_queue_status_text(queue, startup_diagnostics))
        self.queue_status.setStyleSheet("background: #d8f0f2; border: 1px solid #89bec4; padding: 6px;")
        sidebar_layout.addWidget(self.queue_status)
        self.list_widget = QListWidget()
        sidebar_layout.addWidget(self.list_widget, 1)
        splitter.addWidget(sidebar)

        main = QWidget()
        main_layout = QGridLayout(main)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setHorizontalSpacing(12)
        main_layout.setVerticalSpacing(12)
        header = QHBoxLayout()
        selected_title = QLabel("Selected Fixture")
        selected_title.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.ready_badge = QLabel("Ready" if not startup_diagnostics else "Diagnostics")
        self.ready_badge.setStyleSheet("background: #d8f0f2; border: 1px solid #89bec4; padding: 6px 18px;")
        self.previous_button = QPushButton("Previous")
        self.previous_button.setObjectName("previousFixtureButton")
        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("nextFixtureButton")
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.setObjectName("resetPreviewButton")
        header.addWidget(selected_title, 1)
        header.addWidget(self.ready_badge)
        header.addWidget(self.reset_view_button)
        header.addWidget(self.previous_button)
        header.addWidget(self.next_button)
        main_layout.addLayout(header, 0, 0, 1, 2)

        self.preview_frame = QFrame()
        self.preview_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.preview_frame.setMinimumSize(360, 260)
        self.preview_layout = QVBoxLayout(self.preview_frame)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_surface = InteractiveStlPreviewLabel(self.preview_frame)
        self.preview_layout.addWidget(self.preview_surface)
        main_layout.addWidget(self.preview_frame, 1, 0)

        self.detail_tabs = QTabWidget()
        self.detail_tabs.setObjectName("reviewDetailTabs")
        self.detail_tabs.setMinimumSize(320, 260)

        context = QWidget()
        context_layout = QVBoxLayout(context)
        context_title = QLabel("Context")
        context_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.context_text = QLabel("")
        self.context_text.setWordWrap(True)
        context_layout.addWidget(context_title)
        context_layout.addWidget(self.context_text, 1)

        notes = QWidget()
        notes_layout = QVBoxLayout(notes)
        notes_title = QLabel("Notes")
        notes_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.notes = QTextEdit()
        self.notes.setPlaceholderText("Review notes")
        notes_layout.addWidget(notes_title)
        notes_layout.addWidget(self.notes, 1)

        artifacts = QWidget()
        artifacts_layout = QVBoxLayout(artifacts)
        artifact_title = QLabel("Artifacts")
        artifact_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.artifact_thumb = QLabel("")
        self.artifact_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.artifact_thumb.setMinimumHeight(180)
        artifacts_layout.addWidget(artifact_title)
        artifacts_layout.addWidget(self.artifact_thumb, 1)

        self.detail_tabs.addTab(context, "Context")
        self.detail_tabs.addTab(notes, "Notes")
        self.detail_tabs.addTab(artifacts, "Artifacts")
        main_layout.addWidget(self.detail_tabs, 1, 1)

        codex = QFrame()
        codex.setFrameShape(QFrame.Shape.StyledPanel)
        codex_layout = QVBoxLayout(codex)
        codex_title = QLabel("Codex")
        codex_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.codex_stream = QLabel("No active stream.")
        prompt_row = QHBoxLayout()
        self.prompt = QTextEdit()
        self.prompt.setFixedHeight(40)
        self.prompt.setPlaceholderText("Fixture-scoped prompt")
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("sendPromptButton")
        prompt_row.addWidget(self.prompt, 1)
        prompt_row.addWidget(self.send_button)
        codex_layout.addWidget(codex_title)
        codex_layout.addWidget(self.codex_stream, 1)
        codex_layout.addLayout(prompt_row)
        main_layout.addWidget(codex, 2, 0, 1, 2)
        main_layout.setColumnStretch(0, 3)
        main_layout.setColumnStretch(1, 2)
        main_layout.setRowStretch(1, 3)
        main_layout.setRowStretch(2, 2)
        splitter.addWidget(main)
        splitter.setStretchFactor(1, 1)

        self._populate_queue_items()

        self.refresh_button.clicked.connect(self._refresh_queue)
        self.reset_view_button.clicked.connect(self.preview_surface.reset_view)
        self.previous_button.clicked.connect(self._previous_fixture)
        self.next_button.clicked.connect(self._next_fixture)
        self.send_button.clicked.connect(self._send_prompt)
        self.list_widget.currentRowChanged.connect(self._select_index)
        self.show()
        self.raise_()
        self.activateWindow()
        if self._selected_index >= 0:
            self._select_initial_fixture(self._selected_index)
        else:
            self._sync_properties()

    def _select_initial_fixture(self, index: int) -> None:
        if self._offscreen:
            self.list_widget.setCurrentRow(index)
            return
        from PySide6.QtCore import QTimer

        self._sync_properties()
        QTimer.singleShot(75, lambda: self._apply_initial_selection(index))

    def _apply_initial_selection(self, index: int) -> None:
        if self.list_widget.count() == 0:
            return
        self.list_widget.setCurrentRow(max(0, min(index, self.list_widget.count() - 1)))

    def _populate_queue_items(self) -> None:
        from PySide6.QtWidgets import QListWidgetItem

        self.list_widget.clear()
        for item in self._fixture_items:
            label = f"{item['fixture_id']}\n{item['artifact_display_path'] or item['source_display_path']}"
            widget_item = QListWidgetItem(label)
            widget_item.setData(256, item["fixture_id"])
            self.list_widget.addItem(widget_item)

    def _refresh_queue(self) -> None:
        if self._fixture_items:
            suffix = "s" if len(self._fixture_items) != 1 else ""
            self.queue_status.setText(f"{len(self._fixture_items)} fixture{suffix} loaded")
            self._select_index(max(self._selected_index, 0))
        else:
            self.queue_status.setText("No fixtures loaded")
            self._select_index(-1)

    def _previous_fixture(self) -> None:
        self.list_widget.setCurrentRow(max(0, self._selected_index - 1))

    def _next_fixture(self) -> None:
        self.list_widget.setCurrentRow(min(len(self._fixture_items) - 1, self._selected_index + 1))

    def _send_prompt(self) -> None:
        if self._selected_index < 0:
            self.codex_stream.setText("No fixture selected.")
            self.setProperty("codexStreamText", "No fixture selected.")
        else:
            self.codex_stream.setText("Request queued.")
            self.setProperty("codexStreamText", "Request queued.")

    def _select_index(self, index: int) -> None:
        if index < 0 or index >= len(self._fixture_items):
            self._selected_index = -1
            self.preview_surface.set_artifact(None)
            self.context_text.setText("No fixture context loaded.")
            self.artifact_thumb.setText("")
            self._sync_properties()
            return
        self._selected_index = index
        item = self._fixture_items[index]
        self.context_text.setText(
            f"{item['fixture_id']}\n\n"
            f"Source: {item['source_display_path']}\n"
            f"Expected: {item['expected_output'] or 'not declared'}\n"
            f"Artifact: {item['artifact_display_path']}"
        )
        self._load_artifact_preview(item)
        self._sync_properties()

    def _load_artifact_preview(self, item: dict[str, object]) -> None:
        from PySide6.QtCore import QTimer

        record = self.queue.records[self._selected_index]
        artifact_path = record.artifact_paths[0] if record.artifact_paths else None
        live_artifact = artifact_path if artifact_path and artifact_path.is_file() else None
        self._preview_load_generation += 1
        generation = self._preview_load_generation
        self.artifact_thumb.setText(str(item.get("artifact_display_path", "")))
        self.preview_surface.prepare_artifact(live_artifact)
        delay_ms = 0 if self._offscreen else 75
        QTimer.singleShot(delay_ms, lambda: self._apply_preview_load(generation, live_artifact))

    def _apply_preview_load(self, generation: int, artifact_path: Path | None) -> None:
        if generation != self._preview_load_generation:
            return
        self.preview_surface.set_artifact(artifact_path)

    def _sync_properties(self) -> None:
        has_fixture = self._selected_index >= 0
        selected = self._fixture_items[self._selected_index] if has_fixture else None
        self.setProperty("reviewFixtures", self._fixture_items)
        self.setProperty("queueStatusText", self.queue_status.text())
        self.setProperty("selectedMessageText", selected["fixture_id"] if selected else "No fixture selected.")
        self.setProperty("hasFixture", has_fixture)
        self.setProperty("codexStreamText", self.codex_stream.text())
        self.setProperty("interactivePreviewReady", self._interactive_preview_ready)

    def closeEvent(self, event) -> None:
        self.preview_surface.shutdown()
        super().closeEvent(event)


def default_bridge_registry(
    fixture_records: tuple[ReviewSourceModelRecord, ...] = (),
) -> BridgeRegistry:
    from PySide6.QtCore import QObject

    registry = BridgeRegistry()
    for name in ("queueBridge", "selectionBridge", "codexBridge", "notesBridge", "artifactsBridge"):
        registry = registry.register(BridgeRecord(name=name, bridge=QObject()))
    return registry


def load_fixture_records(
    *,
    fixture_files: tuple[Path, ...] = (),
    fixture_roots: tuple[Path, ...] = (),
    fixture_databases: tuple[Path, ...] = (),
) -> tuple[tuple[ReviewSourceModelRecord, ...], tuple[str, ...]]:
    summaries: list[DiscoverySummary] = []
    if fixture_roots:
        summaries.append(discover_source_records(fixture_roots))
    summaries.extend(load_source_records_from_file(path) for path in fixture_files)
    summaries.extend(load_source_records_from_database(path) for path in fixture_databases)
    records: list[ReviewSourceModelRecord] = []
    diagnostics: list[str] = []
    for summary in summaries:
        records.extend(item.record for item in summary.valid_items)
        diagnostics.extend(diagnostic.code for diagnostic in summary.diagnostics)
        for item in summary.items:
            diagnostics.extend(diagnostic.code for diagnostic in item.validation.diagnostics)
    return tuple(records), tuple(diagnostics)


def _fixture_items_for_qml(
    queue: FixtureQueueViewModel,
    artifact_previews: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    artifact_previews = artifact_previews or {}
    items: list[dict[str, object]] = []
    for item in queue.items:
        preview = artifact_previews.get(item.fixture_id)
        items.append(
            {
                "fixture_id": item.fixture_id,
                "feature_name": item.feature_name,
                "source_display_path": item.source_display_path,
                "expected_output": item.expected_output or "",
                "artifact_display_path": item.artifact_display_path or "",
                "artifact_preview_url": getattr(preview, "preview_url", "") if preview is not None else "",
                "artifact_preview_status": getattr(preview, "diagnostic", None)
                if preview is not None and getattr(preview, "diagnostic", None)
                else "ready",
                "status": item.status,
            }
        )
    return items


def _queue_status_text(queue: FixtureQueueViewModel, diagnostics: tuple[str, ...]) -> str:
    if queue.items:
        return f"{len(queue.items)} fixture{'s' if len(queue.items) != 1 else ''} loaded"
    if diagnostics:
        return "No valid fixtures loaded"
    return "No fixture file loaded"


def _parse_args(args: Sequence[str]) -> tuple[list[Path], list[Path], list[Path], set[str], str | None]:
    fixture_files: list[Path] = []
    fixture_roots: list[Path] = []
    fixture_databases: list[Path] = []
    flags: set[str] = set()
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in {"--check", "--offscreen"}:
            flags.add(arg)
            index += 1
            continue
        if arg in {"--fixture-file", "--fixture-root", "--fixture-db"}:
            if index + 1 >= len(args):
                return fixture_files, fixture_roots, fixture_databases, flags, f"missing value for {arg}"
            path = Path(args[index + 1])
            if arg == "--fixture-file":
                fixture_files.append(path)
            elif arg == "--fixture-root":
                fixture_roots.append(path)
            else:
                fixture_databases.append(path)
            index += 2
            continue
        return fixture_files, fixture_roots, fixture_databases, flags, f"unknown argument: {arg}"
    return fixture_files, fixture_roots, fixture_databases, flags, None


def main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_LAUNCH

    argv = tuple(argv or sys.argv)
    args = argv[1:]
    if any(arg in {"-h", "--help"} for arg in args):
        print(_USAGE.strip())
        return 0
    fixture_files, fixture_roots, fixture_databases, flags, error = _parse_args(args)
    if error is not None:
        print(error, file=sys.stderr)
        print(_USAGE.strip(), file=sys.stderr)
        return 2
    fixture_records, fixture_diagnostics = load_fixture_records(
        fixture_files=tuple(fixture_files),
        fixture_roots=tuple(fixture_roots),
        fixture_databases=tuple(fixture_databases),
    )
    result = launch_workbench(
        argv,
        bridges=default_bridge_registry(fixture_records),
        fixture_records=fixture_records,
        fixture_diagnostics=fixture_diagnostics,
        offscreen="--offscreen" in flags,
    )
    if not result.launched:
        for diagnostic in result.diagnostics:
            print(diagnostic, file=sys.stderr)
        return 1
    _ACTIVE_LAUNCH = result
    if "--check" in flags:
        print("Reference Review Workbench launch check passed")
        return 0
    from PySide6.QtWidgets import QApplication

    return QApplication.instance().exec()


if __name__ == "__main__":
    raise SystemExit(main())
