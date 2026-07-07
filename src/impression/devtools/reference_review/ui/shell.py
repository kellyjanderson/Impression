"""Launcher/bootstrap for the Reference Review Workbench."""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Callable, Sequence

from .artifact_preview import (
    ArtifactPreviewRecord,
    ArtifactPreviewRenderer,
    PreviewCameraState,
    render_stl_preview,
)
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
    artifact_previews: dict[str, ArtifactPreviewRecord] = {}
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
    artifact_previews = _artifact_previews_for_records(fixture_records)
    artifact_preview_diagnostics = tuple(
        preview.diagnostic
        for preview in artifact_previews.values()
        if preview.diagnostic is not None
    )
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
class _PreviewRenderRequest:
    request_id: int
    generation: int
    artifact_path: Path
    camera: PreviewCameraState
    window_size: tuple[int, int]


@dataclass(frozen=True)
class _PreviewRenderResult:
    request_id: int
    generation: int
    record: ArtifactPreviewRecord | None = None
    error: str | None = None


class StlPreviewRenderLoop:
    """Single-owner background render loop for settled artifact preview snapshots."""

    def __init__(
        self,
        *,
        cache_root: Path,
        fps: int = 6,
        renderer: Callable[..., ArtifactPreviewRecord] | None = None,
        renderer_factory: Callable[..., ArtifactPreviewRenderer] = ArtifactPreviewRenderer,
    ) -> None:
        self._cache_root = cache_root
        self._renderer = renderer
        self._renderer_factory = renderer_factory
        self._owned_renderer: ArtifactPreviewRenderer | None = None
        self._frame_interval = 1.0 / fps
        self._condition = threading.Condition()
        self._results: SimpleQueue[_PreviewRenderResult] = SimpleQueue()
        self._latest_request: _PreviewRenderRequest | None = None
        self._latest_request_id = 0
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run,
            name="stl-preview-render-loop",
            daemon=True,
        )
        self._thread.start()

    def request_frame(
        self,
        artifact_path: Path,
        *,
        generation: int,
        camera: PreviewCameraState,
        window_size: tuple[int, int],
    ) -> int:
        with self._condition:
            self._latest_request_id += 1
            request = _PreviewRenderRequest(
                self._latest_request_id,
                generation,
                artifact_path,
                camera.normalized(),
                window_size,
            )
            self._latest_request = request
            self._condition.notify()
            return request.request_id

    def clear(self) -> int:
        with self._condition:
            self._latest_request_id += 1
            self._latest_request = None
            self._condition.notify()
            return self._latest_request_id

    def take_results(self) -> list[_PreviewRenderResult]:
        results: list[_PreviewRenderResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except Empty:
                return results

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        next_frame_at = 0.0
        try:
            while True:
                with self._condition:
                    while not self._stopping and self._latest_request is None:
                        self._condition.wait()
                    if self._stopping:
                        return
                    request = self._latest_request
                    self._latest_request = None
                if request is None:
                    continue
                sleep_for = next_frame_at - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                next_frame_at = time.monotonic() + self._frame_interval
                try:
                    record = self._render_request(request)
                except Exception as exc:
                    self._results.put(
                        _PreviewRenderResult(
                            request.request_id,
                            request.generation,
                            error=exc.__class__.__name__,
                        )
                    )
                else:
                    self._results.put(
                        _PreviewRenderResult(request.request_id, request.generation, record=record)
                    )
        finally:
            self._close_owned_renderer()

    def _render_request(self, request: _PreviewRenderRequest) -> ArtifactPreviewRecord:
        if self._renderer is not None:
            return self._renderer(
                request.artifact_path,
                cache_root=self._cache_root,
                window_size=request.window_size,
                camera=request.camera,
            )
        if self._owned_renderer is None:
            self._owned_renderer = self._renderer_factory(cache_root=self._cache_root)
        return self._owned_renderer.render(
            request.artifact_path,
            window_size=request.window_size,
            camera=request.camera,
        )

    def _close_owned_renderer(self) -> None:
        if self._owned_renderer is not None:
            self._owned_renderer.close()
            self._owned_renderer = None


class InteractiveStlPreviewLabel(QLabel):
    """In-window STL preview surface with CLI-like mouse controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        from PySide6.QtCore import Qt

        super().__init__(parent)
        self.setObjectName("embeddedPreviewSurface")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 260)
        self.setText("No fixture selected.")
        self.setMouseTracking(True)
        self._artifact_path: Path | None = None
        self._camera = PreviewCameraState()
        self._last_pos = None
        self._cache_root = Path(".cache/reference-review/stl-previews")
        self._render_loop: StlPreviewRenderLoop | None = None
        self._latest_request_id = 0
        self._render_generation = 0
        self._render_poll_timer = None
        self._base_pixmap = None

    def set_artifact(self, artifact_path: Path | None) -> None:
        self._artifact_path = artifact_path
        self._render_generation += 1
        self._camera = PreviewCameraState()
        self._schedule_render()

    def reset_view(self) -> None:
        self._camera = PreviewCameraState()
        self._schedule_render()

    def mousePressEvent(self, event) -> None:
        self._last_pos = event.position()

    def mouseMoveEvent(self, event) -> None:
        if self._artifact_path is None or self._last_pos is None:
            return
        delta = event.position() - self._last_pos
        self._last_pos = event.position()
        self._apply_pointer_delta(delta, event.buttons(), event.modifiers(), event.position())

    def _apply_pointer_delta(self, delta, buttons, modifiers, position) -> None:
        from PySide6.QtCore import Qt

        shift_pressed = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        ctrl_pressed = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        if buttons & Qt.MouseButton.LeftButton and ctrl_pressed and shift_pressed:
            self._dolly_by_pixels(delta.y())
        elif buttons & Qt.MouseButton.LeftButton and shift_pressed:
            self._pan_by_pixels(delta.x(), delta.y())
        elif buttons & Qt.MouseButton.LeftButton and ctrl_pressed:
            self._spin_to_pointer(position, delta)
        elif buttons & Qt.MouseButton.LeftButton:
            self._camera = PreviewCameraState(
                azimuth_deg=self._camera.azimuth_deg - delta.x() * 0.45,
                elevation_deg=self._camera.elevation_deg - delta.y() * 0.45,
                roll_deg=self._camera.roll_deg,
                zoom=self._camera.zoom,
                pan_x=self._camera.pan_x,
                pan_y=self._camera.pan_y,
            ).normalized()
        elif buttons & Qt.MouseButton.MiddleButton:
            self._pan_by_pixels(delta.x(), delta.y())
        elif buttons & Qt.MouseButton.RightButton and not shift_pressed:
            self._dolly_by_pixels(delta.y())

    def _pan_by_pixels(self, dx: float, dy: float) -> None:
        self._camera = PreviewCameraState(
            azimuth_deg=self._camera.azimuth_deg,
            elevation_deg=self._camera.elevation_deg,
            roll_deg=self._camera.roll_deg,
            zoom=self._camera.zoom,
            pan_x=self._camera.pan_x - dx * 0.004,
            pan_y=self._camera.pan_y + dy * 0.004,
        ).normalized()

    def _dolly_by_pixels(self, dy: float) -> None:
        factor = 1.1 ** (20.0 * dy / max(float(self.height()), 1.0))
        self._camera = PreviewCameraState(
            azimuth_deg=self._camera.azimuth_deg,
            elevation_deg=self._camera.elevation_deg,
            roll_deg=self._camera.roll_deg,
            zoom=self._camera.zoom * factor,
            pan_x=self._camera.pan_x,
            pan_y=self._camera.pan_y,
        ).normalized()

    def _spin_to_pointer(self, position, delta) -> None:
        import math

        center_x = self.width() / 2.0
        center_y = self.height() / 2.0
        old_x = position.x() - delta.x()
        old_y = position.y() - delta.y()
        new_angle = math.degrees(math.atan2(position.y() - center_y, position.x() - center_x))
        old_angle = math.degrees(math.atan2(old_y - center_y, old_x - center_x))
        self._camera = PreviewCameraState(
            azimuth_deg=self._camera.azimuth_deg,
            elevation_deg=self._camera.elevation_deg,
            roll_deg=self._camera.roll_deg + new_angle - old_angle,
            zoom=self._camera.zoom,
            pan_x=self._camera.pan_x,
            pan_y=self._camera.pan_y,
        ).normalized()

    def mouseReleaseEvent(self, event) -> None:
        self._last_pos = None
        self._schedule_render()

    def wheelEvent(self, event) -> None:
        if self._artifact_path is None:
            return
        factor = 1.12 if event.angleDelta().y() > 0 else 1.0 / 1.12
        self._camera = PreviewCameraState(
            azimuth_deg=self._camera.azimuth_deg,
            elevation_deg=self._camera.elevation_deg,
            roll_deg=self._camera.roll_deg,
            zoom=self._camera.zoom * factor,
            pan_x=self._camera.pan_x,
            pan_y=self._camera.pan_y,
        ).normalized()
        self._schedule_render()

    def shutdown(self) -> None:
        if self._render_loop is not None:
            self._render_loop.stop()
            self._render_loop = None

    def _schedule_render(self) -> None:
        from PySide6.QtCore import QTimer

        if self._artifact_path is None:
            if self._render_loop is not None:
                self._latest_request_id = self._render_loop.clear()
            self._base_pixmap = None
            self.clear()
            self.setText("No fixture selected.")
            return
        size = (max(self.width(), 360), max(self.height(), 260))
        if self._base_pixmap is None:
            self.setText("Rendering preview...")
        self._latest_request_id = self._ensure_render_loop().request_frame(
            self._artifact_path,
            generation=self._render_generation,
            camera=self._camera,
            window_size=size,
        )
        if self._render_poll_timer is None:
            self._render_poll_timer = QTimer(self)
            self._render_poll_timer.setInterval(33)
            self._render_poll_timer.timeout.connect(self._poll_render)
        if not self._render_poll_timer.isActive():
            self._render_poll_timer.start()

    def _poll_render(self) -> None:
        if self._render_loop is None:
            return
        latest_result: _PreviewRenderResult | None = None
        handled_result = False
        for result in self._render_loop.take_results():
            handled_result = True
            if result.generation != self._render_generation:
                continue
            latest_result = result
        if latest_result is not None:
            result = latest_result
            if result.error is not None:
                self.clear()
                self.setText(f"Preview unavailable: {result.error}")
            elif result.record is not None:
                self._apply_render(result.record)
        if self._render_poll_timer is not None and handled_result and self._artifact_path is None:
            self._render_poll_timer.stop()

    def _ensure_render_loop(self) -> StlPreviewRenderLoop:
        if self._render_loop is None:
            self._render_loop = StlPreviewRenderLoop(cache_root=self._cache_root)
        return self._render_loop

    def _apply_render(self, render: ArtifactPreviewRecord) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap

        if render.preview_path is None:
            self.clear()
            self.setText(render.diagnostic or "Preview unavailable")
            return
        pixmap = QPixmap(str(render.preview_path))
        if pixmap.isNull():
            self.setText("Preview unavailable")
            return
        self._base_pixmap = pixmap
        self._show_rendered_pixmap()

    def _show_rendered_pixmap(self) -> None:
        from PySide6.QtCore import Qt

        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        scaled = self._base_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        self._show_rendered_pixmap()
        super().resizeEvent(event)


class ReferenceReviewWindow(QWidget):
    """Widget-hosted review shell with an embedded PyVista interactor."""

    def __init__(
        self,
        queue: FixtureQueueViewModel,
        *,
        artifact_previews: dict[str, ArtifactPreviewRecord],
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
        if self._selected_index >= 0:
            self.list_widget.setCurrentRow(self._selected_index)
        else:
            self._sync_properties()
        self.show()
        self.raise_()
        self.activateWindow()

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
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap

        record = self.queue.records[self._selected_index]
        artifact_path = record.artifact_paths[0] if record.artifact_paths else None
        preview_url = str(item.get("artifact_preview_url", ""))
        if preview_url.startswith("file://"):
            pixmap = QPixmap(Path(preview_url.removeprefix("file://")).as_posix())
            if not pixmap.isNull():
                self.artifact_thumb.setPixmap(
                    pixmap.scaled(
                        148,
                        92,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        self.preview_surface.set_artifact(artifact_path if artifact_path and artifact_path.is_file() else None)

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
    artifact_previews: dict[str, ArtifactPreviewRecord] | None = None,
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
                "artifact_preview_url": preview.preview_url if preview is not None else "",
                "artifact_preview_status": preview.diagnostic
                if preview is not None and preview.diagnostic
                else "ready",
                "status": item.status,
            }
        )
    return items


def _artifact_previews_for_records(
    records: tuple[ReviewSourceModelRecord, ...],
) -> dict[str, ArtifactPreviewRecord]:
    cache_root = Path(".cache/reference-review/stl-previews")
    previews: dict[str, ArtifactPreviewRecord] = {}
    for record in records:
        if not record.artifact_paths:
            continue
        previews[record.fixture_id] = render_stl_preview(record.artifact_paths[0], cache_root=cache_root)
    return previews


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
