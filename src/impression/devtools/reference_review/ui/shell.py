"""Launcher/bootstrap for the Reference Review Workbench."""

from __future__ import annotations

import os
import sys
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from .bridge import BridgeRecord, BridgeRegistry
from .packaging import qml_resource_root
from .preview_controls import (
    PreviewDisplayControlBar,
    PreviewDisplayOptions,
    route_preview_display_command,
)
from .preview_render_queue import (
    PreviewRenderCommand,
    PreviewRenderCommandKind,
    PreviewRenderCommandResult,
    PreviewRenderIdentity,
)
from .preview_widget import PreviewRendererLifecycleWidget
from .queue_context import FixtureQueueViewModel
from ..source_registry import (
    DiscoverySummary,
    ReferenceReviewStatus,
    ReviewSourceModelRecord,
    approve_reference_artifacts,
    discover_source_records,
    load_source_records_from_database,
    load_source_records_from_file,
    update_fixture_review_status_in_database,
    update_fixture_review_status_in_file,
)
from ..preview_payload_builder import (
    ImpressPreviewBuildResult,
    build_impress_preview_result,
    load_impress_preview_datasets,
)
from ..preview_payload_controller import (
    PreviewPayloadFailedEvent,
    PreviewPayloadProcessController,
    PreviewPayloadReadyEvent,
)
from ..async_core import WorkerResultEnvelope
from ..async_core import ReviewTaskKind, TaskDispatcher, WorkerPolicy
from impression.preview_qt import configure_qt_preview_surface_format

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
        os.environ.setdefault("QT_OPENGL", "desktop")
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            return app
        configure_qt_preview_surface_format()
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
    fixture_files: tuple[Path, ...] = (),
    fixture_databases: tuple[Path, ...] = (),
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
            fixture_files=fixture_files,
            fixture_databases=fixture_databases,
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
            fixture_files=fixture_files,
            fixture_databases=fixture_databases,
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
    fixture_files: tuple[Path, ...],
    fixture_databases: tuple[Path, ...],
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
    return load_impress_preview_datasets(artifact_path)


def _build_impress_preview_result(generation: int, artifact_path: Path) -> _LivePreviewBuildResult:
    result = build_impress_preview_result(generation, artifact_path)
    return _LivePreviewBuildResult(
        result.generation,
        result.artifact_path,
        datasets=result.datasets,
        diagnostic=result.diagnostic,
    )


class LiveArtifactPreviewWidget(PreviewRendererLifecycleWidget):
    """In-window live artifact preview using the same VTK interaction model as CLI preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._artifact_path: Path | None = None
        self._current_datasets = []
        self._load_generation = 0

    def set_artifact(self, artifact_path: Path | None) -> None:
        self._artifact_path = artifact_path
        self._current_datasets = []
        self._load_generation += 1
        if artifact_path is None:
            self._clear_scene("No fixture selected.")
            return
        self.prepare_artifact(artifact_path)
        self._clear_scene("Preview payload pending.")

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
        self._plotter.reset_camera()
        self._plotter.reset_camera_clipping_range()
        self._plotter.render()

    def shutdown(self) -> None:
        self.dispose_renderer()

    def _apply_build_result(self, result: _LivePreviewBuildResult | ImpressPreviewBuildResult) -> None:
        if result.generation != self._load_generation or result.artifact_path != self._artifact_path:
            return
        if result.diagnostic is not None:
            self._clear_scene(f"Preview unavailable: {result.diagnostic}")
            return
        try:
            plotter = self._ensure_plotter()
            self._status.hide()
            self._current_datasets = list(result.datasets)
            set_datasets = getattr(plotter, "set_datasets", None)
            if not callable(set_datasets):
                raise RuntimeError("preview-renderer-missing-set-datasets")
            set_datasets(tuple(self._current_datasets))
            self._apply_display_options_to_plotter()
            self.reset_view()
        except Exception as exc:
            self._clear_scene(f"Preview unavailable: {exc.__class__.__name__}")

    def _ensure_plotter(self):
        return self.ensure_renderer()

    def _clear_scene(self, message: str) -> None:
        self.clear_renderer_scene(message)


InteractiveStlPreviewLabel = LiveArtifactPreviewWidget


class ReferenceReviewWindow(QWidget):
    """Widget-hosted review shell with an embedded PyVista interactor."""

    def __init__(
        self,
        queue: FixtureQueueViewModel,
        *,
        artifact_previews: dict[str, object],
        startup_diagnostics: tuple[str, ...],
        fixture_files: tuple[Path, ...],
        fixture_databases: tuple[Path, ...],
        offscreen: bool,
    ) -> None:
        from PySide6.QtCore import Qt, QTimer
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
        self._fixture_files = fixture_files
        self._fixture_databases = fixture_databases
        self._offscreen = offscreen
        self._interactive_preview_ready = True
        self._selected_index = queue.selected_index if queue.selected_index is not None else -1
        self._fixture_items = _fixture_items_for_qml(queue, artifact_previews)
        self._preview_load_generation = 0
        self._preview_display_options = PreviewDisplayOptions()
        self._preview_controller = PreviewPayloadProcessController(
            cwd=Path.cwd(),
            dispatcher=TaskDispatcher(
                max_workers=1,
                policies={
                    ReviewTaskKind.PREVIEW_BUILD: WorkerPolicy(max_pending=1, coalesce=True),
                },
            ),
            owns_dispatcher=True,
        )
        self._preview_futures: list[Future[WorkerResultEnvelope]] = []
        self._preview_future_identities: dict[
            Future[WorkerResultEnvelope],
            PreviewRenderIdentity | None,
        ] = {}
        self._pending_preview_record: ReviewSourceModelRecord | None = None
        self._preview_poll_timer = QTimer(self)
        self._preview_poll_timer.setInterval(30)
        self._preview_poll_timer.timeout.connect(self._poll_preview_payloads)

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
        self.approve_button = QPushButton("Approve" if not startup_diagnostics else "Diagnostics")
        self.approve_button.setObjectName("approveFixtureButton")
        self.decline_button = QPushButton("Decline")
        self.decline_button.setObjectName("declineFixtureButton")
        self.previous_button = QPushButton("Previous")
        self.previous_button.setObjectName("previousFixtureButton")
        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("nextFixtureButton")
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.setObjectName("resetPreviewButton")
        self.preview_display_controls = PreviewDisplayControlBar(options=self._preview_display_options)
        self.preview_display_controls.set_ready(False)
        header.addWidget(self.preview_display_controls, 1)
        header.addWidget(self.approve_button)
        header.addWidget(self.decline_button)
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
        self.preview_surface.renderCommandApplied.connect(
            self._handle_preview_render_command_result
        )
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

        main_layout.setColumnStretch(0, 3)
        main_layout.setColumnStretch(1, 2)
        main_layout.setRowStretch(1, 3)
        splitter.addWidget(main)
        splitter.setStretchFactor(1, 1)

        self._populate_queue_items()

        self.refresh_button.clicked.connect(self._refresh_queue)
        self.reset_view_button.clicked.connect(self.preview_surface.reset_view)
        self.preview_display_controls.commandTriggered.connect(self._route_preview_display_command)
        self.approve_button.clicked.connect(self._approve_fixture)
        self.decline_button.clicked.connect(self._decline_fixture)
        self.previous_button.clicked.connect(self._previous_fixture)
        self.next_button.clicked.connect(self._next_fixture)
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
            label = (
                f"{item['fixture_id']} [{item['status']}]\n"
                f"{item['artifact_display_path'] or item['source_display_path']}"
            )
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

    def _approve_fixture(self) -> None:
        record = self._selected_record()
        if record is None:
            self.queue_status.setText("No fixture selected")
            return
        promotion = approve_reference_artifacts(record)
        if not promotion.updated:
            self.queue_status.setText(_diagnostic_summary("Approval failed", promotion.diagnostics))
            return
        approved_record = replace(
            record,
            artifact_paths=promotion.artifact_paths,
            review_status=ReferenceReviewStatus.APPROVED,
        )
        if not self._persist_review_status(
            approved_record,
            ReferenceReviewStatus.APPROVED,
            artifact_paths=promotion.artifact_paths,
        ):
            return
        self._replace_selected_record(approved_record)
        self.queue_status.setText(f"{record.fixture_id} approved")
        self._sync_properties()

    def _decline_fixture(self) -> None:
        record = self._selected_record()
        if record is None:
            self.queue_status.setText("No fixture selected")
            return
        declined_record = replace(record, review_status=ReferenceReviewStatus.DECLINED)
        if not self._persist_review_status(declined_record, ReferenceReviewStatus.DECLINED):
            return
        self._replace_selected_record(declined_record)
        self.queue_status.setText(f"{record.fixture_id} declined")
        self._sync_properties()

    def _selected_record(self) -> ReviewSourceModelRecord | None:
        if self._selected_index < 0 or self._selected_index >= len(self.queue.records):
            return None
        return self.queue.records[self._selected_index]

    def _persist_review_status(
        self,
        record: ReviewSourceModelRecord,
        status: ReferenceReviewStatus,
        *,
        artifact_paths: tuple[Path, ...] | None = None,
    ) -> bool:
        targets = list(self._fixture_files) + list(self._fixture_databases)
        if not targets:
            self.queue_status.setText("Review status changed locally; no fixture file/db configured")
            return True
        diagnostics: list[str] = []
        updated = False
        for fixture_file in self._fixture_files:
            result = update_fixture_review_status_in_file(
                fixture_file,
                fixture_id=record.fixture_id,
                status=status,
                artifact_paths=artifact_paths,
            )
            updated = updated or result.updated
            diagnostics.extend(diagnostic.code for diagnostic in result.diagnostics)
        for fixture_database in self._fixture_databases:
            result = update_fixture_review_status_in_database(
                fixture_database,
                fixture_id=record.fixture_id,
                status=status,
                artifact_paths=artifact_paths,
            )
            updated = updated or result.updated
            diagnostics.extend(diagnostic.code for diagnostic in result.diagnostics)
        if not updated:
            self.queue_status.setText(f"Review status not saved: {', '.join(diagnostics) or 'no source'}")
            return False
        return True

    def _replace_selected_record(self, record: ReviewSourceModelRecord) -> None:
        records = list(self.queue.records)
        records[self._selected_index] = record
        self.queue.records = tuple(records)
        self.queue.statuses[record.fixture_id] = record.review_status.value
        self._fixture_items = _fixture_items_for_qml(self.queue, self.artifact_previews)
        self._populate_queue_items()
        self.list_widget.setCurrentRow(self._selected_index)
        self._select_index(self._selected_index)

    def _select_index(self, index: int) -> None:
        if index < 0 or index >= len(self._fixture_items):
            self._selected_index = -1
            self.preview_surface.set_artifact(None)
            self.preview_display_controls.set_ready(False)
            self.context_text.setText("No fixture context loaded.")
            self.artifact_thumb.setText("")
            self._sync_properties()
            return
        self._selected_index = index
        item = self._fixture_items[index]
        self.context_text.setText(
            f"{item['fixture_id']}\n\n"
            f"Review: {item['status']}\n"
            f"Source: {item['source_display_path']}\n"
            f"Expected: {item['expected_output'] or 'not declared'}\n"
            f"Artifact: {item['artifact_display_path']}"
        )
        self._load_artifact_preview(item)
        self._sync_properties()

    def _load_artifact_preview(self, item: dict[str, object]) -> None:
        record = self.queue.records[self._selected_index]
        artifact_path = record.artifact_paths[0] if record.artifact_paths else None
        live_artifact = artifact_path if artifact_path and artifact_path.is_file() else None
        self._preview_load_generation += 1
        self.artifact_thumb.setText(str(item.get("artifact_display_path", "")))
        self.preview_surface.prepare_artifact(live_artifact)
        self.preview_surface.enqueue_render_command(PreviewRenderCommand.loading())
        self.preview_display_controls.set_ready(False)
        if live_artifact is None:
            self.preview_surface.enqueue_render_command(PreviewRenderCommand.clear())
            return
        result = self._preview_controller.launch(record)
        if not result.accepted or result.future is None:
            if result.diagnostic in {"queue_full", "coalesced"} and self._preview_futures:
                self._pending_preview_record = record
                if not self._preview_poll_timer.isActive():
                    self._preview_poll_timer.start()
                return
            identity = self._preview_identity_for_dispatch_request(getattr(result, "request", None))
            self.preview_surface.enqueue_render_command(
                PreviewRenderCommand.failure(
                    f"Preview unavailable: {result.diagnostic or 'queue_full'}",
                    diagnostic=result.diagnostic,
                    identity=identity,
                )
            )
            return
        self._preview_futures.append(result.future)
        self._preview_future_identities[result.future] = (
            self._preview_identity_for_dispatch_request(getattr(result, "request", None))
        )
        if not self._preview_poll_timer.isActive():
            self._preview_poll_timer.start()

    def _apply_preview_load(self, generation: int, artifact_path: Path | None) -> None:
        if generation != self._preview_load_generation:
            return
        self.preview_surface.set_artifact(artifact_path)

    def _poll_preview_payloads(self) -> None:
        pending: list[Future[WorkerResultEnvelope]] = []
        for future in self._preview_futures:
            if not future.done():
                pending.append(future)
                continue
            try:
                envelope = future.result()
            except Exception as exc:
                identity = self._preview_future_identities.pop(future, None)
                if self._preview_identity_is_current(identity):
                    self.preview_surface.enqueue_render_command(
                        PreviewRenderCommand.failure(
                            f"Preview unavailable: {exc.__class__.__name__}",
                            diagnostic=str(exc) or exc.__class__.__name__,
                            identity=identity,
                        )
                    )
                continue
            self._preview_future_identities.pop(future, None)
            self._preview_controller.handle_completion(
                envelope,
                self._apply_preview_payload_ready,
                self._apply_preview_payload_failed,
            )
        self._preview_futures = pending
        if not self._preview_futures:
            self._preview_poll_timer.stop()
            self._launch_pending_preview_record()

    def _launch_pending_preview_record(self) -> None:
        record = self._pending_preview_record
        self._pending_preview_record = None
        if record is None:
            return
        if self._selected_index < 0 or self._selected_index >= len(self.queue.records):
            return
        if self.queue.records[self._selected_index].fixture_id != record.fixture_id:
            return
        self.preview_surface.enqueue_render_command(PreviewRenderCommand.loading())
        result = self._preview_controller.launch(record)
        if not result.accepted or result.future is None:
            if result.diagnostic in {"queue_full", "coalesced"} and self._preview_futures:
                self._pending_preview_record = record
                if not self._preview_poll_timer.isActive():
                    self._preview_poll_timer.start()
                return
            identity = self._preview_identity_for_dispatch_request(getattr(result, "request", None))
            self.preview_surface.enqueue_render_command(
                PreviewRenderCommand.failure(
                    f"Preview unavailable: {result.diagnostic or 'queue_full'}",
                    diagnostic=result.diagnostic,
                    identity=identity,
                )
            )
            return
        self._preview_futures.append(result.future)
        self._preview_future_identities[result.future] = (
            self._preview_identity_for_dispatch_request(getattr(result, "request", None))
        )
        if not self._preview_poll_timer.isActive():
            self._preview_poll_timer.start()

    def _apply_preview_payload_ready(self, event: PreviewPayloadReadyEvent) -> None:
        self.preview_surface.enqueue_render_command(
            PreviewRenderCommand.payload_ready(
                event.payload,
                display_options=self._preview_display_options,
            )
        )

    def _apply_preview_payload_failed(self, event: PreviewPayloadFailedEvent) -> None:
        self.preview_surface.enqueue_render_command(
            PreviewRenderCommand.failure(
                f"Preview unavailable: {event.diagnostic.message}",
                diagnostic=event.diagnostic.message,
                identity=self._preview_controller.active_identity,
            )
        )
        self.preview_display_controls.set_ready(False)

    def _handle_preview_render_command_result(
        self,
        result: PreviewRenderCommandResult,
    ) -> None:
        if result.command.kind is PreviewRenderCommandKind.PAYLOAD:
            self.preview_display_controls.set_ready(result.ready)
            if result.ready and result.command.payload is not None:
                self._preview_controller.cleanup_payload(result.command.payload, reason="completed")
            return
        if result.command.kind in {
            PreviewRenderCommandKind.CLEAR,
            PreviewRenderCommandKind.FAILURE,
            PreviewRenderCommandKind.LOADING,
        }:
            self.preview_display_controls.set_ready(False)

    def _preview_identity_for_dispatch_request(
        self,
        request: object,
    ) -> PreviewRenderIdentity | None:
        owner = getattr(request, "owner", None)
        request_id = getattr(request, "request_id", None)
        fixture_id = getattr(request, "fixture_id", None)
        payload = getattr(request, "payload", None)
        generation = payload.get("generation") if isinstance(payload, dict) else None
        if (
            not isinstance(owner, str)
            or not isinstance(request_id, int)
            or not isinstance(fixture_id, str)
            or not isinstance(generation, int)
        ):
            return None
        return (owner, request_id, fixture_id, generation)

    def _preview_identity_is_current(self, identity: PreviewRenderIdentity | None) -> bool:
        return identity is not None and identity == self._preview_controller.active_identity

    def _route_preview_display_command(self, command: str) -> None:
        result = route_preview_display_command(
            self._preview_display_options,
            command,
            ready=bool(self.preview_surface.payload_state.ready),
        )
        if not result.executed:
            return
        self._preview_display_options = result.options
        self.preview_display_controls.set_options(result.options)
        self.preview_surface.enqueue_render_command(
            PreviewRenderCommand.display(
                result.options,
                identity=self._preview_controller.active_identity,
            )
        )

    def _sync_properties(self) -> None:
        has_fixture = self._selected_index >= 0
        selected = self._fixture_items[self._selected_index] if has_fixture else None
        self.setProperty("reviewFixtures", self._fixture_items)
        self.setProperty("queueStatusText", self.queue_status.text())
        self.setProperty("selectedMessageText", selected["fixture_id"] if selected else "No fixture selected.")
        self.setProperty("hasFixture", has_fixture)
        self.setProperty("interactivePreviewReady", self._interactive_preview_ready)

    def closeEvent(self, event) -> None:
        self._preview_poll_timer.stop()
        for future in self._preview_futures:
            future.cancel()
        self._preview_futures = []
        self._preview_future_identities = {}
        self._pending_preview_record = None
        self._preview_controller.close()
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


def _diagnostic_summary(prefix: str, diagnostics: tuple[object, ...]) -> str:
    codes = [getattr(diagnostic, "code", str(diagnostic)) for diagnostic in diagnostics]
    return f"{prefix}: {', '.join(codes) if codes else 'unknown'}"


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

    configure_qt_preview_surface_format()
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
        fixture_files=tuple(fixture_files),
        fixture_databases=tuple(fixture_databases),
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
