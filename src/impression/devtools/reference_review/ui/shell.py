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
    update_fixture_notes_in_file,
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
_RENDERABLE_ARTIFACT_SUFFIXES = frozenset({".stl", ".impress"})


@dataclass(frozen=True)
class WorkbenchLaunchResult:
    launched: bool
    diagnostics: tuple[str, ...] = ()
    engine: object | None = None


@dataclass(frozen=True)
class ContextEvidenceSummary:
    bundle_id: str
    evidence_kind: str
    role_summary: str

    def display_text(self) -> str:
        return f"{self.bundle_id} ({self.evidence_kind}): {self.role_summary}"


@dataclass(frozen=True)
class ArtifactEvidenceRow:
    bundle_id: str
    role: str
    kind: str
    path_label: str
    status: str

    def display_text(self) -> str:
        return f"{self.role} [{self.kind}] {self.path_label} - {self.status}"


def map_context_evidence_summary(record: ReviewSourceModelRecord) -> tuple[ContextEvidenceSummary, ...]:
    """Map fixture evidence bundles to compact context-tab labels."""

    summaries: list[ContextEvidenceSummary] = []
    for bundle in record.evidence_bundles:
        roles = tuple(artifact.role for artifact in bundle.artifacts)
        role_summary = ", ".join(roles) if roles else "no artifacts"
        summaries.append(
            ContextEvidenceSummary(
                bundle_id=bundle.bundle_id,
                evidence_kind=bundle.evidence_kind,
                role_summary=role_summary,
            )
        )
    return tuple(summaries)


def format_missing_artifact_status(path: Path, *, required: bool) -> str:
    """Format missing evidence artifact status without exposing absolute paths."""

    return "missing required" if required else "missing optional"


def map_artifact_evidence_rows(record: ReviewSourceModelRecord) -> tuple[ArtifactEvidenceRow, ...]:
    """Map fixture evidence artifacts to metadata-only artifact-tab rows."""

    rows: list[ArtifactEvidenceRow] = []
    for bundle in record.evidence_bundles:
        for artifact in bundle.artifacts:
            status = "available" if artifact.path.is_file() else format_missing_artifact_status(
                artifact.path,
                required=artifact.required,
            )
            rows.append(
                ArtifactEvidenceRow(
                    bundle_id=bundle.bundle_id,
                    role=artifact.role,
                    kind=artifact.kind,
                    path_label=artifact.path.name,
                    status=status,
                )
            )
    return tuple(rows)


def _renderable_artifact_paths(record: ReviewSourceModelRecord) -> tuple[Path, ...]:
    return tuple(
        path for path in record.artifact_paths if path.suffix.lower() in _RENDERABLE_ARTIFACT_SUFFIXES
    )


def _fixture_artifact_kind_label(record: ReviewSourceModelRecord) -> str:
    renderable_paths = _renderable_artifact_paths(record)
    if renderable_paths:
        return renderable_paths[0].name
    expected = (record.expected_output or "").lower()
    if "diagnostic" in expected:
        return "diagnostic evidence"
    if "evidence" in expected:
        return "evidence payload"
    if not record.artifact_paths:
        return "no renderable artifact"
    return "non-renderable artifact"


def _fixture_preview_empty_message(record: ReviewSourceModelRecord) -> str:
    renderable_paths = _renderable_artifact_paths(record)
    if renderable_paths:
        path = renderable_paths[0]
        if path.is_file():
            return ""
        return f"Renderable artifact missing: {path.name}"
    expected = (record.expected_output or "").lower()
    if "diagnostic" in expected or "evidence" in expected:
        return (
            "Diagnostic/evidence fixture: no STL or .impress artifact to render. "
            "Review the Context and Artifacts tabs."
        )
    return "No STL or .impress artifact is declared for this fixture."


def _ensure_qt_app(argv: Sequence[str], *, offscreen: bool, widgets: bool = False) -> object:
    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if widgets:
        os.environ.setdefault("QT_OPENGL", "desktop")
        os.environ.setdefault("QT_WIDGETS_RHI", "0")
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
            QCheckBox,
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
        self._selected_index = -1
        self._all_fixture_items = _fixture_items_for_qml(queue, artifact_previews)
        self._show_approved = False
        self._visible_record_indices: list[int] = []
        self._fixture_items = self._filtered_fixture_items()
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
        self._loading_notes = False
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
        self.show_approved_checkbox = QCheckBox("show approved")
        self.show_approved_checkbox.setObjectName("showApprovedCheckBox")
        self.show_approved_checkbox.setChecked(False)
        sidebar_layout.addWidget(self.show_approved_checkbox)
        self.list_widget = QListWidget()
        self.list_widget.setObjectName("fixtureQueueList")
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
        self.preview_stale_label = QLabel("Preview current.")
        self.preview_stale_label.setObjectName("reviewPreviewStaleIndicator")
        self.preview_stale_label.setStyleSheet("color: #9fb0c2; padding: 4px 8px;")
        self.preview_surface = InteractiveStlPreviewLabel(self.preview_frame)
        self.preview_surface.renderCommandApplied.connect(
            self._handle_preview_render_command_result
        )
        self.preview_layout.addWidget(self.preview_stale_label)
        self.preview_layout.addWidget(self.preview_surface)
        main_layout.addWidget(self.preview_frame, 1, 0, 2, 1)

        self.review_status_badge = QLabel("UNREVIEWED")
        self.review_status_badge.setObjectName("reviewStatusBadge")
        self.review_status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.review_status_badge.setMinimumHeight(28)
        self.review_status_badge.setMaximumHeight(34)
        self._set_review_status_badge(ReferenceReviewStatus.UNREVIEWED.value)
        main_layout.addWidget(self.review_status_badge, 1, 1)

        self.detail_tabs = QTabWidget()
        self.detail_tabs.setObjectName("reviewDetailTabs")
        self.detail_tabs.setMinimumSize(320, 260)

        context = QWidget()
        context_layout = QVBoxLayout(context)
        context_title = QLabel("Context")
        context_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.context_text = QLabel("")
        self.context_text.setObjectName("selectedFixtureContextText")
        self.context_text.setWordWrap(True)
        context_layout.addWidget(context_title)
        context_layout.addWidget(self.context_text, 1)

        notes = QWidget()
        notes_layout = QVBoxLayout(notes)
        notes_title = QLabel("Notes")
        notes_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.notes = QTextEdit()
        self.notes.setObjectName("reviewNotesTextEdit")
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
        main_layout.addWidget(self.detail_tabs, 2, 1)

        main_layout.setColumnStretch(0, 3)
        main_layout.setColumnStretch(1, 2)
        main_layout.setRowStretch(1, 0)
        main_layout.setRowStretch(2, 3)
        splitter.addWidget(main)
        splitter.setStretchFactor(1, 1)

        self._populate_queue_items()
        self.queue_status.setText(_visible_queue_status_text(self._fixture_items))

        self.refresh_button.clicked.connect(self._refresh_queue)
        self.reset_view_button.clicked.connect(self.preview_surface.reset_view)
        self.preview_display_controls.commandTriggered.connect(self._route_preview_display_command)
        self.approve_button.clicked.connect(self._approve_fixture)
        self.decline_button.clicked.connect(self._decline_fixture)
        self.previous_button.clicked.connect(self._previous_fixture)
        self.next_button.clicked.connect(self._next_fixture)
        self.show_approved_checkbox.toggled.connect(self._set_show_approved)
        self.list_widget.currentRowChanged.connect(self._select_visible_index)
        self.notes.textChanged.connect(self._persist_selected_notes)
        self.show()
        self.raise_()
        self.activateWindow()
        self._sync_properties()

    def _select_initial_fixture(self, index: int) -> None:
        row = self._visible_row_for_record_index(index)
        if self._offscreen:
            self.list_widget.setCurrentRow(row)
            return
        from PySide6.QtCore import QTimer

        self._sync_properties()
        QTimer.singleShot(75, lambda: self._apply_initial_selection(row))

    def _apply_initial_selection(self, index: int) -> None:
        if self.list_widget.count() == 0:
            return
        self.list_widget.setCurrentRow(max(0, min(index, self.list_widget.count() - 1)))

    def _filtered_fixture_items(self) -> list[dict[str, object]]:
        visible: list[dict[str, object]] = []
        self._visible_record_indices = []
        for index, item in enumerate(self._all_fixture_items):
            if not self._show_approved and item.get("status") == ReferenceReviewStatus.APPROVED.value:
                continue
            visible.append(item)
            self._visible_record_indices.append(index)
        return visible

    def _visible_row_for_record_index(self, record_index: int) -> int:
        try:
            return self._visible_record_indices.index(record_index)
        except ValueError:
            return 0 if self._visible_record_indices else -1

    def _populate_queue_items(self) -> None:
        from PySide6.QtWidgets import QListWidgetItem

        blocked = self.list_widget.blockSignals(True)
        try:
            self.list_widget.clear()
            for row, item in enumerate(self._fixture_items):
                artifact_label = item["artifact_kind_label"]
                label = (
                    f"{item['fixture_id']} [{item['status']}]\n"
                    f"{artifact_label}"
                )
                widget_item = QListWidgetItem(label)
                widget_item.setData(256, item["fixture_id"])
                widget_item.setData(257, self._visible_record_indices[row])
                self.list_widget.addItem(widget_item)
        finally:
            self.list_widget.blockSignals(blocked)

    def _refresh_queue(self) -> None:
        if self._fixture_items:
            self.queue_status.setText(_visible_queue_status_text(self._fixture_items))
            self.list_widget.setCurrentRow(self._visible_row_for_record_index(self._selected_index))
        else:
            self.queue_status.setText("No fixtures shown")
            self._select_index(-1)

    def _previous_fixture(self) -> None:
        self.list_widget.setCurrentRow(max(0, self.list_widget.currentRow() - 1))

    def _next_fixture(self) -> None:
        self.list_widget.setCurrentRow(min(len(self._fixture_items) - 1, self.list_widget.currentRow() + 1))

    def _set_show_approved(self, checked: bool) -> None:
        selected_fixture_id = self.queue.records[self._selected_index].fixture_id if self._selected_record() else None
        self._show_approved = checked
        self._fixture_items = self._filtered_fixture_items()
        self._populate_queue_items()
        if not self._fixture_items:
            self._select_index(-1)
            self.queue_status.setText("No fixtures shown")
            return
        next_row = 0
        if selected_fixture_id is not None:
            for row, item in enumerate(self._fixture_items):
                if item["fixture_id"] == selected_fixture_id:
                    next_row = row
                    break
        self.list_widget.setCurrentRow(next_row)
        self.queue_status.setText(_visible_queue_status_text(self._fixture_items))

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
        self._all_fixture_items = _fixture_items_for_qml(self.queue, self.artifact_previews)
        self._fixture_items = self._filtered_fixture_items()
        self._populate_queue_items()
        row = self._visible_row_for_record_index(self._selected_index)
        if row < 0:
            self._select_index(-1)
            return
        self.list_widget.setCurrentRow(row)
        if self.list_widget.currentRow() != row:
            self._select_visible_index(row)

    def _select_visible_index(self, row: int) -> None:
        if row < 0 or row >= len(self._visible_record_indices):
            self._select_index(-1)
            return
        self._select_index(self._visible_record_indices[row])

    def _select_index(self, index: int) -> None:
        if index < 0 or index >= len(self.queue.records):
            self._selected_index = -1
            self.preview_surface.set_artifact(None)
            self.preview_display_controls.set_ready(False)
            self.context_text.setText("No fixture context loaded.")
            self._load_selected_notes(None)
            self.artifact_thumb.setText("")
            self._set_review_status_badge(ReferenceReviewStatus.UNREVIEWED.value)
            self._sync_properties()
            return
        self._selected_index = index
        item = self._all_fixture_items[index]
        self._set_review_status_badge(str(item["status"]))
        self.context_text.setText(
            f"{item['fixture_id']}\n\n"
            f"Review: {item['status']}\n"
            f"Purpose: {item['purpose'] or 'not provided'}\n\n"
            f"Methodology: {item['methodology'] or 'not provided'}\n\n"
            f"Rendered result: {item['render_description'] or 'not provided'}\n\n"
            f"Source: {item['source_display_path']}\n"
            f"Expected: {item['expected_output'] or 'not declared'}\n"
            f"Artifact: {item['artifact_display_path']}"
            f"{item['evidence_summary_text']}"
        )
        self._load_artifact_preview(item)
        self._load_selected_notes(self.queue.records[index])
        self._sync_properties()

    def _load_selected_notes(self, record: ReviewSourceModelRecord | None) -> None:
        self._loading_notes = True
        try:
            self.notes.setPlainText(record.notes if record is not None else "")
        finally:
            self._loading_notes = False

    def _persist_selected_notes(self) -> None:
        if self._loading_notes:
            return
        record = self._selected_record()
        if record is None:
            return
        notes = self.notes.toPlainText()
        if notes == record.notes:
            return
        updated_record = replace(record, notes=notes)
        self._replace_selected_record_without_reselection(updated_record)
        if not self._persist_fixture_notes(updated_record):
            return
        self._sync_properties()

    def _persist_fixture_notes(self, record: ReviewSourceModelRecord) -> bool:
        if not self._fixture_files:
            self.queue_status.setText("Review notes changed locally; no fixture file configured")
            return True
        diagnostics: list[str] = []
        updated = False
        for fixture_file in self._fixture_files:
            result = update_fixture_notes_in_file(
                fixture_file,
                fixture_id=record.fixture_id,
                notes=record.notes,
            )
            updated = updated or result.updated
            diagnostics.extend(diagnostic.code for diagnostic in result.diagnostics)
        if not updated:
            self.queue_status.setText(f"Review notes not saved: {', '.join(diagnostics) or 'no source'}")
            return False
        return True

    def _load_artifact_preview(self, item: dict[str, object]) -> None:
        record_index = self._record_index_for_fixture_id(str(item.get("fixture_id", "")))
        if record_index is None:
            self.preview_surface.enqueue_render_command(
                PreviewRenderCommand.clear("No fixture selected.")
            )
            return
        record = self.queue.records[record_index]
        renderable_paths = _renderable_artifact_paths(record)
        artifact_path = renderable_paths[0] if renderable_paths else None
        live_artifact = artifact_path if artifact_path and artifact_path.is_file() else None
        self._preview_load_generation += 1
        self.artifact_thumb.setText(str(item.get("artifact_kind_label", "")))
        if item.get("artifact_evidence_text"):
            self.artifact_thumb.setText(str(item["artifact_evidence_text"]))
        self.preview_surface.prepare_artifact(live_artifact)
        self.preview_surface.enqueue_render_command(PreviewRenderCommand.loading())
        self.preview_stale_label.setText("Preview loading...")
        self.preview_display_controls.set_ready(False)
        if live_artifact is None:
            self.preview_surface.enqueue_render_command(
                PreviewRenderCommand.clear(str(item.get("preview_empty_message", "")) or "No fixture selected.")
            )
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

    def _record_index_for_fixture_id(self, fixture_id: str) -> int | None:
        for index, record in enumerate(self.queue.records):
            if record.fixture_id == fixture_id:
                return index
        return None

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
        self.preview_stale_label.setText("Preview current.")
        self.preview_surface.enqueue_render_command(
            PreviewRenderCommand.payload_ready(
                event.payload,
                display_options=self._preview_display_options,
            )
        )

    def _apply_preview_payload_failed(self, event: PreviewPayloadFailedEvent) -> None:
        if self.preview_surface.payload_state.ready:
            self.preview_stale_label.setText(f"Preview stale: {event.diagnostic.message}")
            self.preview_display_controls.set_ready(True)
            return
        self.preview_surface.enqueue_render_command(
            PreviewRenderCommand.failure(
                f"Preview unavailable: {event.diagnostic.message}",
                diagnostic=event.diagnostic.message,
                identity=self._preview_controller.active_identity,
            )
        )
        self.preview_stale_label.setText("Preview unavailable.")
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
        has_fixture = 0 <= self._selected_index < len(self._all_fixture_items)
        selected = self._all_fixture_items[self._selected_index] if has_fixture else None
        self.setProperty("reviewFixtures", self._fixture_items)
        self.setProperty("queueStatusText", self.queue_status.text())
        self.setProperty("selectedMessageText", selected["fixture_id"] if selected else "No fixture selected.")
        self.setProperty("hasFixture", has_fixture)
        self.setProperty("showApproved", self._show_approved)
        self.setProperty("interactivePreviewReady", self._interactive_preview_ready)

    def _replace_selected_record_without_reselection(self, record: ReviewSourceModelRecord) -> None:
        records = list(self.queue.records)
        records[self._selected_index] = record
        self.queue.records = tuple(records)
        self._all_fixture_items = _fixture_items_for_qml(self.queue, self.artifact_previews)
        self._fixture_items = self._filtered_fixture_items()

    def _set_review_status_badge(self, status: str) -> None:
        label, stylesheet = _review_status_badge_style(status)
        self.review_status_badge.setText(label)
        self.review_status_badge.setStyleSheet(stylesheet)

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
    for item, record in zip(queue.items, queue.records, strict=True):
        preview = artifact_previews.get(item.fixture_id)
        evidence_summary = tuple(summary.display_text() for summary in map_context_evidence_summary(record))
        artifact_rows = tuple(row.display_text() for row in map_artifact_evidence_rows(record))
        items.append(
            {
                "fixture_id": item.fixture_id,
                "feature_name": item.feature_name,
                "source_display_path": item.source_display_path,
                "expected_output": item.expected_output or "",
                "purpose": item.purpose or "",
                "methodology": item.methodology or "",
                "render_description": item.render_description or "",
                "artifact_display_path": item.artifact_display_path or "",
                "artifact_kind_label": _fixture_artifact_kind_label(record),
                "renderable_artifact": bool(_renderable_artifact_paths(record)),
                "preview_empty_message": _fixture_preview_empty_message(record),
                "artifact_preview_url": getattr(preview, "preview_url", "") if preview is not None else "",
                "artifact_preview_status": getattr(preview, "diagnostic", None)
                if preview is not None and getattr(preview, "diagnostic", None)
                else "ready",
                "evidence_summary_text": ""
                if not evidence_summary
                else "\n\nEvidence:\n" + "\n".join(evidence_summary),
                "artifact_evidence_text": ""
                if not artifact_rows
                else "\n".join(artifact_rows),
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


def _visible_queue_status_text(items: list[dict[str, object]]) -> str:
    if items:
        return f"{len(items)} fixture{'s' if len(items) != 1 else ''} shown"
    return "No fixtures shown"


def _review_status_badge_style(status: str) -> tuple[str, str]:
    normalized = status if status in {item.value for item in ReferenceReviewStatus} else ReferenceReviewStatus.UNREVIEWED.value
    palette = {
        ReferenceReviewStatus.APPROVED.value: ("APPROVED", "#1f7a4d", "#ffffff"),
        ReferenceReviewStatus.DECLINED.value: ("DECLINED", "#b42318", "#ffffff"),
        ReferenceReviewStatus.UNREVIEWED.value: ("UNREVIEWED", "#5f6368", "#ffffff"),
    }
    label, background, foreground = palette[normalized]
    return (
        label,
        (
            f"background: {background}; color: {foreground};"
            " border: 1px solid rgba(255, 255, 255, 0.22);"
            " border-radius: 4px; padding: 4px 12px;"
            " font-size: 12px; font-weight: 700;"
        ),
    )


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
