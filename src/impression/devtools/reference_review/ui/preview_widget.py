"""Qt preview widget lifecycle host for Reference Review."""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from impression.mesh import Mesh, Polyline
from impression.preview import (
    PreviewSceneApplyOptions,
)
from impression.preview_qt import (
    QtPreviewSurface,
    QtPreviewSurfaceConfig,
    apply_qt_preview_scene,
    configure_qt_preview_surface_format,
    qt_preview_supported_environment,
)
from ..async_core.qt_handoff import sanitize_error_text
from ..preview_payload import PreviewPayload
from ..preview_payload_builder import PREVIEW_PAYLOAD_FORMAT
from .preview_controls import (
    COLOR_MODE_AUTHORED,
    LIGHTING_MODE_CAMERA,
    LIGHTING_MODE_FACE_NORMALS,
    LIGHTING_MODE_FLAT,
    PreviewDisplayOptions,
)
from .preview_render_queue import (
    PreviewRenderCommand,
    PreviewRenderCommandKind,
    PreviewRenderCommandQueue,
    PreviewRenderCommandResult,
)

_PREVIEW_WIDGET_APP: QApplication | None = None
_SOFTWARE_PREVIEW_BACKGROUND = QColor("#07111f")
_SOFTWARE_PREVIEW_BACKGROUND_TOP = QColor("#10223a")
_SOFTWARE_PREVIEW_DEFAULT_MESH = QColor("#f4a261")
_SOFTWARE_PREVIEW_EDGE = QColor("#ffd2a8")
_SOFTWARE_PREVIEW_GRID = QColor("#31445f")
_SOFTWARE_PREVIEW_AXIS_X = QColor("#ff7d6e")
_SOFTWARE_PREVIEW_AXIS_Y = QColor("#7fd37d")
_SOFTWARE_PREVIEW_AXIS_Z = QColor("#72a8ff")
_WHEEL_ZOOM_REVERSE_THRESHOLD_UNITS = 2.0


@dataclass(frozen=True)
class PreviewRendererLifecycleState:
    """Inspectable renderer lifecycle state for tests and diagnostics."""

    created: bool = False
    disposed: bool = False
    diagnostic: str | None = None


@dataclass(frozen=True)
class PreviewWidgetPayloadState:
    """Current decoded payload state owned by the preview widget."""

    generation: int | None = None
    fixture_id: str | None = None
    ready: bool = False
    diagnostic: str | None = None


@dataclass(frozen=True)
class PreviewPaneVisibleState:
    """Workbench-visible state owned outside the render widget."""

    mode: str
    fixture_id: str | None = None
    message: str = ""
    diagnostic: str | None = None
    toolbar_enabled: bool = False


@dataclass(frozen=True)
class PreviewToolbarCommandRecord:
    """Result of routing a preview toolbar command."""

    command: str
    executed: bool
    enabled: bool
    diagnostic: str | None = None


@dataclass(frozen=True)
class PreviewWrapperSmokeRecord:
    """Manual smoke evidence command for the live embedded preview wrapper."""

    fixture_file: Path
    command: tuple[str, ...]
    expected_fixture_type: str = ".impress"
    verifies_lifecycle: bool = True


@dataclass(frozen=True)
class _WheelZoomDecision:
    direction: int
    latched_direction: int | None
    reverse_delta_units: float = 0.0


@dataclass(frozen=True)
class _ProjectedFace:
    depth: float
    polygon: QPolygonF
    color: QColor
    edge_segments: tuple[tuple[QPointF, QPointF], ...]
    triangle_segments: tuple[tuple[QPointF, QPointF], ...]


@dataclass(frozen=True)
class _ProjectedPolyline:
    depth: float
    polygon: QPolygonF
    color: QColor
    closed: bool


@dataclass(frozen=True)
class _ProjectedLine:
    depth: float
    start: QPointF
    end: QPointF
    color: QColor
    width: float = 1.0


@dataclass(frozen=True)
class _ProjectedPreviewScene:
    faces: tuple[_ProjectedFace, ...]
    polylines: tuple[_ProjectedPolyline, ...]
    grid_lines: tuple[_ProjectedLine, ...]
    axis_lines: tuple[_ProjectedLine, ...]


@dataclass(frozen=True)
class _PreparedFaceEdges:
    edge_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _PreparedMesh:
    vertices: np.ndarray
    faces: np.ndarray
    authored_color: QColor | None
    face_colors: tuple[QColor | None, ...]
    face_edges: tuple[_PreparedFaceEdges, ...]
    triangle_edges: tuple[_PreparedFaceEdges, ...]
    face_normals: np.ndarray


@dataclass(frozen=True)
class _PreparedPolyline:
    points: np.ndarray
    color: QColor
    closed: bool


@dataclass(frozen=True)
class _PreparedPreviewGeometry:
    center: np.ndarray
    span: float
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    items: tuple[_PreparedMesh | _PreparedPolyline, ...]


def build_preview_wrapper_smoke_record(
    fixture_file: Path = Path("tests/reference_review_fixtures/dirty-impress-fixtures.json"),
) -> PreviewWrapperSmokeRecord:
    return PreviewWrapperSmokeRecord(
        fixture_file=fixture_file,
        command=(
            ".venv/bin/impression-reference-review",
            "--fixture-file",
            fixture_file.as_posix(),
        ),
    )


def preview_pane_empty_state() -> PreviewPaneVisibleState:
    return PreviewPaneVisibleState(
        mode="empty",
        message="No fixture selected.",
    )


def preview_pane_loading_state(fixture_id: str) -> PreviewPaneVisibleState:
    return PreviewPaneVisibleState(
        mode="loading",
        fixture_id=fixture_id,
        message="Loading preview...",
    )


def preview_pane_ready_state(payload_state: PreviewWidgetPayloadState) -> PreviewPaneVisibleState:
    return PreviewPaneVisibleState(
        mode="ready",
        fixture_id=payload_state.fixture_id,
        message="Preview ready.",
        toolbar_enabled=payload_state.ready,
    )


def preview_pane_failure_state(
    fixture_id: str | None,
    diagnostic: str,
    *,
    cwd: Path | None = None,
    home: Path | None = None,
) -> PreviewPaneVisibleState:
    sanitized = sanitize_error_text(diagnostic, cwd=cwd, home=home)
    return PreviewPaneVisibleState(
        mode="failure",
        fixture_id=fixture_id,
        message="Preview unavailable.",
        diagnostic=sanitized,
        toolbar_enabled=False,
    )


def resolve_preview_toolbar_enabled(state: PreviewPaneVisibleState) -> bool:
    return state.toolbar_enabled and state.mode == "ready"


def route_preview_toolbar_command(
    widget: object,
    state: PreviewPaneVisibleState,
    command: str,
) -> PreviewToolbarCommandRecord:
    enabled = resolve_preview_toolbar_enabled(state)
    if not enabled:
        return PreviewToolbarCommandRecord(
            command=command,
            executed=False,
            enabled=False,
            diagnostic="preview-toolbar-disabled",
        )
    if command == "reset":
        reset = getattr(widget, "reset_view", None)
        if not callable(reset):
            return PreviewToolbarCommandRecord(
                command=command,
                executed=False,
                enabled=True,
                diagnostic="preview-widget-reset-unavailable",
            )
        reset()
        return PreviewToolbarCommandRecord(command=command, executed=True, enabled=True)
    if command in {"front", "top", "right"}:
        preset = getattr(widget, "apply_camera_preset", None)
        if not callable(preset):
            return PreviewToolbarCommandRecord(
                command=command,
                executed=False,
                enabled=True,
                diagnostic="preview-widget-preset-unavailable",
            )
        preset(command)
        return PreviewToolbarCommandRecord(command=command, executed=True, enabled=True)
    return PreviewToolbarCommandRecord(
        command=command,
        executed=False,
        enabled=True,
        diagnostic="unsupported-preview-toolbar-command",
    )


class PreviewRendererLifecycleWidget(QWidget):
    """Widget host that owns one long-lived embedded preview renderer."""

    renderCommandApplied = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        _ensure_widget_app()
        super().__init__(parent)
        self.setObjectName("embeddedPreviewSurface")
        self.setMinimumSize(360, 260)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._status = QLabel("No fixture selected.", self)
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._status)
        self._plotter = None
        self._previewer = None
        self._renderer_state = PreviewRendererLifecycleState()
        self._payload_state = PreviewWidgetPayloadState()
        self._current_datasets: list[Mesh | Polyline] = []
        self._display_options = PreviewDisplayOptions()
        self._render_queue = PreviewRenderCommandQueue()
        self._render_drain_scheduled = False

    @property
    def renderer_state(self) -> PreviewRendererLifecycleState:
        return self._renderer_state

    @property
    def payload_state(self) -> PreviewWidgetPayloadState:
        return self._payload_state

    def ensure_renderer(
        self,
        *,
        renderer_factory: Callable[[QWidget], object] | None = None,
        previewer_factory: Callable[[], object] | None = None,
    ) -> object:
        if self._plotter is not None:
            return self._plotter
        plotter = renderer_factory(self) if renderer_factory is not None else self._default_renderer_factory()
        self._layout.addWidget(plotter, 1)
        self._plotter = plotter
        self._previewer = None
        self._renderer_state = PreviewRendererLifecycleState(created=True)
        self._apply_display_options_to_plotter()
        return self._plotter

    @property
    def display_options(self) -> PreviewDisplayOptions:
        return self._display_options

    def set_display_options(self, options: PreviewDisplayOptions) -> None:
        self._display_options = options
        self._apply_display_options_to_plotter()

    def enqueue_render_command(
        self,
        command: PreviewRenderCommand,
    ) -> PreviewRenderCommandResult:
        result = self._render_queue.enqueue(command)
        if not self._render_drain_scheduled:
            self._render_drain_scheduled = True
            QTimer.singleShot(0, self._drain_preview_render_queue)
        return result

    def _drain_preview_render_queue(self) -> None:
        self._render_drain_scheduled = False
        command = self._render_queue.pop_next()
        if command is not None:
            result = self._apply_render_command(command)
            self.renderCommandApplied.emit(result)
        if self._render_queue.state.pending_count and not self._render_drain_scheduled:
            self._render_drain_scheduled = True
            QTimer.singleShot(0, self._drain_preview_render_queue)

    def _apply_render_command(
        self,
        command: PreviewRenderCommand,
    ) -> PreviewRenderCommandResult:
        try:
            if command.kind is PreviewRenderCommandKind.PAYLOAD:
                if command.payload is None:
                    return PreviewRenderCommandResult(
                        command=command,
                        accepted=False,
                        status="missing-payload",
                        diagnostic="preview-render-command-missing-payload",
                    )
                if command.display_options is not None:
                    self._display_options = command.display_options
                state = self.set_preview_payload(command.payload)
                return PreviewRenderCommandResult(
                    command=command,
                    accepted=state.ready,
                    status="applied" if state.ready else "failed",
                    ready=state.ready,
                    diagnostic=state.diagnostic,
                )
            if command.kind is PreviewRenderCommandKind.DISPLAY:
                if command.display_options is None:
                    return PreviewRenderCommandResult(
                        command=command,
                        accepted=False,
                        status="missing-display-options",
                        diagnostic="preview-render-command-missing-display-options",
                    )
                self.set_display_options(command.display_options)
                return PreviewRenderCommandResult(
                    command=command,
                    accepted=True,
                    status="applied",
                    ready=self._payload_state.ready,
                )
            if command.kind is PreviewRenderCommandKind.RESET_CAMERA:
                self.reset_view()
                return PreviewRenderCommandResult(
                    command=command,
                    accepted=True,
                    status="applied",
                    ready=self._payload_state.ready,
                )
            if command.kind is PreviewRenderCommandKind.LOADING:
                message = command.message or "Loading preview..."
                self._status.setText(message)
                self._status.show()
                return PreviewRenderCommandResult(command=command, accepted=True, status="applied")
            if command.kind is PreviewRenderCommandKind.CLEAR:
                state = self.clear_preview(command.message or "No fixture selected.")
                return PreviewRenderCommandResult(
                    command=command,
                    accepted=True,
                    status="applied",
                    ready=state.ready,
                    diagnostic=state.diagnostic,
                )
            if command.kind is PreviewRenderCommandKind.FAILURE:
                message = command.message or "Preview unavailable."
                diagnostic = command.diagnostic or message
                state = self.clear_preview(message)
                return PreviewRenderCommandResult(
                    command=command,
                    accepted=False,
                    status="failed",
                    ready=state.ready,
                    diagnostic=diagnostic,
                )
        except Exception as exc:
            diagnostic = sanitize_error_text(str(exc) or exc.__class__.__name__)
            self.clear_preview(f"Preview unavailable: {diagnostic}")
            return PreviewRenderCommandResult(
                command=command,
                accepted=False,
                status="exception",
                diagnostic=diagnostic,
            )
        return PreviewRenderCommandResult(
            command=command,
            accepted=False,
            status="unsupported",
            diagnostic="unsupported-preview-render-command",
        )

    def clear_renderer_scene(self, message: str) -> None:
        if self._plotter is not None:
            self._plotter.clear()
        self._status.setText(message)
        self._status.show()

    def clear_preview(self, message: str = "No fixture selected.") -> PreviewWidgetPayloadState:
        self._current_datasets = []
        self._payload_state = PreviewWidgetPayloadState(diagnostic=message)
        self.clear_renderer_scene(message)
        return self._payload_state

    def reset_view(self) -> None:
        if self._plotter is None:
            return
        reset_camera = getattr(self._plotter, "reset_camera", None)
        if callable(reset_camera):
            reset_camera()
        reset_clipping = getattr(self._plotter, "reset_camera_clipping_range", None)
        if callable(reset_clipping):
            reset_clipping()
        render = getattr(self._plotter, "render", None)
        if callable(render):
            render()

    def set_preview_payload(
        self,
        payload: PreviewPayload,
        *,
        renderer_factory: Callable[[QWidget], object] | None = None,
        previewer_factory: Callable[[], object] | None = None,
    ) -> PreviewWidgetPayloadState:
        if payload.payload_path is None:
            return self._fail_payload(payload, "preview-payload-missing-path")
        try:
            datasets = _load_payload_datasets(payload.payload_path)
            plotter = self.ensure_renderer(
                renderer_factory=renderer_factory,
                previewer_factory=previewer_factory,
            )
            self._status.hide()
            set_datasets = getattr(plotter, "set_datasets", None)
            if not callable(set_datasets):
                raise RuntimeError("preview-renderer-missing-set-datasets")
            set_datasets(datasets)
            self._apply_display_options_to_plotter()
            self._current_datasets = list(datasets)
            self._payload_state = PreviewWidgetPayloadState(
                generation=payload.request.generation,
                fixture_id=payload.request.fixture_id,
                ready=True,
            )
            return self._payload_state
        except Exception as exc:
            return self._fail_payload(payload, str(exc) or exc.__class__.__name__)

    def dispose_renderer(self) -> None:
        self._render_queue.clear()
        self._render_drain_scheduled = False
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
        self._renderer_state = PreviewRendererLifecycleState(disposed=True)

    def _fail_payload(self, payload: PreviewPayload, diagnostic: str) -> PreviewWidgetPayloadState:
        self._payload_state = PreviewWidgetPayloadState(
            generation=payload.request.generation,
            fixture_id=payload.request.fixture_id,
            diagnostic=sanitize_error_text(diagnostic),
        )
        self._current_datasets = []
        self._status.setText(f"Preview unavailable: {self._payload_state.diagnostic}")
        self._status.show()
        return self._payload_state

    def _default_renderer_factory(self):
        if _should_use_pyvistaqt_preview():
            return PyVistaQtPreviewSurface(self)
        return SoftwarePreviewSurface(self)

    def _default_previewer_factory(self):
        return None

    def _apply_display_options_to_plotter(self) -> None:
        if self._plotter is None:
            return
        set_display_options = getattr(self._plotter, "set_display_options", None)
        if callable(set_display_options):
            set_display_options(self._display_options)


class SoftwarePreviewSurface(QWidget):
    """Small Qt-only mesh preview surface for the workbench."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 260)
        self.setMouseTracking(True)
        self._datasets: tuple[Mesh | Polyline, ...] = ()
        self._rotation_x = -0.35
        self._rotation_y = 0.45
        self._zoom = 1.0
        self._last_pos = None
        self._wheel_zoom_direction: int | None = None
        self._wheel_zoom_reverse_delta_units = 0.0
        self._geometry: _PreparedPreviewGeometry | None = None
        self._display_options = PreviewDisplayOptions()

    @property
    def display_options(self) -> PreviewDisplayOptions:
        return self._display_options

    def set_display_options(self, options: PreviewDisplayOptions) -> None:
        self._display_options = options
        self.update()

    def clear(self) -> None:
        self._datasets = ()
        self._geometry = None
        self.update()

    def set_datasets(self, datasets: tuple[Mesh | Polyline, ...]) -> None:
        self._datasets = tuple(datasets)
        self._geometry = _prepare_preview_geometry(self._datasets)
        self.update()

    def reset_camera(self) -> None:
        self._rotation_x = -0.35
        self._rotation_y = 0.45
        self._zoom = 1.0
        self.update()

    def reset_camera_clipping_range(self) -> None:
        return

    def render(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], QPainter) and not kwargs:
            return super().render(args[0], QPoint(0, 0))
        if args or kwargs:
            return super().render(*args, **kwargs)
        self.update()
        return None

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._display_options.show_gradient_background:
            gradient = QLinearGradient(0.0, 0.0, 0.0, float(max(self.height(), 1)))
            gradient.setColorAt(0.0, _SOFTWARE_PREVIEW_BACKGROUND_TOP)
            gradient.setColorAt(1.0, _SOFTWARE_PREVIEW_BACKGROUND)
            painter.fillRect(self.rect(), gradient)
        else:
            painter.fillRect(self.rect(), _SOFTWARE_PREVIEW_BACKGROUND)
        if self._geometry is None:
            return
        projected = _project_prepared_geometry(
            self._geometry,
            width=max(self.width(), 1),
            height=max(self.height(), 1),
            rotation_x=self._rotation_x,
            rotation_y=self._rotation_y,
            zoom=self._zoom,
            options=self._display_options,
        )
        for line in projected.grid_lines:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(line.color, line.width))
            painter.drawLine(line.start, line.end)

        edge_pen = QPen(_SOFTWARE_PREVIEW_EDGE, 1.2)
        painter.setPen(Qt.PenStyle.NoPen)
        for face in projected.faces:
            if self._display_options.show_object_fill:
                painter.setBrush(QColor(face.color))
                painter.drawPolygon(face.polygon)
            if face.edge_segments:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(edge_pen)
                for start, end in face.edge_segments:
                    painter.drawLine(start, end)
                painter.setPen(Qt.PenStyle.NoPen)
            if face.triangle_segments:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#f7d6ba"), 0.8))
                for start, end in face.triangle_segments:
                    painter.drawLine(start, end)
                painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(edge_pen)
        for polyline in projected.polylines:
            if polyline.closed:
                painter.drawPolygon(polyline.polygon)
            else:
                painter.drawPolyline(polyline.polygon)
        for line in projected.axis_lines:
            painter.setPen(QPen(line.color, line.width))
            painter.drawLine(line.start, line.end)

    def mousePressEvent(self, event) -> None:
        self._last_pos = event.position()

    def mouseMoveEvent(self, event) -> None:
        if self._last_pos is None:
            self._last_pos = event.position()
            return
        pos = event.position()
        delta = pos - self._last_pos
        self._last_pos = pos
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._rotation_y += delta.x() * 0.01
            self._rotation_x += delta.y() * 0.01
            self.update()

    def mouseReleaseEvent(self, _event) -> None:
        self._last_pos = None

    def wheelEvent(self, event) -> None:
        decision = _wheel_zoom_decision(
            event,
            self._wheel_zoom_direction,
            self._wheel_zoom_reverse_delta_units,
        )
        self._wheel_zoom_direction = decision.latched_direction
        self._wheel_zoom_reverse_delta_units = decision.reverse_delta_units
        if decision.direction == 0:
            event.accept()
            return
        self._zoom = _clamped_zoom(self._zoom, 1.1 if decision.direction > 0 else 0.9)
        self.update()
        event.accept()


class PyVistaQtPreviewSurface(QWidget):
    """Reference Review adapter for the shared Qt preview surface."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._display_options = PreviewDisplayOptions()
        self._datasets: tuple[Mesh | Polyline, ...] = ()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._surface = QtPreviewSurface(
            self,
            config=QtPreviewSurfaceConfig.workbench_default(),
        )
        layout.addWidget(self._surface)

    @property
    def display_options(self) -> PreviewDisplayOptions:
        return self._display_options

    @property
    def plotter(self):
        return self._surface.plotter

    def set_display_options(self, options: PreviewDisplayOptions) -> None:
        self._display_options = options
        if self._datasets:
            self._surface.replace_scene(
                _datasets_for_display_options(self._datasets, options),
                apply_options=_preview_scene_apply_options(options, align_camera=False),
                align_camera=False,
            )
        else:
            self._surface.set_apply_options(
                _preview_scene_apply_options(options, align_camera=False)
            )

    def set_datasets(self, datasets: tuple[Mesh | Polyline, ...]) -> None:
        self._datasets = tuple(datasets)
        self._surface.replace_scene(
            _datasets_for_display_options(self._datasets, self._display_options),
            apply_options=_preview_scene_apply_options(self._display_options, align_camera=True),
            align_camera=True,
        )

    def clear(self) -> None:
        self._datasets = ()
        self._surface.clear()

    def reset_camera(self) -> None:
        self._surface.reset_camera()

    def reset_camera_clipping_range(self) -> None:
        self._surface.reset_camera_clipping_range()

    def render(self, *args, **kwargs):
        return self._surface.render(*args, **kwargs)

    def close(self) -> bool:
        self._surface.close()
        return super().close()


def _should_use_pyvistaqt_preview() -> bool:
    return (
        os.environ.get("IMPRESSION_REFERENCE_REVIEW_FORCE_SOFTWARE") != "1"
        and qt_preview_supported_environment()
    )


def _apply_pyvistaqt_scene(
    scene_controller: object,
    plotter: object,
    datasets: tuple[Mesh | Polyline, ...],
    options: PreviewDisplayOptions,
    *,
    align_camera: bool,
) -> None:
    apply_qt_preview_scene(
        scene_controller,
        plotter,
        _datasets_for_display_options(datasets, options),
        _preview_scene_apply_options(options, align_camera=align_camera),
    )


def _preview_scene_apply_options(
    options: PreviewDisplayOptions,
    *,
    align_camera: bool,
) -> PreviewSceneApplyOptions:
    return PreviewSceneApplyOptions(
        show_edges=options.show_triangle_wireframe,
        face_edges=options.show_object_edges,
        show_bounds=options.show_bounds_grid,
        show_axes=options.show_axis_triad,
        align_camera=align_camera,
    )


def _datasets_for_display_options(
    datasets: tuple[Mesh | Polyline, ...],
    options: PreviewDisplayOptions,
) -> tuple[Mesh | Polyline, ...]:
    if options.color_mode == COLOR_MODE_AUTHORED:
        return datasets
    prepared: list[Mesh | Polyline] = []
    for dataset in datasets:
        if isinstance(dataset, Mesh):
            prepared.append(
                Mesh(
                    vertices=np.asarray(dataset.vertices, dtype=float),
                    faces=np.asarray(dataset.faces, dtype=int),
                    color=None,
                    face_colors=None,
                    metadata=dict(dataset.metadata),
                )
            )
        else:
            prepared.append(
                Polyline(
                    np.asarray(dataset.points, dtype=float),
                    closed=dataset.closed,
                    color=None,
                )
            )
    return tuple(prepared)


def _project_datasets(
    datasets: tuple[Mesh | Polyline, ...],
    *,
    width: int,
    height: int,
    rotation_x: float,
    rotation_y: float,
    zoom: float,
    options: PreviewDisplayOptions | None = None,
) -> _ProjectedPreviewScene:
    geometry = _prepare_preview_geometry(datasets)
    if geometry is None:
        return _ProjectedPreviewScene(faces=(), polylines=(), grid_lines=(), axis_lines=())
    return _project_prepared_geometry(
        geometry,
        width=width,
        height=height,
        rotation_x=rotation_x,
        rotation_y=rotation_y,
        zoom=zoom,
        options=options,
    )


def _clamped_zoom(current_zoom: float, factor: float) -> float:
    return min(4.0, max(0.25, current_zoom * factor))


def _wheel_zoom_direction(event: object, latched_direction: int | None) -> int:
    return _wheel_zoom_decision(event, latched_direction, 0.0).direction


def _wheel_zoom_decision(
    event: object,
    latched_direction: int | None,
    reverse_delta_units: float,
) -> _WheelZoomDecision:
    delta_units = _wheel_delta_units(event)
    raw_direction = _direction_from_delta(delta_units)
    phase = _wheel_phase(event)
    if phase == Qt.ScrollPhase.ScrollEnd:
        return _WheelZoomDecision(0, None, 0.0)
    if phase == Qt.ScrollPhase.NoScrollPhase:
        return _WheelZoomDecision(raw_direction, None, 0.0)
    if raw_direction == 0:
        return _WheelZoomDecision(0, latched_direction, reverse_delta_units)
    if latched_direction is None:
        return _WheelZoomDecision(raw_direction, raw_direction, 0.0)
    if raw_direction == latched_direction:
        return _WheelZoomDecision(raw_direction, latched_direction, 0.0)

    reverse_delta_units += abs(delta_units)
    if reverse_delta_units >= _WHEEL_ZOOM_REVERSE_THRESHOLD_UNITS:
        return _WheelZoomDecision(raw_direction, raw_direction, 0.0)
    return _WheelZoomDecision(latched_direction, latched_direction, reverse_delta_units)


def _wheel_delta_units(event: object) -> float:
    delta = _wheel_delta_y(event, "pixelDelta")
    if delta != 0:
        return float(delta) / 24.0
    return float(_wheel_delta_y(event, "angleDelta")) / 120.0


def _direction_from_delta(delta: float) -> int:
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def _wheel_delta_y(event: object, method_name: str) -> int:
    method = getattr(event, method_name, None)
    if not callable(method):
        return 0
    delta = method()
    y = getattr(delta, "y", None)
    if not callable(y):
        return 0
    return int(y())


def _wheel_phase(event: object) -> Qt.ScrollPhase:
    phase = getattr(event, "phase", None)
    if not callable(phase):
        return Qt.ScrollPhase.NoScrollPhase
    value = phase()
    return value if isinstance(value, Qt.ScrollPhase) else Qt.ScrollPhase.NoScrollPhase


def _prepare_preview_geometry(
    datasets: tuple[Mesh | Polyline, ...],
) -> _PreparedPreviewGeometry | None:
    points = []
    for dataset in datasets:
        if isinstance(dataset, Mesh) and len(dataset.vertices):
            points.append(np.asarray(dataset.vertices, dtype=float))
        elif isinstance(dataset, Polyline) and len(dataset.points):
            points.append(np.asarray(dataset.points, dtype=float))
    if not points:
        return None
    all_points = np.vstack(points)
    bounds_min = all_points.min(axis=0)
    bounds_max = all_points.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0
    span = float(np.max(bounds_max - bounds_min))
    items: list[_PreparedMesh | _PreparedPolyline] = []
    for dataset in datasets:
        if isinstance(dataset, Polyline):
            items.append(
                _PreparedPolyline(
                    np.asarray(dataset.points, dtype=float),
                    _polyline_qcolor(dataset),
                    dataset.closed,
                )
            )
            continue
        items.append(
            _prepare_mesh_geometry(
                dataset,
                np.asarray(dataset.vertices, dtype=float),
                np.asarray(dataset.faces, dtype=int),
            )
        )
    return _PreparedPreviewGeometry(
        center=center,
        span=span,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        items=tuple(items),
    )


def _prepare_mesh_geometry(mesh: Mesh, vertices: np.ndarray, faces: np.ndarray) -> _PreparedMesh:
    object_edges = _object_edge_keys(mesh)
    face_edges = []
    triangle_edges = []
    face_normals = []
    for face in faces:
        edge_pairs = []
        triangle_pairs = []
        face_indices = tuple(int(index) for index in face)
        for local_index, vertex_index in enumerate(face_indices):
            next_local_index = (local_index + 1) % len(face_indices)
            next_vertex_index = face_indices[next_local_index]
            triangle_pairs.append((local_index, next_local_index))
            edge_key = (
                min(vertex_index, next_vertex_index),
                max(vertex_index, next_vertex_index),
            )
            if edge_key in object_edges:
                edge_pairs.append((local_index, next_local_index))
        face_edges.append(_PreparedFaceEdges(tuple(edge_pairs)))
        triangle_edges.append(_PreparedFaceEdges(tuple(triangle_pairs)))
        normal = _face_normal(vertices, face)
        face_normals.append(
            normal if normal is not None else np.asarray((0.0, 0.0, 1.0))
        )
    return _PreparedMesh(
        vertices,
        faces,
        _mesh_authored_qcolor(mesh),
        _mesh_face_qcolors(mesh, len(faces)),
        tuple(face_edges),
        tuple(triangle_edges),
        np.asarray(face_normals, dtype=float),
    )


def _project_prepared_geometry(
    geometry: _PreparedPreviewGeometry,
    *,
    width: int,
    height: int,
    rotation_x: float,
    rotation_y: float,
    zoom: float,
    options: PreviewDisplayOptions | None = None,
) -> _ProjectedPreviewScene:
    options = options or PreviewDisplayOptions()
    center = geometry.center
    span = geometry.span
    scale = (min(width, height) * 0.72 * zoom) / max(span, 1e-6)
    rotation = _rotation_matrix(rotation_x, rotation_y)
    project_point = _make_projector(width, height, scale)

    faces: list[_ProjectedFace] = []
    polylines: list[_ProjectedPolyline] = []
    grid_lines = (
        _project_bounds_grid(geometry, rotation=rotation, project_point=project_point)
        if options.show_bounds_grid
        else ()
    )
    axis_lines = (
        _project_axis_triad(geometry, rotation=rotation, project_point=project_point)
        if options.show_axis_triad
        else ()
    )
    for item in geometry.items:
        if isinstance(item, _PreparedPolyline):
            if not options.show_polylines:
                continue
            rotated = (item.points - center) @ rotation.T
            polygon = QPolygonF([project_point(point) for point in rotated])
            polylines.append(
                _ProjectedPolyline(
                    float(np.mean(rotated[:, 2])) if len(rotated) else 0.0,
                    polygon,
                    item.color,
                    item.closed,
                )
            )
            continue

        vertices = (item.vertices - center) @ rotation.T
        normals = item.face_normals @ rotation.T
        for face_index, (face, prepared_edges, prepared_triangles, normal) in enumerate(
            zip(item.faces, item.face_edges, item.triangle_edges, normals)
        ):
            if len(face) < 3:
                continue
            face_points = vertices[face]
            polygon = QPolygonF([project_point(point) for point in face_points])
            edge_segments = []
            for local_index, next_local_index in (
                prepared_edges.edge_pairs if options.show_object_edges else ()
            ):
                edge_segments.append(
                    (
                        QPointF(polygon[local_index]),
                        QPointF(polygon[next_local_index]),
                    )
                )
            triangle_segments = []
            for local_index, next_local_index in (
                prepared_triangles.edge_pairs if options.show_triangle_wireframe else ()
            ):
                triangle_segments.append(
                    (
                        QPointF(polygon[local_index]),
                        QPointF(polygon[next_local_index]),
                    )
                )
            base_color = _display_face_color(item, face_index, options)
            faces.append(
                _ProjectedFace(
                    float(np.mean(face_points[:, 2])),
                    polygon,
                    _lit_color(base_color, normal, options.lighting_mode),
                    tuple(edge_segments),
                    tuple(triangle_segments),
                )
            )
    faces.sort(key=lambda item: item.depth)
    polylines.sort(key=lambda item: item.depth)
    return _ProjectedPreviewScene(
        faces=tuple(faces),
        polylines=tuple(polylines),
        grid_lines=grid_lines,
        axis_lines=axis_lines,
    )


def _object_edge_keys(mesh: Mesh, *, sharp_angle_degrees: float = 30.0) -> set[tuple[int, int]]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    normals = [_face_normal(vertices, face) for face in faces]
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        face_indices = tuple(int(index) for index in face)
        for local_index, vertex_index in enumerate(face_indices):
            next_vertex_index = face_indices[(local_index + 1) % len(face_indices)]
            edge_key = (
                min(vertex_index, next_vertex_index),
                max(vertex_index, next_vertex_index),
            )
            edge_faces.setdefault(edge_key, []).append(face_index)

    sharp_dot_threshold = math.cos(math.radians(sharp_angle_degrees))
    object_edges: set[tuple[int, int]] = set()
    for edge_key, adjacent_faces in edge_faces.items():
        if len(adjacent_faces) != 2:
            object_edges.add(edge_key)
            continue
        first = normals[adjacent_faces[0]]
        second = normals[adjacent_faces[1]]
        if first is None or second is None or float(np.dot(first, second)) < sharp_dot_threshold:
            object_edges.add(edge_key)
    return object_edges


def _make_projector(width: int, height: int, scale: float):
    def project(point: np.ndarray) -> QPointF:
        return QPointF(width / 2.0 + point[0] * scale, height / 2.0 - point[1] * scale)

    return project


def _project_bounds_grid(
    geometry: _PreparedPreviewGeometry,
    *,
    rotation: np.ndarray,
    project_point,
) -> tuple[_ProjectedLine, ...]:
    bounds_min = geometry.bounds_min
    bounds_max = geometry.bounds_max
    center = geometry.center
    span = max(geometry.span, 1e-6)
    z = bounds_min[2]
    x_min = bounds_min[0]
    x_max = bounds_max[0]
    y_min = bounds_min[1]
    y_max = bounds_max[1]
    if abs(x_max - x_min) < 1e-9:
        x_min -= span * 0.5
        x_max += span * 0.5
    if abs(y_max - y_min) < 1e-9:
        y_min -= span * 0.5
        y_max += span * 0.5

    lines: list[_ProjectedLine] = []
    steps = 4
    for index in range(steps + 1):
        t = index / steps
        x = x_min + (x_max - x_min) * t
        y = y_min + (y_max - y_min) * t
        x_start = (np.asarray((x, y_min, z), dtype=float) - center) @ rotation.T
        x_end = (np.asarray((x, y_max, z), dtype=float) - center) @ rotation.T
        y_start = (np.asarray((x_min, y, z), dtype=float) - center) @ rotation.T
        y_end = (np.asarray((x_max, y, z), dtype=float) - center) @ rotation.T
        width = 1.4 if index in {0, steps} else 0.8
        lines.append(
            _ProjectedLine(
                float((x_start[2] + x_end[2]) * 0.5),
                project_point(x_start),
                project_point(x_end),
                _SOFTWARE_PREVIEW_GRID,
                width,
            )
        )
        lines.append(
            _ProjectedLine(
                float((y_start[2] + y_end[2]) * 0.5),
                project_point(y_start),
                project_point(y_end),
                _SOFTWARE_PREVIEW_GRID,
                width,
            )
        )
    return tuple(lines)


def _project_axis_triad(
    geometry: _PreparedPreviewGeometry,
    *,
    rotation: np.ndarray,
    project_point,
) -> tuple[_ProjectedLine, ...]:
    center = geometry.center
    origin = np.asarray(
        (geometry.bounds_min[0], geometry.bounds_min[1], geometry.bounds_min[2]),
        dtype=float,
    )
    axis_length = max(geometry.span, 1e-6) * 0.28
    specs = (
        (np.asarray((axis_length, 0.0, 0.0), dtype=float), _SOFTWARE_PREVIEW_AXIS_X),
        (np.asarray((0.0, axis_length, 0.0), dtype=float), _SOFTWARE_PREVIEW_AXIS_Y),
        (np.asarray((0.0, 0.0, axis_length), dtype=float), _SOFTWARE_PREVIEW_AXIS_Z),
    )
    projected_origin = (origin - center) @ rotation.T
    lines = []
    for offset, color in specs:
        projected_end = (origin + offset - center) @ rotation.T
        lines.append(
            _ProjectedLine(
                float((projected_origin[2] + projected_end[2]) * 0.5),
                project_point(projected_origin),
                project_point(projected_end),
                color,
                2.0,
            )
        )
    return tuple(lines)


def _shaded_color(color: QColor, normal: np.ndarray) -> QColor:
    color = _opaque_color(color)
    normal = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return _opaque_color(color)
    normal = normal / norm
    light = np.asarray((-0.35, 0.45, 0.82), dtype=float)
    light = light / float(np.linalg.norm(light))
    diffuse = max(0.0, float(np.dot(normal, light)))
    backlight = max(0.0, float(np.dot(-normal, light))) * 0.18
    rim = (1.0 - abs(float(normal[2]))) * 0.18
    intensity = min(1.18, 0.48 + diffuse * 0.46 + backlight + rim)
    return QColor.fromRgbF(
        min(1.0, color.redF() * intensity),
        min(1.0, color.greenF() * intensity),
        min(1.0, color.blueF() * intensity),
        1.0,
    )


def _camera_light_color(color: QColor, normal: np.ndarray) -> QColor:
    color = _opaque_color(color)
    normal = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return _opaque_color(color)
    normal = normal / norm
    light = np.asarray((0.0, 0.0, 1.0), dtype=float)
    diffuse = max(0.0, float(np.dot(normal, light)))
    intensity = min(1.25, 0.62 + diffuse * 0.5)
    return QColor.fromRgbF(
        min(1.0, color.redF() * intensity),
        min(1.0, color.greenF() * intensity),
        min(1.0, color.blueF() * intensity),
        1.0,
    )


def _lit_color(color: QColor, normal: np.ndarray, lighting_mode: str) -> QColor:
    if lighting_mode == LIGHTING_MODE_FLAT:
        return _opaque_color(color)
    if lighting_mode == LIGHTING_MODE_CAMERA:
        return _camera_light_color(color, normal)
    if lighting_mode == LIGHTING_MODE_FACE_NORMALS:
        return _shaded_color(color, normal)
    return _opaque_color(color)


def _display_face_color(
    mesh: _PreparedMesh,
    face_index: int,
    options: PreviewDisplayOptions,
) -> QColor:
    if options.color_mode == COLOR_MODE_AUTHORED:
        if face_index < len(mesh.face_colors) and mesh.face_colors[face_index] is not None:
            return _opaque_color(mesh.face_colors[face_index])
        if mesh.authored_color is not None:
            return _opaque_color(mesh.authored_color)
    return QColor(_SOFTWARE_PREVIEW_DEFAULT_MESH)


def _opaque_color(color: QColor) -> QColor:
    result = QColor(color)
    result.setAlphaF(1.0)
    return result


def _face_normal(vertices: np.ndarray, face: np.ndarray) -> np.ndarray | None:
    if len(face) < 3:
        return None
    face_points = vertices[face]
    origin = face_points[0]
    for first_index in range(1, len(face_points) - 1):
        first = face_points[first_index] - origin
        for second_index in range(first_index + 1, len(face_points)):
            second = face_points[second_index] - origin
            normal = np.cross(first, second)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-9:
                return normal / norm
    return None


def _rotation_matrix(rotation_x: float, rotation_y: float) -> np.ndarray:
    cx = math.cos(rotation_x)
    sx = math.sin(rotation_x)
    cy = math.cos(rotation_y)
    sy = math.sin(rotation_y)
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    ry = np.asarray(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    return ry @ rx


def _mesh_qcolor(mesh: Mesh) -> QColor:
    color = mesh.color
    if color is None:
        return QColor(_SOFTWARE_PREVIEW_DEFAULT_MESH)
    return _color_value_qcolor(color) or QColor(_SOFTWARE_PREVIEW_DEFAULT_MESH)


def _mesh_authored_qcolor(mesh: Mesh) -> QColor | None:
    if mesh.color is None:
        return None
    return _color_value_qcolor(mesh.color)


def _mesh_face_qcolors(mesh: Mesh, face_count: int) -> tuple[QColor | None, ...]:
    if mesh.face_colors is None:
        return tuple(None for _ in range(face_count))
    colors = np.asarray(mesh.face_colors, dtype=float)
    result: list[QColor | None] = []
    for face_index in range(face_count):
        if face_index >= len(colors):
            result.append(None)
        else:
            result.append(_color_value_qcolor(colors[face_index]))
    return tuple(result)


def _color_value_qcolor(color: object) -> QColor | None:
    if isinstance(color, QColor):
        return QColor(color)
    if isinstance(color, str):
        qcolor = QColor(color)
        return qcolor if qcolor.isValid() else None
    values = tuple(float(value) for value in color)
    if len(values) >= 3 and all(0.0 <= value <= 1.0 for value in values[:3]):
        alpha = values[3] if len(values) >= 4 else 1.0
        return QColor.fromRgbF(values[0], values[1], values[2], alpha)
    if len(values) >= 3:
        alpha = int(values[3]) if len(values) >= 4 else 255
        return QColor(int(values[0]), int(values[1]), int(values[2]), alpha)
    return None


def _polyline_qcolor(polyline: Polyline) -> QColor:
    color = polyline.color
    if color is None:
        return QColor(_SOFTWARE_PREVIEW_DEFAULT_MESH)
    if isinstance(color, str):
        return QColor(color)
    values = tuple(float(value) for value in color)
    if len(values) >= 3 and all(0.0 <= value <= 1.0 for value in values[:3]):
        alpha = values[3] if len(values) >= 4 else 1.0
        return QColor.fromRgbF(values[0], values[1], values[2], alpha)
    if len(values) >= 3:
        alpha = int(values[3]) if len(values) >= 4 else 255
        return QColor(int(values[0]), int(values[1]), int(values[2]), alpha)
    return QColor(_SOFTWARE_PREVIEW_DEFAULT_MESH)


def _ensure_widget_app() -> QApplication:
    global _PREVIEW_WIDGET_APP
    app = QApplication.instance()
    if app is not None:
        return app
    configure_qt_preview_surface_format()
    _PREVIEW_WIDGET_APP = QApplication([])
    return _PREVIEW_WIDGET_APP


def _load_payload_datasets(path: Path) -> tuple[Mesh | Polyline, ...]:
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != PREVIEW_PAYLOAD_FORMAT:
        raise ValueError("unsupported-preview-payload-format")
    datasets = tuple(_decode_dataset(item) for item in payload.get("datasets", ()))
    if not datasets:
        raise ValueError("empty-preview-payload")
    return datasets


def _decode_dataset(item: object) -> Mesh | Polyline:
    if not isinstance(item, dict):
        raise ValueError("invalid-preview-payload-dataset")
    kind = item.get("kind")
    if kind == "mesh":
        return Mesh(
            vertices=np.asarray(item.get("vertices", ()), dtype=float),
            faces=np.asarray(item.get("faces", ()), dtype=int),
            color=None if item.get("color") is None else tuple(item["color"]),
            face_colors=None
            if item.get("face_colors") is None
            else np.asarray(item["face_colors"], dtype=float),
            metadata=dict(item.get("metadata", {})),
        )
    if kind == "polyline":
        return Polyline(
            points=np.asarray(item.get("points", ()), dtype=float),
            closed=bool(item.get("closed", False)),
            color=None if item.get("color") is None else tuple(item["color"]),
        )
    raise ValueError("unsupported-preview-payload-dataset")
