"""Qt preview widget lifecycle host for Reference Review."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

os.environ.setdefault("QT_OPENGL", "desktop")

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
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
class PreviewCameraDisplayMetadata:
    width: int
    height: int
    payload_generation: int | None = None
    fixture_id: str | None = None
    display_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "display_options", MappingProxyType(dict(self.display_options)))


@dataclass(frozen=True)
class PreviewImageFileReference:
    path: Path
    image_format: str
    mime_type: str
    byte_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True)
class PreviewImageCaptureRecord:
    file_reference: PreviewImageFileReference | None
    metadata: PreviewCameraDisplayMetadata | None = None
    diagnostic: str | None = None

    @property
    def ok(self) -> bool:
        return self.file_reference is not None and self.diagnostic is None


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
    captureDiagnostic = Signal(object)

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

    def capture_visible_preview_image(
        self,
        output_dir: Path | str,
        *,
        basename: str = "preview-capture",
        image_format: str = "png",
    ) -> PreviewImageCaptureRecord:
        if self._render_drain_scheduled or self._render_queue.state.pending_count:
            return self._capture_diagnostic("preview-capture-busy")
        if self._plotter is None or not self._payload_state.ready:
            return self._capture_diagnostic("preview-capture-missing-render")
        image_format = image_format.lower()
        if image_format != "png":
            return self._capture_diagnostic("preview-capture-unsupported-format")
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = _unique_capture_path(output_root, basename, image_format)
        target = self._plotter if isinstance(self._plotter, QWidget) else self
        pixmap = target.grab()
        if pixmap.isNull():
            return self._capture_diagnostic("preview-capture-empty-image")
        if not pixmap.save(str(output_path), image_format.upper()):
            return self._capture_diagnostic("preview-capture-save-failed")
        metadata = self._capture_metadata(target)
        reference = PreviewImageFileReference(
            output_path,
            image_format,
            "image/png",
            output_path.stat().st_size,
        )
        return PreviewImageCaptureRecord(reference, metadata)

    def set_preview_payload(
        self,
        payload: PreviewPayload,
        *,
        renderer_factory: Callable[[QWidget], object] | None = None,
        previewer_factory: Callable[[], object] | None = None,
    ) -> PreviewWidgetPayloadState:
        if payload.diagnostic is not None:
            return self._fail_payload(payload, payload.diagnostic.message)
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

    def _capture_metadata(self, target: QWidget) -> PreviewCameraDisplayMetadata:
        return PreviewCameraDisplayMetadata(
            width=max(target.width(), 0),
            height=max(target.height(), 0),
            payload_generation=self._payload_state.generation,
            fixture_id=self._payload_state.fixture_id,
            display_options={
                "show_object_fill": self._display_options.show_object_fill,
                "show_object_edges": self._display_options.show_object_edges,
                "show_gradient_background": self._display_options.show_gradient_background,
                "color_mode": self._display_options.color_mode,
                "lighting_mode": self._display_options.lighting_mode,
            },
        )

    def _capture_diagnostic(self, code: str) -> PreviewImageCaptureRecord:
        record = PreviewImageCaptureRecord(None, diagnostic=code)
        self.captureDiagnostic.emit(record)
        return record

    def dispose_renderer(self) -> None:
        self._render_queue.clear()
        self._render_drain_scheduled = False
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
        self._renderer_state = PreviewRendererLifecycleState(disposed=True)

    def _fail_payload(self, payload: PreviewPayload, diagnostic: str) -> PreviewWidgetPayloadState:
        sanitized = sanitize_error_text(diagnostic)
        if (
            self._payload_state.ready
            and self._payload_state.fixture_id == payload.request.fixture_id
            and self._current_datasets
        ):
            self._payload_state = PreviewWidgetPayloadState(
                generation=self._payload_state.generation,
                fixture_id=self._payload_state.fixture_id,
                ready=True,
                diagnostic=sanitized,
            )
            self._status.setText(f"Preview stale: {sanitized}")
            self._status.show()
            return self._payload_state
        self._payload_state = PreviewWidgetPayloadState(
            generation=payload.request.generation,
            fixture_id=payload.request.fixture_id,
            diagnostic=sanitized,
        )
        self._current_datasets = []
        self._status.setText(f"Preview unavailable: {self._payload_state.diagnostic}")
        self._status.show()
        return self._payload_state

    def _default_renderer_factory(self):
        if not qt_preview_supported_environment():
            raise RuntimeError("pyvistaqt-preview-unavailable")
        return PyVistaQtPreviewSurface(self)

    def _default_previewer_factory(self):
        return None

    def _apply_display_options_to_plotter(self) -> None:
        if self._plotter is None:
            return
        set_display_options = getattr(self._plotter, "set_display_options", None)
        if callable(set_display_options):
            set_display_options(self._display_options)


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

def _unique_capture_path(output_dir: Path, basename: str, image_format: str) -> Path:
    safe_basename = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in basename
    )
    candidate = output_dir / f"{safe_basename or 'preview-capture'}.{image_format}"
    index = 2
    while candidate.exists():
        candidate = output_dir / f"{safe_basename or 'preview-capture'}-{index:02d}.{image_format}"
        index += 1
    return candidate


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
    lighting = options.lighting_mode != LIGHTING_MODE_FLAT
    smooth_shading = options.lighting_mode == LIGHTING_MODE_CAMERA
    return PreviewSceneApplyOptions(
        show_edges=options.show_triangle_wireframe,
        face_edges=options.show_object_edges,
        show_bounds=options.show_bounds_grid,
        show_axes=options.show_axis_triad,
        align_camera=align_camera,
        show_object_fill=options.show_object_fill,
        show_polylines=options.show_polylines,
        smooth_shading=smooth_shading,
        lighting=lighting,
        lighting_profile=options.lighting_mode,
        specular=0.2 if options.lighting_mode == LIGHTING_MODE_CAMERA else 0.0,
        background="#07111f",
        background_top="#10223a" if options.show_gradient_background else None,
    )


def _datasets_for_display_options(
    datasets: tuple[Mesh | Polyline, ...],
    options: PreviewDisplayOptions,
) -> tuple[Mesh | Polyline, ...]:
    if options.color_mode == COLOR_MODE_AUTHORED:
        return tuple(
            dataset
            for dataset in datasets
            if not isinstance(dataset, Polyline) or options.show_polylines
        )
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
            if not options.show_polylines:
                continue
            prepared.append(
                Polyline(
                    np.asarray(dataset.points, dtype=float),
                    closed=dataset.closed,
                    color=None,
                )
            )
    return tuple(prepared)


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
