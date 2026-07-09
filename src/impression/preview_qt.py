"""Qt host for the shared Impression preview renderer."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from typing import Iterable

from PySide6.QtWidgets import QVBoxLayout, QWidget

from impression.mesh import Mesh, Polyline
from impression.preview import (
    PreviewControllerOptions,
    PreviewInteractionPolicy,
    PreviewSceneApplyOptions,
    PreviewSceneController,
    PreviewStyle,
)


@dataclass(frozen=True)
class QtPreviewSurfaceConfig:
    """Configuration for a Qt-embedded shared preview surface."""

    controller_options: PreviewControllerOptions = field(default_factory=PreviewControllerOptions)
    apply_options: PreviewSceneApplyOptions = field(default_factory=PreviewSceneApplyOptions)
    allow_offscreen: bool = False
    auto_update: bool | float = False
    qvtk_base: str | None = None

    @classmethod
    def workbench_default(cls) -> "QtPreviewSurfaceConfig":
        return cls(
            controller_options=PreviewControllerOptions(
                style=PreviewStyle.workbench_default(),
                interaction=PreviewInteractionPolicy(
                    show_bounds=False,
                    show_axes=False,
                    enable_eye_dome_lighting=False,
                ),
            ),
            apply_options=PreviewSceneApplyOptions(
                show_edges=False,
                face_edges=False,
                show_bounds=False,
                show_axes=False,
                align_camera=True,
            ),
            allow_offscreen=False,
            auto_update=False,
            qvtk_base="QOpenGLWidget",
        )


def qt_preview_supported_environment(*, allow_offscreen: bool = False) -> bool:
    """Return whether a native Qt preview surface should be created now."""

    return allow_offscreen or os.environ.get("QT_QPA_PLATFORM") != "offscreen"


def apply_qt_preview_scene(
    scene_controller: PreviewSceneController,
    plotter: object,
    datasets: Iterable[Mesh | Polyline],
    options: PreviewSceneApplyOptions,
) -> None:
    """Apply shared preview scene semantics to a caller-owned Qt plotter."""

    scene_controller.apply_scene(
        plotter,
        datasets,
        show_edges=options.show_edges,
        face_edges=options.face_edges,
        show_bounds=options.show_bounds,
        show_axes=options.show_axes,
        align_camera=options.align_camera,
        show_object_fill=options.show_object_fill,
        show_polylines=options.show_polylines,
        smooth_shading=options.smooth_shading,
        lighting=options.lighting,
        specular=options.specular,
        background=options.background,
        background_top=options.background_top,
    )


def preview_scene_options_for_camera_state(
    options: PreviewSceneApplyOptions,
    *,
    align_camera: bool,
    camera_aligned: bool,
) -> PreviewSceneApplyOptions:
    """Return scene options preserving display flags while resolving camera alignment."""

    return replace(options, align_camera=align_camera and not camera_aligned)


def configure_qvtk_backend(base: str | None) -> None:
    """Configure the QVTK widget backend before pyvistaqt imports it."""

    if base is None:
        return
    if "pyvistaqt.rwi" in sys.modules:
        return
    import vtkmodules.qt

    vtkmodules.qt.QVTKRWIBase = base


def configure_qt_preview_surface_format() -> None:
    """Configure the OpenGL surface format used by embedded preview widgets."""

    from PySide6.QtGui import QSurfaceFormat

    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    fmt.setVersion(2, 1)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)


class QtPreviewSurface(QWidget):
    """Reusable Qt-embedded PyVista preview surface.

    The surface owns one long-lived QtInteractor and routes all scene changes
    through PreviewSceneController. Host applications provide datasets and
    apply options; they do not create renderers or directly manipulate VTK.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        config: QtPreviewSurfaceConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 260)
        self._config = config or QtPreviewSurfaceConfig()
        if not qt_preview_supported_environment(allow_offscreen=self._config.allow_offscreen):
            raise RuntimeError("qt-preview-offscreen-disabled")
        self._scene_controller = PreviewSceneController(options=self._config.controller_options)
        self._apply_options = self._config.apply_options
        self._datasets: tuple[Mesh | Polyline, ...] = ()
        self._camera_aligned = False

        configure_qvtk_backend(self._config.qvtk_base)
        from pyvistaqt import QtInteractor

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._plotter = QtInteractor(
            self,
            off_screen=os.environ.get("QT_QPA_PLATFORM") == "offscreen",
            auto_update=self._config.auto_update,
        )
        layout.addWidget(self._plotter)
        self._configure_plotter()

    @property
    def plotter(self):
        return self._plotter

    @property
    def apply_options(self) -> PreviewSceneApplyOptions:
        return self._apply_options

    def set_apply_options(self, options: PreviewSceneApplyOptions) -> None:
        self._apply_options = options
        if self._datasets:
            self._apply_scene(align_camera=False)

    def set_datasets(
        self,
        datasets: Iterable[Mesh | Polyline],
        *,
        align_camera: bool = True,
    ) -> None:
        self._datasets = tuple(datasets)
        if align_camera:
            self._camera_aligned = False
        self._apply_scene(align_camera=align_camera)

    def replace_scene(
        self,
        datasets: Iterable[Mesh | Polyline],
        *,
        apply_options: PreviewSceneApplyOptions,
        align_camera: bool = True,
    ) -> None:
        self._apply_options = apply_options
        self._datasets = tuple(datasets)
        if align_camera:
            self._camera_aligned = False
        self._apply_scene(align_camera=align_camera)

    def clear(self) -> None:
        self._datasets = ()
        self._camera_aligned = False
        self._plotter.clear()
        self._configure_plotter()
        self._plotter.render()

    def reset_camera(self) -> None:
        self._scene_controller.reset_camera(self._plotter, self._datasets)
        self._camera_aligned = True
        self._plotter.render()

    def reset_camera_clipping_range(self) -> None:
        reset = getattr(self._plotter, "reset_camera_clipping_range", None)
        if callable(reset):
            reset()

    def render(self, *args, **kwargs):
        return self._plotter.render(*args, **kwargs)

    def close(self) -> bool:
        self._plotter.close()
        return super().close()

    def _configure_plotter(self) -> None:
        self._scene_controller.configure_plotter(
            self._plotter,
            show_bounds=self._apply_options.show_bounds,
            show_axes=self._apply_options.show_axes,
        )

    def _apply_scene(self, *, align_camera: bool) -> None:
        options = preview_scene_options_for_camera_state(
            self._apply_options,
            align_camera=align_camera,
            camera_aligned=self._camera_aligned,
        )
        self._configure_plotter()
        apply_qt_preview_scene(
            self._scene_controller,
            self._plotter,
            self._datasets,
            options,
        )
        if align_camera:
            self._camera_aligned = True
        self._plotter.render()
