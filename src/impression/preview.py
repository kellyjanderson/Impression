from __future__ import annotations

import os
import queue
import threading
import math
import importlib
import importlib.util
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
from rich.console import Console
from rich.panel import Panel
from watchfiles import Change, watch

from impression._config import UnitSettings, get_unit_settings
from impression._vtk_runtime import ensure_vtk_runtime
from impression.modeling._color import COLOR_CELL_DATA, get_mesh_color



def _prepare_vtk_runtime() -> None:
    ensure_vtk_runtime()

SceneFactory = Callable[[], object]


class PreviewBackendError(RuntimeError):
    """Raised when a preview backend cannot run."""


def _collect_datasets_from_scene(scene: object, pv_module) -> List[object]:
    datasets: List[object] = []

    def visit(item: object) -> None:
        if item is None:
            return

        if isinstance(item, pv_module.MultiBlock):
            for block in item:
                visit(block)
            return

        if isinstance(item, pv_module.DataSet):
            datasets.append(item)
            return

        if isinstance(item, (list, tuple, set)):
            for value in item:
                visit(value)
            return

        raise PreviewBackendError(
            "Model build() must return PyVista datasets (e.g., pv.Cube(), pv.Sphere(), or a list of them)."
        )

    visit(scene)
    if not datasets:
        raise PreviewBackendError("Scene did not produce any PyVista datasets.")
    return datasets


class PyVistaPreviewer:
    """Render scenes using PyVista and provide optional hot reload support."""

    def __init__(self, console: Console, unit_settings: UnitSettings | None = None):
        self.console = console
        self._pv = None
        self._unit_settings = unit_settings or get_unit_settings()

    def show(
        self,
        scene_factory: SceneFactory,
        initial_scene: object,
        model_path: Path,
        watch_files: bool,
        target_fps: int,
        screenshot_path: Path | None = None,
        show_edges: bool = False,
        face_edges: bool = False,
    ) -> None:
        pv = self._ensure_backend()
        datasets = self.collect_datasets(initial_scene)
        plotter = pv.Plotter(window_size=(1280, 800))
        self._configure_plotter(plotter)
        self._apply_scene(
            plotter,
            datasets,
            show_edges=show_edges,
            face_edges=face_edges,
            align_camera=True,
        )

        if screenshot_path is not None:
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            plotter.show(title="Impression Preview", auto_close=True, screenshot=str(screenshot_path))
            plotter.close()
            return

        if not watch_files:
            plotter.show(title="Impression Preview")
            plotter.close()
            return

        reload_queue: queue.Queue[float] = queue.Queue()
        stop_event = threading.Event()
        watcher_thread = threading.Thread(
            target=self._watch_model_file,
            args=(model_path, reload_queue, stop_event),
            name="impression-watch",
            daemon=True,
        )
        watcher_thread.start()

        def process_queue() -> None:
            reload_requested = False
            while True:
                try:
                    reload_queue.get_nowait()
                    reload_requested = True
                except queue.Empty:
                    break
            if not reload_requested:
                return

            self.console.print(f"[yellow]Reloading {model_path}…[/yellow]")
            datasets = self.collect_datasets(scene_factory())
            self._apply_scene(
                plotter,
                datasets,
                show_edges=show_edges,
                face_edges=face_edges,
                align_camera=False,
            )
            plotter.render()
            self.console.print(f"[green]Reloaded {model_path}[/green]")

        interval_seconds = max(1.0 / max(target_fps, 1), 0.05)
        callback_cleanup = None
        if watch_files:
            def guarded_process_queue() -> None:
                try:
                    process_queue()
                except Exception as exc:  # pragma: no cover - surfaced via console
                    panel = Panel.fit(str(exc), title="Reload failed", style="red")
                    self.console.print(panel)

            callback_cleanup = self._install_timer_callback(plotter, guarded_process_queue, interval_seconds)

        try:
            plotter.show(title="Impression Preview", auto_close=False)
        finally:
            stop_event.set()
            if callback_cleanup is not None:
                callback_cleanup()
            plotter.close()

    # Internal helpers -----------------------------------------------------

    def _ensure_backend(self):
        if self._pv is None:
            _prepare_vtk_runtime()
            try:
                import pyvista as pv
            except ImportError as exc:  # pragma: no cover - runtime dep
                raise PreviewBackendError(
                    "PyVista is required for previewing. Install impression with `pip install -e .`."
                ) from exc
            pv.set_plot_theme("document")
            self._pv = pv
        return self._pv

    def collect_datasets(self, scene: object) -> List[object]:
        """Return all PyVista datasets contained within a scene object."""

        pv = self._ensure_backend()
        return _collect_datasets_from_scene(scene, pv)

    @property
    def unit_name(self) -> str:
        return self._unit_settings.name

    @property
    def unit_label(self) -> str:
        return self._unit_settings.label

    @property
    def unit_scale_to_mm(self) -> float:
        return self._unit_settings.scale_to_mm

    def combine_to_polydata(self, datasets: Iterable[object]):
        """Return a single PolyData mesh representing the collection."""

        pv = self._ensure_backend()
        datasets = list(datasets)
        if not datasets:
            raise PreviewBackendError("Cannot merge an empty dataset collection.")

        if len(datasets) == 1:
            merged = datasets[0].copy()
        else:
            block = pv.MultiBlock()
            for mesh in datasets:
                block.append(mesh)
            merged = block.combine()

        poly = merged.extract_geometry()
        if hasattr(poly, "clean"):
            poly = poly.clean()
        if hasattr(poly, "triangulate"):
            poly = poly.triangulate()
        return poly

    def _configure_plotter(self, plotter) -> None:
        plotter.set_background("#090c10", top="#1b2333")
        plotter.enable_eye_dome_lighting()
        plotter.add_axes(interactive=True)
        self._show_bounds_with_units(plotter)

    def _apply_scene(
        self,
        plotter,
        datasets: Iterable[object],
        show_edges: bool,
        face_edges: bool,
        align_camera: bool = False,
    ) -> None:
        datasets = list(datasets)
        color_cycle = ["#6ab0ff", "#f58f7c", "#9cdcfe", "#fadb5f", "#9ae6b4", "#d4b5ff"]
        edge_color = "#cdd7ff"
        edge_angle = 60.0
        plotter.clear()
        self._show_bounds_with_units(plotter)
        plotter.add_axes(interactive=True)

        for index, mesh in enumerate(datasets):
            cell_colors = mesh.cell_data.get(COLOR_CELL_DATA)
            if cell_colors is not None and len(cell_colors) == mesh.n_cells:
                scalars = np.asarray(cell_colors)
                rgba_mode = scalars.shape[1] >= 4
                plotter.add_mesh(
                    mesh,
                    name=f"mesh-{index}",
                    show_edges=show_edges,
                    scalars=scalars,
                    rgba=rgba_mode,
                    smooth_shading=True,
                    specular=0.2,
                )
                if face_edges:
                    self._add_feature_edges(plotter, mesh, edge_color, edge_angle, index)
                continue

            color_info = get_mesh_color(mesh)
            if color_info:
                rgb, alpha = color_info
                color = rgb
                opacity = alpha
            else:
                color = color_cycle[index % len(color_cycle)]
                opacity = 1.0

            plotter.add_mesh(
                mesh,
                name=f"mesh-{index}",
                show_edges=show_edges,
                color=color,
                opacity=opacity,
                smooth_shading=True,
                specular=0.2,
            )
            if face_edges:
                self._add_feature_edges(plotter, mesh, edge_color, edge_angle, index)

        if align_camera:
            self._reset_camera(plotter, datasets)

    def _add_feature_edges(self, plotter, mesh, color: str, angle: float, index: int) -> None:
        if not hasattr(mesh, "extract_feature_edges"):
            return
        edges = mesh.extract_feature_edges(angle=angle)
        if edges.n_cells == 0:
            return
        plotter.add_mesh(
            edges,
            name=f"mesh-{index}-edges",
            color=color,
            line_width=1.0,
            render_lines_as_tubes=False,
        )

    def _reset_camera(self, plotter, datasets: Iterable[object]) -> None:
        bounds = None
        for mesh in datasets:
            mesh_bounds = mesh.bounds
            if bounds is None:
                bounds = list(mesh_bounds)
            else:
                bounds[0] = min(bounds[0], mesh_bounds[0])
                bounds[1] = max(bounds[1], mesh_bounds[1])
                bounds[2] = min(bounds[2], mesh_bounds[2])
                bounds[3] = max(bounds[3], mesh_bounds[3])
                bounds[4] = min(bounds[4], mesh_bounds[4])
                bounds[5] = max(bounds[5], mesh_bounds[5])

        if bounds is None:
            return

        x_center = (bounds[0] + bounds[1]) / 2.0
        y_center = (bounds[2] + bounds[3]) / 2.0
        z_center = (bounds[4] + bounds[5]) / 2.0

        diag = math.sqrt(
            (bounds[1] - bounds[0]) ** 2
            + (bounds[3] - bounds[2]) ** 2
            + (bounds[5] - bounds[4]) ** 2
        )
        distance = max(diag, 1.0) * 1.2

        camera_pos = (x_center, y_center + distance, z_center)
        focal_point = (x_center, y_center, z_center)
        view_up = (0.0, 0.0, 1.0)
        plotter.camera_position = [camera_pos, focal_point, view_up]

    def _install_timer_callback(
        self,
        plotter,
        callback: Callable[[], None],
        interval_seconds: float,
    ):
        """Install a repeating timer callback compatible with the current PyVista backend."""

        add_callback = getattr(plotter, "add_callback", None)
        if callable(add_callback):
            callback_id = add_callback(callback, interval=interval_seconds)

            def cleanup() -> None:
                remove_callback = getattr(plotter, "remove_callback", None)
                if callable(remove_callback):
                    remove_callback(callback_id)

            return cleanup

        interactor = getattr(plotter, "iren", None)
        if interactor is None:
            setup_interactor = getattr(plotter, "_setup_interactor", None)
            if callable(setup_interactor):
                setup_interactor()
                interactor = getattr(plotter, "iren", None)
        if interactor is None:
            raise PreviewBackendError("PyVista interactor unavailable; cannot attach timer callbacks.")

        duration_ms = max(int(interval_seconds * 1000), 10)
        timer_id = interactor.create_timer(duration=duration_ms, repeating=True)

        def timer_handler(*_: object) -> None:
            callback()

        observer_id = interactor.add_observer("TimerEvent", timer_handler)

        def cleanup() -> None:
            try:
                if observer_id is not None:
                    interactor.remove_observer(observer_id)
            except Exception:
                pass
            try:
                if timer_id is not None:
                    interactor.destroy_timer(timer_id)
            except Exception:
                pass

        return cleanup

    def _show_bounds_with_units(self, plotter) -> None:
        label = self._unit_settings.label
        plotter.show_bounds(
            grid="front",
            color="#5a677d",
            xlabel=f"X ({label})",
            ylabel=f"Y ({label})",
            zlabel=f"Z ({label})",
        )


    def _watch_model_file(
        self,
        model_path: Path,
        reload_queue: "queue.Queue[float]",
        stop_event: threading.Event,
    ) -> None:
        resolved_model = model_path.resolve()
        watch_root = resolved_model if resolved_model.is_dir() else resolved_model.parent

        for changes in watch(str(watch_root), stop_event=stop_event, debounce=300):
            if stop_event.is_set():
                return

            for change, changed_path in changes:
                if Change.deleted == change and Path(changed_path) == resolved_model:
                    reload_queue.put_nowait(0.0)
                    break
                if Path(changed_path).resolve() == resolved_model:
                    reload_queue.put_nowait(0.0)
                    break
