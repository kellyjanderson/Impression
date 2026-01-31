from __future__ import annotations

import os
import queue
import threading
import math
import traceback
import sys
import signal
from pathlib import Path
from typing import Callable, Iterable, List, MutableMapping

import numpy as np
from rich.console import Console
from rich.panel import Panel
from watchfiles import Change, watch

from impression._config import UnitSettings, get_unit_settings
from impression._vtk_runtime import ensure_vtk_runtime
from impression.mesh import Mesh, Polyline, analyze_mesh, combine_meshes, mesh_to_pyvista
from impression.modeling._color import get_mesh_color
from impression.modeling.group import MeshGroup
from impression.modeling.drawing2d import Path2D, Profile2D
from impression.modeling.path3d import Path3D





def _prepare_vtk_runtime() -> None:
    ensure_vtk_runtime()

SceneFactory = Callable[[], object]
ModelPathState = MutableMapping[str, Path]


class PreviewBackendError(RuntimeError):
    """Raised when a preview backend cannot run."""


def _pyvista_safe_mode() -> bool:
    value = os.environ.get("IMPRESSION_PYVISTA_SAFE")
    if value is None:
        return False
    return value.lower() not in {"0", "false", "no"}


def _format_exception(exc: BaseException) -> str:
    """Return a formatted traceback string for display."""
    return "".join(traceback.format_exception(exc))


def _collect_datasets_from_scene(scene: object) -> List[Mesh | Polyline]:
    datasets: List[Mesh | Polyline] = []

    def visit(item: object) -> None:
        if item is None:
            return

        if isinstance(item, MeshGroup) or (hasattr(item, "to_meshes") and callable(getattr(item, "to_meshes"))):
            for mesh in item.to_meshes():
                visit(mesh)
            return

        if isinstance(item, Mesh):
            datasets.append(item)
            return

        if isinstance(item, Polyline):
            datasets.append(item)
            return

        if isinstance(item, Path2D):
            datasets.append(item.to_polyline())
            return

        if isinstance(item, Profile2D):
            datasets.extend(item.to_polylines())
            return

        if isinstance(item, Path3D):
            datasets.append(item.to_polyline())
            return

        if isinstance(item, (list, tuple, set)):
            for value in item:
                visit(value)
            return

        raise PreviewBackendError(
            "Model build() must return internal meshes (e.g., impression.modeling primitives or a list of them)."
        )

    visit(scene)
    if not datasets:
        raise PreviewBackendError("Scene did not produce any meshes.")
    return datasets


class PyVistaPreviewer:
    """Render scenes using PyVista and provide optional hot reload support."""

    def __init__(self, console: Console, unit_settings: UnitSettings | None = None):
        self.console = console
        self._pv = None
        self._unit_settings = unit_settings or get_unit_settings()
        self._home_camera = None

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
        control_file: Path | None = None,
        model_path_state: ModelPathState | None = None,
    ) -> None:
        datasets = []
        if initial_scene is not None:
            try:
                datasets = self.collect_datasets(initial_scene)
            except Exception as exc:
                if watch_files:
                    panel = Panel.fit(_format_exception(exc), title="Initial build failed — watching for changes", style="red")
                    self.console.print(panel)
                else:
                    raise

        pv = self._ensure_backend()
        plotter = pv.Plotter(window_size=(1280, 800))
        self._configure_plotter(plotter)
        if datasets:
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

        model_state = model_path_state or {"path": model_path}
        reload_queue: queue.Queue[float] = queue.Queue()
        stop_event = threading.Event()
        watcher_thread = None
        watch_roots: list[Path] = []

        def _resolve_control_path() -> Path | None:
            if control_file is None:
                return None
            path = control_file.expanduser()
            try:
                return path.resolve()
            except OSError:
                return path

        control_path = _resolve_control_path()

        def _read_control_file() -> Path | None:
            if control_path is None or not control_path.exists():
                return None
            try:
                lines = control_path.read_text().splitlines()
            except OSError:
                return None
            if not lines:
                return None
            candidate_text = lines[-1].strip()
            if not candidate_text:
                return None
            candidate = Path(candidate_text).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate)
            try:
                return candidate.resolve()
            except OSError:
                return candidate

        def _watch_roots_for(path: Path) -> list[Path]:
            roots = []
            resolved = path.resolve()
            roots.append(resolved if resolved.is_dir() else resolved.parent)
            if control_path is not None:
                roots.append(control_path.parent)
            return roots

        def _start_watcher(current_path: Path) -> None:
            nonlocal stop_event, watcher_thread, watch_roots
            stop_event.set()
            stop_event = threading.Event()
            watch_roots = _watch_roots_for(current_path)
            watcher_thread = threading.Thread(
                target=self._watch_model_file,
                args=(model_state, reload_queue, stop_event, watch_roots, control_path),
                name="impression-watch",
                daemon=True,
            )
            watcher_thread.start()

        _start_watcher(model_state["path"])
        previous_handler = None

        def request_reload() -> None:
            try:
                reload_queue.put_nowait(0.0)
            except queue.Full:
                return

        def request_reload_with_message(message: str) -> None:
            self.console.print(message)
            request_reload()

        def maybe_switch_model() -> bool:
            new_path = _read_control_file()
            if new_path is None:
                return False
            if not new_path.exists():
                self.console.print(f"[red]Control file points to missing model: {new_path}[/red]")
                return False
            current = model_state["path"]
            if new_path.resolve() == current.resolve():
                return False
            model_state["path"] = new_path
            self.console.print(f"[cyan]Switched preview model to {new_path}[/cyan]")
            new_roots = _watch_roots_for(new_path)
            if set(new_roots) != set(watch_roots):
                _start_watcher(new_path)
            return True

        if hasattr(signal, "SIGUSR1"):
            def _signal_reload(_signum, _frame) -> None:  # pragma: no cover - signal path
                if maybe_switch_model():
                    request_reload_with_message("[yellow]Reload requested (switch).[/yellow]")
                else:
                    request_reload_with_message("[yellow]Reload requested (SIGUSR1).[/yellow]")

            previous_handler = signal.getsignal(signal.SIGUSR1)
            signal.signal(signal.SIGUSR1, _signal_reload)

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
            maybe_switch_model()
            current_path = model_state["path"]
            self.console.print(f"[yellow]Reloading {current_path}…[/yellow]")
            datasets = self.collect_datasets(scene_factory())
            self._apply_scene(
                plotter,
                datasets,
                show_edges=show_edges,
                face_edges=face_edges,
                align_camera=False,
            )
            plotter.render()
            self.console.print(f"[green]Reloaded {current_path}[/green]")

        interval_seconds = max(1.0 / max(target_fps, 1), 0.05)
        callback_cleanup = None
        if watch_files:
            def guarded_process_queue() -> None:
                try:
                    process_queue()
                except Exception as exc:  # pragma: no cover - surfaced via console
                    panel = Panel.fit(_format_exception(exc), title="Reload failed", style="red")
                    self.console.print(panel)

            callback_cleanup = self._install_timer_callback(plotter, guarded_process_queue, interval_seconds)

            def _handle_key_reload() -> None:
                request_reload_with_message("[yellow]Reload requested (R).[/yellow]")

            plotter.add_key_event("r", _handle_key_reload)
            if control_path is not None:
                def _handle_key_switch() -> None:
                    if maybe_switch_model():
                        request_reload_with_message("[yellow]Reload requested (switch).[/yellow]")
                    else:
                        request_reload_with_message("[yellow]Reload requested (S).[/yellow]")

                plotter.add_key_event("s", _handle_key_switch)
                plotter.add_key_event("S", _handle_key_switch)

        try:
            plotter.show(title="Impression Preview", auto_close=False)
        finally:
            stop_event.set()
            if callback_cleanup is not None:
                callback_cleanup()
            if previous_handler is not None and hasattr(signal, "SIGUSR1"):
                signal.signal(signal.SIGUSR1, previous_handler)
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

    def collect_datasets(self, scene: object) -> List[Mesh | Polyline]:
        """Return all meshes contained within a scene object."""

        datasets = _collect_datasets_from_scene(scene)
        self._log_mesh_analysis(datasets)
        return datasets

    @property
    def unit_name(self) -> str:
        return self._unit_settings.name

    @property
    def unit_label(self) -> str:
        return self._unit_settings.label

    @property
    def unit_scale_to_mm(self) -> float:
        return self._unit_settings.scale_to_mm

    def combine_to_mesh(self, datasets: Iterable[Mesh | Polyline]) -> Mesh:
        """Return a single Mesh representing the collection."""

        datasets = [mesh for mesh in datasets if isinstance(mesh, Mesh)]
        if not datasets:
            raise PreviewBackendError("Cannot merge an empty mesh collection.")
        if len(datasets) == 1:
            return datasets[0].copy()
        return combine_meshes(datasets)

    def _configure_plotter(self, plotter) -> None:
        plotter.set_background("#090c10", top="#1b2333")
        safe_mode = _pyvista_safe_mode()
        if not safe_mode:
            plotter.enable_eye_dome_lighting()
            plotter.add_axes(interactive=True)
            self._show_bounds_with_units(plotter)
            self._install_home_button(plotter)

    def _apply_scene(
        self,
        plotter,
        datasets: Iterable[Mesh | Polyline],
        show_edges: bool,
        face_edges: bool,
        align_camera: bool = False,
    ) -> None:
        datasets = list(datasets)
        color_cycle = ["#6ab0ff", "#f58f7c", "#9cdcfe", "#fadb5f", "#9ae6b4", "#d4b5ff"]
        edge_color = "#cdd7ff"
        edge_angle = 60.0
        plotter.clear()
        if not _pyvista_safe_mode():
            self._show_bounds_with_units(plotter)
            plotter.add_axes(interactive=True)

        for index, mesh in enumerate(datasets):
            if isinstance(mesh, Polyline):
                pv_mesh = self._polyline_to_pyvista(mesh)
                plotter.add_mesh(
                    pv_mesh,
                    name=f"mesh-{index}",
                    color=mesh.color or "#9aa6bf",
                    line_width=2.0,
                    render_lines_as_tubes=False,
                )
                continue

            pv_mesh = mesh_to_pyvista(mesh)
            cell_colors = mesh.face_colors
            if cell_colors is not None and len(cell_colors) == mesh.n_faces:
                scalars = np.asarray(cell_colors)
                rgba_mode = scalars.shape[1] >= 4
                rgb_mode = scalars.shape[1] == 3
                plotter.add_mesh(
                    pv_mesh,
                    name=f"mesh-{index}",
                    show_edges=show_edges,
                    scalars=scalars,
                    rgb=rgb_mode,
                    rgba=rgba_mode,
                    smooth_shading=True,
                    specular=0.2,
                )
                if face_edges:
                    self._add_feature_edges(plotter, pv_mesh, edge_color, edge_angle, index)
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
                pv_mesh,
                name=f"mesh-{index}",
                show_edges=show_edges,
                color=color,
                opacity=opacity,
                smooth_shading=True,
                specular=0.2,
            )
            if face_edges:
                self._add_feature_edges(plotter, pv_mesh, edge_color, edge_angle, index)

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

    def _reset_camera(self, plotter, datasets: Iterable[Mesh | Polyline]) -> None:
        bounds = None
        for mesh in datasets:
            if isinstance(mesh, Polyline):
                if mesh.points.size == 0:
                    continue
                mins = mesh.points.min(axis=0)
                maxs = mesh.points.max(axis=0)
                mesh_bounds = (
                    float(mins[0]),
                    float(maxs[0]),
                    float(mins[1]),
                    float(maxs[1]),
                    float(mins[2]),
                    float(maxs[2]),
                )
            else:
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
        distance = max(diag, 1.0) * 1.4

        camera_pos = (
            x_center + distance * 0.6,
            y_center + distance * 0.6,
            z_center + distance * 0.5,
        )
        focal_point = (x_center, y_center, z_center)
        view_up = (0.0, 0.0, 1.0)
        plotter.camera_position = [camera_pos, focal_point, view_up]
        self._home_camera = plotter.camera_position

    def _install_home_button(self, plotter) -> None:
        """Add a simple 'home' button to reset the camera to the stored default."""

        def go_home():
            if self._home_camera is None:
                return
            plotter.camera_position = self._home_camera
            plotter.reset_camera_clipping_range()
            plotter.render()

        try:
            plotter.add_button_widget(
                go_home,
                position=(10, 10),
                size=40,
                color_on="white",
                color_off="white",
                style="modern",
                tooltip="Reset view",
            )
        except Exception:
            return


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
            xtitle=f"X ({label})",
            ytitle=f"Y ({label})",
            ztitle=f"Z ({label})",
        )

    def _log_mesh_analysis(self, datasets: Iterable[Mesh | Polyline]) -> None:
        for index, mesh in enumerate(datasets):
            if isinstance(mesh, Polyline):
                continue
            analysis = analyze_mesh(mesh)
            issues = analysis.issues()
            if not issues:
                continue
            issue_text = ", ".join(issues)
            self.console.print(f"[yellow]Mesh {index} analysis: {issue_text}.[/yellow]")

    def _polyline_to_pyvista(self, polyline: Polyline):
        pv = self._ensure_backend()
        pts = polyline.points
        if polyline.closed and len(pts) > 1 and not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        n_pts = len(pts)
        if n_pts == 0:
            return pv.PolyData()
        cells = np.hstack(([n_pts], np.arange(n_pts)))
        return pv.PolyData(pts, cells, deep=True)

    def _watch_model_file(
        self,
        model_path_state: ModelPathState,
        reload_queue: "queue.Queue[float]",
        stop_event: threading.Event,
        watch_roots: list[Path],
        control_path: Path | None,
    ) -> None:
        watch_paths = [str(root) for root in watch_roots]
        resolved_control = control_path.resolve() if control_path is not None else None

        for changes in watch(*watch_paths, stop_event=stop_event, debounce=300):
            if stop_event.is_set():
                return

            for change, changed_path in changes:
                changed = Path(changed_path).resolve()
                resolved_model = model_path_state["path"].resolve()
                if resolved_control is not None and changed == resolved_control:
                    reload_queue.put_nowait(0.0)
                    break
                if Change.deleted == change and changed == resolved_model:
                    reload_queue.put_nowait(0.0)
                    break
                if changed == resolved_model:
                    reload_queue.put_nowait(0.0)
                    break
