from __future__ import annotations

import os
import queue
import threading
import math
import traceback
import sys
import signal
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, MutableMapping

import numpy as np
from rich.console import Console
from rich.panel import Panel
from watchfiles import Change, watch

from impression._config import UnitSettings, get_unit_settings
from impression._vtk_runtime import ensure_vtk_runtime
from impression.mesh import Mesh, Polyline, analyze_mesh, combine_meshes, mesh_to_pyvista
from impression.io import write_stl
from impression.modeling._color import get_mesh_color
from impression.modeling.group import MeshGroup
from impression.modeling.drawing2d import Path2D
from impression.modeling.path3d import Path3D
from impression.modeling.topology import Region, Section





def _prepare_vtk_runtime() -> None:
    ensure_vtk_runtime()

SceneFactory = Callable[[], object]
ModelPathState = MutableMapping[str, Path]

def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    n = 1
    while True:
        candidate = parent / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
        n += 1


class PreviewBackendError(RuntimeError):
    """Raised when a preview backend cannot run."""


@dataclass(frozen=True)
class PreviewStyle:
    """Visual defaults for Impression preview scenes."""

    background: str = "#090c10"
    background_top: str | None = "#1b2333"
    light_background: str = "#e6e9ef"
    bounds_color: str = "#5a677d"
    default_polyline_color: str = "#9aa6bf"
    feature_edge_color: str = "#cdd7ff"
    feature_edge_angle: float = 60.0
    color_cycle: tuple[str, ...] = (
        "#6ab0ff",
        "#f58f7c",
        "#9cdcfe",
        "#fadb5f",
        "#9ae6b4",
        "#d4b5ff",
    )

    @classmethod
    def workbench_default(cls) -> "PreviewStyle":
        """Return the Reference Review Workbench preview color defaults."""

        return cls(
            background="#07111f",
            background_top=None,
            color_cycle=("#f4a261",),
            default_polyline_color="#f4a261",
            feature_edge_color="#ffd2a8",
        )


@dataclass(frozen=True)
class PreviewInteractionPolicy:
    """Host-neutral interaction and plotter decoration defaults."""

    show_bounds: bool = True
    show_axes: bool = True
    enable_eye_dome_lighting: bool = True


@dataclass(frozen=True)
class PreviewControllerOptions:
    """Configuration passed to the shared preview controller."""

    style: PreviewStyle = field(default_factory=PreviewStyle)
    interaction: PreviewInteractionPolicy = field(default_factory=PreviewInteractionPolicy)


@dataclass(frozen=True)
class PreviewStyleDiagnostic:
    """Resolved plotter configuration reported by the preview controller."""

    background: str
    background_top: str | None
    show_bounds: bool
    show_axes: bool
    eye_dome_lighting: bool
    safe_mode: bool


@dataclass(frozen=True)
class PreviewSceneApplyOptions:
    """Scene application options for a caller-owned preview render surface."""

    show_edges: bool = False
    face_edges: bool = False
    show_bounds: bool = True
    show_axes: bool = True
    align_camera: bool = False
    show_object_fill: bool = True
    show_polylines: bool = True
    smooth_shading: bool = True
    lighting: bool = True
    lighting_profile: str = "camera"
    specular: float = 0.2
    background: str | None = None
    background_top: str | None = None


@dataclass(frozen=True)
class PreviewCameraDiagnostic:
    """Camera reset result for a preview scene."""

    bounds: tuple[float, float, float, float, float, float] | None
    camera_position: object | None


class PreviewSceneController:
    """Shared host-neutral preview behavior for CLI and embedded workbench previews."""

    def __init__(
        self,
        *,
        unit_settings: UnitSettings | None = None,
        options: PreviewControllerOptions | None = None,
        pyvista_provider: Callable[[], object] | None = None,
    ) -> None:
        self._unit_settings = unit_settings or get_unit_settings()
        self.options = options or PreviewControllerOptions()
        self._pyvista_provider = pyvista_provider
        self._home_camera = None

    @property
    def style(self) -> PreviewStyle:
        return self.options.style

    @property
    def interaction(self) -> PreviewInteractionPolicy:
        return self.options.interaction

    @property
    def home_camera(self):
        return self._home_camera

    def configure_plotter(
        self,
        plotter,
        *,
        show_bounds: bool | None = None,
        show_axes: bool | None = None,
    ) -> PreviewStyleDiagnostic:
        """Apply shared preview style and plotter decorations to a caller-owned plotter."""

        safe_mode = _pyvista_safe_mode()
        resolved_show_bounds = self.interaction.show_bounds if show_bounds is None else show_bounds
        resolved_show_axes = self.interaction.show_axes if show_axes is None else show_axes
        eye_dome = self.interaction.enable_eye_dome_lighting and not safe_mode

        style = self.style
        if style.background_top is None:
            plotter.set_background(style.background)
        else:
            plotter.set_background(style.background, top=style.background_top)

        if eye_dome:
            plotter.enable_eye_dome_lighting()
            if resolved_show_axes:
                plotter.add_axes(interactive=True)
            if resolved_show_bounds:
                self.show_bounds_with_units(plotter)

        return PreviewStyleDiagnostic(
            background=style.background,
            background_top=style.background_top,
            show_bounds=resolved_show_bounds and eye_dome,
            show_axes=resolved_show_axes and eye_dome,
            eye_dome_lighting=eye_dome,
            safe_mode=safe_mode,
        )

    def show_bounds_with_units(self, plotter) -> None:
        label = self._unit_settings.label
        plotter.show_bounds(
            grid="front",
            color=self.style.bounds_color,
            xtitle=f"X ({label})",
            ytitle=f"Y ({label})",
            ztitle=f"Z ({label})",
        )

    def apply_scene(
        self,
        plotter,
        datasets: Iterable[Mesh | Polyline],
        *,
        show_edges: bool = False,
        face_edges: bool = False,
        show_bounds: bool = True,
        show_axes: bool = True,
        align_camera: bool = False,
        show_object_fill: bool = True,
        show_polylines: bool = True,
        smooth_shading: bool = True,
        lighting: bool = True,
        lighting_profile: str = "camera",
        specular: float = 0.2,
        background: str | None = None,
        background_top: str | None = None,
    ) -> None:
        """Apply datasets to a caller-owned plotter using shared preview semantics."""

        datasets = list(datasets)
        style = self.style
        plotter.clear()
        self._clear_scene_decoration_state(plotter)
        if background is not None:
            if background_top is None:
                plotter.set_background(background)
            else:
                plotter.set_background(background, top=background_top)
        if not _pyvista_safe_mode():
            if show_bounds:
                self.show_bounds_with_units(plotter)
            if show_axes:
                plotter.add_axes(interactive=True)

        for index, mesh in enumerate(datasets):
            if isinstance(mesh, Polyline):
                if not show_polylines:
                    continue
                pv_mesh = self.polyline_to_pyvista(mesh)
                actor = plotter.add_mesh(
                    pv_mesh,
                    name=f"mesh-{index}",
                    color=mesh.color or style.default_polyline_color,
                    line_width=2.0,
                    render_lines_as_tubes=False,
                )
                self._configure_actor_lighting(
                    actor,
                    lighting=False,
                    lighting_profile="flat",
                )
                continue

            pv_mesh = mesh_to_pyvista(mesh)
            if not show_object_fill:
                if show_edges:
                    actor = plotter.add_mesh(
                        pv_mesh,
                        name=f"mesh-{index}-wireframe",
                        color=style.feature_edge_color,
                        style="wireframe",
                        line_width=1.0,
                        lighting=False,
                    )
                    self._configure_actor_lighting(
                        actor,
                        lighting=False,
                        lighting_profile="flat",
                    )
                if face_edges:
                    self.add_feature_edges(plotter, pv_mesh, index)
                continue

            cell_colors = mesh.face_colors
            if cell_colors is not None and len(cell_colors) == mesh.n_faces:
                scalars = np.asarray(cell_colors)
                rgba_mode = scalars.shape[1] >= 4
                rgb_mode = scalars.shape[1] == 3
                actor = plotter.add_mesh(
                    pv_mesh,
                    name=f"mesh-{index}",
                    show_edges=show_edges,
                    scalars=scalars,
                    rgb=rgb_mode,
                    rgba=rgba_mode,
                    smooth_shading=smooth_shading,
                    lighting=lighting,
                    specular=specular,
                )
                self._configure_actor_lighting(
                    actor,
                    lighting=lighting,
                    lighting_profile=lighting_profile,
                )
                if face_edges:
                    self.add_feature_edges(plotter, pv_mesh, index)
                continue

            color_info = get_mesh_color(mesh)
            if color_info:
                rgb, alpha = color_info
                color = rgb
                opacity = alpha
            else:
                color = style.color_cycle[index % len(style.color_cycle)]
                opacity = 1.0

            actor = plotter.add_mesh(
                pv_mesh,
                name=f"mesh-{index}",
                show_edges=show_edges,
                color=color,
                opacity=opacity,
                smooth_shading=smooth_shading,
                lighting=lighting,
                specular=specular,
            )
            self._configure_actor_lighting(
                actor,
                lighting=lighting,
                lighting_profile=lighting_profile,
            )
            if face_edges:
                self.add_feature_edges(plotter, pv_mesh, index)

        if align_camera:
            self.reset_camera(plotter, datasets)

    def _clear_scene_decoration_state(self, plotter) -> None:
        for method_name in (
            "hide_axes_all",
            "hide_axes",
            "remove_bounds_axes",
            "remove_bounding_box",
            "disable_eye_dome_lighting",
        ):
            method = getattr(plotter, method_name, None)
            if callable(method):
                method()

    def _configure_actor_lighting(
        self,
        actor,
        *,
        lighting: bool,
        lighting_profile: str,
    ) -> None:
        get_property = getattr(actor, "GetProperty", None)
        if not callable(get_property):
            return
        prop = get_property()
        if prop is None:
            return
        if lighting:
            lighting_on = getattr(prop, "LightingOn", None)
            if callable(lighting_on):
                lighting_on()
        else:
            lighting_off = getattr(prop, "LightingOff", None)
            if callable(lighting_off):
                lighting_off()
        if lighting_profile == "camera":
            interpolation = getattr(prop, "SetInterpolationToPhong", None)
        else:
            interpolation = getattr(prop, "SetInterpolationToFlat", None)
        if callable(interpolation):
            interpolation()

    def add_feature_edges(self, plotter, mesh, index: int) -> None:
        if not hasattr(mesh, "extract_feature_edges"):
            return
        edges = self._extract_feature_edges(mesh)
        if edges is None:
            return
        if edges.n_cells == 0:
            return
        plotter.add_mesh(
            edges,
            name=f"mesh-{index}-edges",
            color=self.style.feature_edge_color,
            line_width=1.0,
            render_lines_as_tubes=False,
        )

    def _extract_feature_edges(self, mesh):
        extractor = getattr(mesh, "extract_feature_edges", None)
        if not callable(extractor):
            return None
        try:
            return extractor(angle=self.style.feature_edge_angle)
        except TypeError as exc:
            if "angle" not in str(exc):
                raise
        try:
            return extractor(
                boundary_edges=True,
                feature_edges=True,
                non_manifold_edges=True,
                manifold_edges=False,
                feature_angle=self.style.feature_edge_angle,
            )
        except TypeError:
            return None

    def reset_camera(self, plotter, datasets: Iterable[Mesh | Polyline]) -> PreviewCameraDiagnostic:
        bounds = self._combined_bounds(datasets)
        if bounds is None:
            return PreviewCameraDiagnostic(bounds=None, camera_position=None)

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
        return PreviewCameraDiagnostic(bounds=tuple(bounds), camera_position=plotter.camera_position)

    def polyline_to_pyvista(self, polyline: Polyline):
        pv = self._ensure_pyvista()
        pts = polyline.points
        if polyline.closed and len(pts) > 1 and not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        n_pts = len(pts)
        if n_pts == 0:
            return pv.PolyData()
        cells = np.hstack(([n_pts], np.arange(n_pts)))
        return pv.PolyData(pts, cells, deep=True)

    def _ensure_pyvista(self):
        if self._pyvista_provider is not None:
            return self._pyvista_provider()
        _prepare_vtk_runtime()
        try:
            import pyvista as pv
        except ImportError as exc:  # pragma: no cover - runtime dep
            raise PreviewBackendError(
                "PyVista is required for previewing. Install impression with `pip install -e .`."
            ) from exc
        return pv

    def _combined_bounds(
        self,
        datasets: Iterable[Mesh | Polyline],
    ) -> list[float] | None:
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
        return bounds


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

    def _loop_to_polyline(loop_points: np.ndarray) -> Polyline:
        pts = np.asarray(loop_points, dtype=float).reshape(-1, 2)
        pts3 = np.column_stack([pts, np.zeros((pts.shape[0], 1), dtype=float)])
        return Polyline(pts3, closed=True, color=None)

    def _region_to_polylines(region: Region) -> list[Polyline]:
        normalized = region.normalized()
        polylines = [_loop_to_polyline(normalized.outer.points)]
        polylines.extend(_loop_to_polyline(hole.points) for hole in normalized.holes)
        return polylines

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

        if hasattr(item, "to_polylines") and callable(getattr(item, "to_polylines")):
            datasets.extend(item.to_polylines())
            return

        if isinstance(item, Region):
            datasets.extend(_region_to_polylines(item))
            return

        if isinstance(item, Section):
            for region in item.normalized().regions:
                datasets.extend(_region_to_polylines(region))
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

    def __init__(
        self,
        console: Console,
        unit_settings: UnitSettings | None = None,
        preview_options: PreviewControllerOptions | None = None,
    ):
        self.console = console
        self._pv = None
        self._unit_settings = unit_settings or get_unit_settings()
        self._scene_controller = PreviewSceneController(
            unit_settings=self._unit_settings,
            options=preview_options,
            pyvista_provider=self._ensure_backend,
        )
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
        show_bounds: bool = True,
        show_axes: bool = True,
        control_file: Path | None = None,
        model_path_state: ModelPathState | None = None,
        auto_rebuild_interval_getter: Callable[[], float | None] | None = None,
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
        self._configure_plotter(plotter, show_bounds=show_bounds, show_axes=show_axes)
        if datasets:
            self._apply_scene(
                plotter,
                datasets,
                show_edges=show_edges,
                face_edges=face_edges,
                show_bounds=show_bounds,
                show_axes=show_axes,
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
        current_datasets: list[Mesh | Polyline] = list(datasets) if datasets else []
        build_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="impression-preview-build")
        build_state: dict[str, object] = {
            "future": None,
            "generation": 0,
            "path": None,
            "queued": False,
        }
        render_state = {
            "show_edges": show_edges,
            "face_edges": face_edges,
            "eye_dome": not _pyvista_safe_mode(),
            "background": "dark",
        }
        options_mode = {"value": False}

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
            previous_thread = watcher_thread
            stop_event.set()
            if previous_thread is not None and previous_thread.is_alive():
                previous_thread.join(timeout=1.0)
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

        def _build_scene_datasets() -> list[Mesh | Polyline]:
            return _collect_datasets_from_scene(scene_factory())

        def _submit_scene_build(message: str | None = None) -> None:
            future = build_state["future"]
            if isinstance(future, Future) and not future.done():
                build_state["queued"] = True
                return
            if message is not None:
                self.console.print(message)
            generation = int(build_state["generation"]) + 1
            build_state["generation"] = generation
            build_state["path"] = model_state["path"]
            build_state["queued"] = False
            build_state["future"] = build_executor.submit(_build_scene_datasets)

        def _poll_scene_build() -> None:
            future = build_state["future"]
            if not isinstance(future, Future) or not future.done():
                return
            build_state["future"] = None
            build_path = build_state["path"]
            try:
                datasets = future.result()
            except Exception as exc:
                panel = Panel.fit(_format_exception(exc), title="Preview rebuild failed", style="red")
                self.console.print(panel)
            else:
                if build_path == model_state["path"]:
                    current_datasets.clear()
                    current_datasets.extend(datasets)
                    self._apply_scene(
                        plotter,
                        datasets,
                        show_edges=render_state["show_edges"],
                        face_edges=render_state["face_edges"],
                        show_bounds=show_bounds,
                        show_axes=show_axes,
                        align_camera=False,
                    )
                    plotter.render()
                    self.console.print(f"[green]Reloaded {model_state['path']}[/green]")
            if build_state["queued"]:
                _submit_scene_build(f"[yellow]Reloading {model_state['path']}…[/yellow]")

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

        render_dirty = {"value": False}

        def _request_render_update() -> None:
            render_dirty["value"] = True

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
            _submit_scene_build(f"[yellow]Reloading {current_path}…[/yellow]")

        interval_seconds = max(1.0 / max(target_fps, 1), 0.05)
        callback_cleanup = None
        if watch_files:
            last_auto_rebuild = {"time": time.monotonic()}

            def _maybe_auto_rebuild() -> None:
                if auto_rebuild_interval_getter is None:
                    return
                interval = auto_rebuild_interval_getter()
                if interval is None or interval <= 0:
                    return
                now = time.monotonic()
                if now - last_auto_rebuild["time"] < interval:
                    return
                last_auto_rebuild["time"] = now
                _submit_scene_build()

            def guarded_process_queue() -> None:
                try:
                    process_queue()
                    _maybe_auto_rebuild()
                    _poll_scene_build()
                    if render_dirty["value"]:
                        render_dirty["value"] = False
                        _render_current()
                except Exception as exc:  # pragma: no cover - surfaced via console
                    panel = Panel.fit(_format_exception(exc), title="Reload failed", style="red")
                    self.console.print(panel)

            callback_cleanup = self._install_timer_callback(plotter, guarded_process_queue, interval_seconds)

            def _ensure_renderer_focus() -> bool:
                try:
                    interactor = getattr(plotter, "iren", None)
                    renderer = getattr(plotter, "renderer", None)
                    if interactor is None or renderer is None:
                        return False
                    style = interactor.GetInteractorStyle()
                    if style is not None:
                        style.SetCurrentRenderer(renderer)
                    return True
                except Exception:
                    return False
                return False

            def _handle_key_reload() -> None:
                request_reload_with_message("[yellow]Reload requested (R).[/yellow]")

            plotter.add_key_event("r", _handle_key_reload)

            def _handle_key_reset() -> None:
                if current_datasets:
                    self._reset_camera(plotter, current_datasets)
                    plotter.render()
                    self.console.print("[green]Camera reset.[/green]")

            plotter.add_key_event("v", _handle_key_reset)

            def _render_current() -> None:
                if not current_datasets:
                    return
                if not _ensure_renderer_focus():
                    self.console.print("[yellow]Renderer not ready; skipping render update.[/yellow]")
                    return
                if render_state["background"] == "dark":
                    plotter.set_background("#090c10", top="#1b2333")
                else:
                    plotter.set_background("#e6e9ef")
                if render_state["eye_dome"]:
                    plotter.enable_eye_dome_lighting()
                else:
                    plotter.disable_eye_dome_lighting()
                self._apply_scene(
                    plotter,
                    current_datasets,
                    show_edges=render_state["show_edges"],
                    face_edges=render_state["face_edges"],
                    show_bounds=show_bounds,
                    show_axes=show_axes,
                    align_camera=False,
                )
                plotter.render()

            # Additional keybindings are currently shelved for reliability.

        try:
            plotter.show(title="Impression Preview", auto_close=False)
        finally:
            stop_event.set()
            if watcher_thread is not None and watcher_thread.is_alive():
                watcher_thread.join(timeout=1.0)
            if callback_cleanup is not None:
                callback_cleanup()
            if previous_handler is not None and hasattr(signal, "SIGUSR1"):
                signal.signal(signal.SIGUSR1, previous_handler)
            build_executor.shutdown(wait=False, cancel_futures=True)
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

    def _configure_plotter(self, plotter, *, show_bounds: bool = True, show_axes: bool = True) -> None:
        diagnostic = self._scene_controller.configure_plotter(
            plotter,
            show_bounds=show_bounds,
            show_axes=show_axes,
        )
        if diagnostic.eye_dome_lighting:
            self._install_home_button(plotter)

    def _apply_scene(
        self,
        plotter,
        datasets: Iterable[Mesh | Polyline],
        show_edges: bool,
        face_edges: bool,
        show_bounds: bool = True,
        show_axes: bool = True,
        align_camera: bool = False,
    ) -> None:
        self._scene_controller.apply_scene(
            plotter,
            datasets,
            show_edges=show_edges,
            face_edges=face_edges,
            show_bounds=show_bounds,
            show_axes=show_axes,
            align_camera=align_camera,
        )
        self._home_camera = self._scene_controller.home_camera

    def _add_feature_edges(self, plotter, mesh, color: str, angle: float, index: int) -> None:
        self._scene_controller.add_feature_edges(plotter, mesh, index)

    def _reset_camera(self, plotter, datasets: Iterable[Mesh | Polyline]) -> None:
        diagnostic = self._scene_controller.reset_camera(plotter, datasets)
        self._home_camera = diagnostic.camera_position

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
        self._scene_controller.show_bounds_with_units(plotter)

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
        return self._scene_controller.polyline_to_pyvista(polyline)

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
