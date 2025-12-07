from __future__ import annotations

import importlib.util
import pathlib
from dataclasses import dataclass
from types import ModuleType
import sys
import traceback
from typing import Callable

import typer
from rich.console import Console
from rich.panel import Panel

from impression.preview import PyVistaPreviewer, PreviewBackendError

console = Console()
app = typer.Typer(help="Experiment with parametric models and preview pipelines.")


@dataclass(frozen=True)
class PreviewOptions:
    watch: bool
    target_fps: int


def _log_active_units(previewer: PyVistaPreviewer) -> None:
    scale = previewer.unit_scale_to_mm
    units = previewer.unit_name
    label = previewer.unit_label
    if abs(scale - 1.0) < 1e-9:
        console.print(f"[magenta]Units: {units} ({label}).[/magenta]")
    else:
        console.print(
            f"[magenta]Units: {units} ({label}); 1 {label} = {scale:.4g} mm.[/magenta]"
        )


def _load_module(path: pathlib.Path) -> ModuleType:
    module_name = "impression_user_model"
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"Unable to import model at {path}")

    module = importlib.util.module_from_spec(spec)
    # Register module so features relying on sys.modules (e.g., dataclasses) work.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ModelBuildError(RuntimeError):
    """Raised when a model module cannot provide a usable scene."""


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc))


def _scene_factory_from_module(model_path: pathlib.Path) -> Callable[[], object]:
    def factory() -> object:
        module = _load_module(model_path)
        builder = getattr(module, "build", None)
        if builder is None or not callable(builder):
            raise ModelBuildError(f"{model_path} must define a callable build() function.")
        return builder()

    return factory


def _next_available_path(path: pathlib.Path) -> pathlib.Path:
    """Return a non-conflicting path by appending ' (n)' before the suffix."""

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


@app.command()
def preview(
    model: pathlib.Path = typer.Argument(..., help="Path to a Python module that defines a model scene."),
    watch: bool = typer.Option(True, help="Watch the model file for changes and hot-reload."),
    target_fps: int = typer.Option(60, min=1, max=240, help="Preview framerate budget."),
    screenshot: pathlib.Path | None = typer.Option(
        None, "--screenshot", help="Optional path to save a screenshot of the preview."
    ),
    show_edges: bool = typer.Option(False, "--show-edges/--hide-edges", help="Toggle triangle edge rendering."),
    face_edges: bool = typer.Option(
        False,
        "--face-edges/--no-face-edges",
        help="Overlay detected face edges (feature edges) for hard-outline visuals.",
    ),
) -> None:
    """
    Load a model module, build PyVista datasets, and open an interactive preview window.
    """

    if not model.exists():
        raise typer.BadParameter(f"Model path {model} does not exist.")

    opts = PreviewOptions(watch=watch, target_fps=target_fps)

    scene_factory = _scene_factory_from_module(model)
    try:
        initial_scene = scene_factory()
    except Exception as exc:
        if opts.watch:
            panel = Panel.fit(_format_exception(exc), title="Initial build failed — watching for changes", style="red")
            console.print(panel)
            initial_scene = None
        else:
            raise typer.BadParameter(f"Model execution failed: {exc}") from exc

    console.rule("Impression Preview")
    console.print(f"Using model [green]{model}[/green]")
    if opts.watch:
        console.print("[cyan]Watching for changes — save to hot reload, close the window to stop.[/cyan]")

    previewer = PyVistaPreviewer(console=console)
    _log_active_units(previewer)
    try:
        previewer.show(
            scene_factory=scene_factory,
            initial_scene=initial_scene,
            model_path=model,
            watch_files=opts.watch,
            target_fps=opts.target_fps,
            screenshot_path=screenshot,
            show_edges=show_edges,
            face_edges=face_edges,
        )
    except PreviewBackendError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def export(
    model: pathlib.Path = typer.Argument(..., help="Model module to export."),
    output: pathlib.Path = typer.Option(
        pathlib.Path("model.stl"),
        "--output",
        "-o",
        help="Path to the STL file that will be produced.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow replacing an existing STL."),
    ascii: bool = typer.Option(False, "--ascii", help="Write ASCII STL instead of binary."),
) -> None:
    """
    Convert the provided model into a merged mesh and save it as an STL file.
    """

    if not model.exists():
        raise typer.BadParameter(f"Model path {model} does not exist.")

    final_output = output
    if output.exists():
        if not overwrite:
            final_output = _next_available_path(output)
            if final_output != output:
                console.print(f"[yellow]Output {output} exists; writing to {final_output} instead.[/yellow]")

    try:
        _, initial_scene = _scene_factory_from_module(model)
    except ModelBuildError as exc:
        raise typer.BadParameter(str(exc)) from exc

    previewer = PyVistaPreviewer(console=console)
    _log_active_units(previewer)
    try:
        datasets = previewer.collect_datasets(initial_scene)
        merged = previewer.combine_to_polydata(datasets)
    except PreviewBackendError as exc:
        raise typer.BadParameter(str(exc)) from exc

    final_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        merged.save(str(final_output), binary=not ascii)
    except Exception as exc:  # pragma: no cover - PyVista I/O failure
        raise typer.BadParameter(f"Failed to export STL: {exc}") from exc

    mode = "ASCII" if ascii else "binary"
    units_note = f"Units: {previewer.unit_name} ({previewer.unit_label})."
    console.print(
        Panel(
            f"Wrote {mode} STL to [green]{final_output}[/green]. {units_note}",
            title="Export complete",
            border_style="green",
        )
    )
