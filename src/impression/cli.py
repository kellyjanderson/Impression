from __future__ import annotations

import importlib.util
import pathlib
from dataclasses import dataclass
from types import ModuleType
import sys
from typing import Callable, Tuple

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


def _scene_factory_from_module(model_path: pathlib.Path) -> Tuple[Callable[[], object], object]:
    def factory() -> object:
        module = _load_module(model_path)
        builder = getattr(module, "build", None)
        if builder is None or not callable(builder):
            raise ModelBuildError(f"{model_path} must define a callable build() function.")
        return builder()

    initial_scene = factory()
    return factory, initial_scene


@app.command()
def preview(
    model: pathlib.Path = typer.Argument(..., help="Path to a Python module that defines a model scene."),
    watch: bool = typer.Option(True, help="Watch the model file for changes and hot-reload."),
    target_fps: int = typer.Option(60, min=1, max=240, help="Preview framerate budget."),
) -> None:
    """
    Load a model module, build PyVista datasets, and open an interactive preview window.
    """

    if not model.exists():
        raise typer.BadParameter(f"Model path {model} does not exist.")

    opts = PreviewOptions(watch=watch, target_fps=target_fps)

    try:
        scene_factory, initial_scene = _scene_factory_from_module(model)
    except ModelBuildError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to CLI users
        raise typer.BadParameter(f"Model execution failed: {exc}") from exc

    console.rule("Impression Preview")
    console.print(f"Using model [green]{model}[/green]")
    if opts.watch:
        console.print("[cyan]Watching for changes — save to hot reload, close the window to stop.[/cyan]")

    previewer = PyVistaPreviewer(console=console)
    try:
        previewer.show(
            scene_factory=scene_factory,
            initial_scene=initial_scene,
            model_path=model,
            watch_files=opts.watch,
            target_fps=opts.target_fps,
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
) -> None:
    """
    Placeholder command that will eventually traverse the canonical geometry graph
    and emit STL/STEP/AMF outputs.
    """

    if not model.exists():
        raise typer.BadParameter(f"Model path {model} does not exist.")

    console.print(
        Panel(
            f"Export pipeline is not implemented yet.\nWould have read [green]{model}[/green] "
            f"and produced [magenta]{output}[/magenta].",
            title="Export stub",
        )
    )
