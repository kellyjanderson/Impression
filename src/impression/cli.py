from __future__ import annotations

import importlib.util
import pathlib
import time
from dataclasses import dataclass
from types import ModuleType

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()
app = typer.Typer(help="Experiment with parametric models and preview pipelines.")


@dataclass(frozen=True)
class PreviewOptions:
    watch: bool
    target_fps: int


def _load_module(path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("impression_user_model", path)
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"Unable to import model at {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render_stub(module: ModuleType) -> None:
    header = f"Preview stub for `{module.__name__}`"
    body = (
        "[bold yellow]Preview pipeline is not wired up yet.[/bold yellow]\n\n"
        "Expected next steps:\n"
        " • integrate pygfx/pyvista backend for orbit controls\n"
        " • wire file watching so edits auto-refresh the scene\n"
        " • surface camera + transform controls via CLI flags\n"
    )
    console.print(Panel(body, title=header))


@app.command()
def preview(
    model: pathlib.Path = typer.Argument(..., help="Path to a Python module that defines a model scene."),
    watch: bool = typer.Option(True, help="Watch the model file for changes and hot-reload."),
    target_fps: int = typer.Option(60, min=1, max=240, help="Preview framerate budget."),
) -> None:
    """
    Load and preview a model module. For now the renderer is stubbed, but the command
    exercises the dynamic import path and upcoming watchflow pieces.
    """

    if not model.exists():
        raise typer.BadParameter(f"Model path {model} does not exist.")

    opts = PreviewOptions(watch=watch, target_fps=target_fps)
    module = _load_module(model)

    console.rule("Impression Preview")
    console.print(f"Loading model from [green]{model}[/green]")
    _render_stub(module)

    if not opts.watch:
        return

    console.print("\n[cyan]Watching for changes (Ctrl+C to stop)...[/cyan]")
    try:
        last_mtime = model.stat().st_mtime
        while True:
            time.sleep(max(1 / opts.target_fps, 0.01))
            current_mtime = model.stat().st_mtime
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                module = _load_module(model)
                console.print(f"\n[green]Reloaded {model}[/green]")
                _render_stub(module)
    except KeyboardInterrupt:
        console.print("\n[red]Stopped watching.[/red]")


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
