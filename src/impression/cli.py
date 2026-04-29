from __future__ import annotations

import importlib.util
import pathlib
from dataclasses import dataclass
from types import ModuleType
import sys
import traceback
import inspect
import time
from typing import Callable

import typer
from rich.console import Console
from rich.panel import Panel

import io
import shutil
import tempfile
import urllib.request
import urllib.error
import zipfile
import os
import re

from impression.io import write_stl
from impression import __version__
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
    for name in list(sys.modules.keys()):
        if name == "impression.modeling" or name.startswith("impression.modeling."):
            del sys.modules[name]

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


def _extract_docs_archive(data: bytes, destination: pathlib.Path, clean: bool) -> None:
    if clean and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if not names:
            raise typer.BadParameter("Downloaded archive is empty.")

        prefixes: list[str] = []
        for name in names:
            if name.startswith("docs/"):
                prefixes.append("docs/")
            idx = name.find("/docs/")
            if idx != -1:
                prefixes.append(name[: idx + len("/docs/")])
        if not prefixes:
            raise typer.BadParameter("Docs folder not found in the downloaded archive.")
        docs_prefix = min(prefixes, key=len)

        extracted = False
        for member in archive.infolist():
            if not member.filename.startswith(docs_prefix):
                continue
            rel_path = pathlib.Path(member.filename[len(docs_prefix):])
            if not rel_path.parts:
                continue
            target_path = destination / rel_path
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted = True

    if not extracted:
        raise typer.BadParameter("Docs folder not found in the downloaded archive.")
    console.print(f"[green]Docs saved to {destination}[/green]")


def _download_docs_archive(
    repo_url: str,
    ref: str,
    destination: pathlib.Path,
    clean: bool,
) -> None:
    repo_url = repo_url.rstrip("/")
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]
    tag_url = f"{repo_url}/archive/refs/tags/{ref}.zip"
    head_url = f"{repo_url}/archive/refs/heads/{ref}.zip"

    console.print(f"[cyan]Downloading docs from {tag_url}...[/cyan]")
    data: bytes | None = None
    try:
        with urllib.request.urlopen(tag_url) as response:
            data = response.read()
    except urllib.error.HTTPError:
        try:
            console.print(f"[cyan]Tag archive missing; trying branch archive {head_url}...[/cyan]")
            with urllib.request.urlopen(head_url) as response:
                data = response.read()
        except urllib.error.HTTPError as exc:
            raise typer.BadParameter(f"Could not download docs archive for ref '{ref}'.") from exc
    if data is None:
        raise typer.BadParameter(f"Could not download docs archive for ref '{ref}'.")
    _extract_docs_archive(data, destination, clean)


def _download_docs_release_asset(
    repo_url: str,
    ref: str,
    destination: pathlib.Path,
    clean: bool,
) -> None:
    repo_url = repo_url.rstrip("/")
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]
    asset_url = f"{repo_url}/releases/download/{ref}/impression-docs-{ref}.zip"
    console.print(f"[cyan]Downloading docs asset from {asset_url}...[/cyan]")
    try:
        with urllib.request.urlopen(asset_url) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        raise typer.BadParameter(f"Docs asset impression-docs-{ref}.zip not found for release {ref}.") from exc
    _extract_docs_archive(data, destination, clean)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the Impression version and exit.",
    ),
    get_docs: bool = typer.Option(
        False,
        "--get-docs",
        "--getDocs",
        help="Download documentation from GitHub and exit.",
    ),
    docs_dest: pathlib.Path | None = typer.Option(
        None,
        "--docs-dest",
        help="Destination folder for downloaded docs (default: ./impression-docs).",
    ),
    docs_repo: str = typer.Option(
        "https://github.com/kellyjanderson/Impression",
        "--docs-repo",
        help="GitHub repo URL for docs download.",
    ),
    docs_ref: str | None = typer.Option(
        None,
        "--docs-ref",
        help="Git ref to fetch docs from (default: installed release tag).",
    ),
    docs_clean: bool = typer.Option(
        False,
        "--docs-clean",
        help="Delete the destination folder before downloading.",
    ),
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()

    if get_docs:
        destination = docs_dest or pathlib.Path.cwd() / "impression-docs"
        resolved_ref = docs_ref or f"v{__version__}"
        try:
            _download_docs_release_asset(docs_repo, resolved_ref, destination, docs_clean)
        except typer.BadParameter:
            _download_docs_archive(docs_repo, resolved_ref, destination, docs_clean)
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        return


def _scene_factory_from_module(
    model_path: pathlib.Path,
    *,
    on_module_loaded: Callable[[ModuleType], None] | None = None,
    cache_module: bool = False,
) -> Callable[[], object]:
    cached_module: ModuleType | None = None
    cached_mtime_ns: int | None = None
    builder_signature: inspect.Signature | None = None
    accepts_kwargs = False
    accepted_kw_names: set[str] = set()
    start_time = time.monotonic()
    last_build_time: float | None = None
    previous_scene: object | None = None

    def _refresh_builder_metadata(builder: Callable[..., object]) -> None:
        nonlocal builder_signature, accepts_kwargs, accepted_kw_names
        builder_signature = inspect.signature(builder)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in builder_signature.parameters.values()
        )
        accepted_kw_names = {
            name
            for name, param in builder_signature.parameters.items()
            if param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }

    def _load_cached_or_fresh_module() -> ModuleType:
        nonlocal cached_module, cached_mtime_ns, start_time, last_build_time, previous_scene
        if not cache_module:
            module = _load_module(model_path)
            if on_module_loaded is not None:
                on_module_loaded(module)
            builder = getattr(module, "build", None)
            if builder is None or not callable(builder):
                raise ModelBuildError(f"{model_path} must define a callable build() function.")
            _refresh_builder_metadata(builder)
            start_time = time.monotonic()
            last_build_time = None
            previous_scene = None
            return module

        try:
            mtime_ns = model_path.stat().st_mtime_ns
        except OSError:
            mtime_ns = None
        needs_reload = cached_module is None or mtime_ns != cached_mtime_ns
        if needs_reload:
            cached_module = _load_module(model_path)
            cached_mtime_ns = mtime_ns
            if on_module_loaded is not None:
                on_module_loaded(cached_module)
            builder = getattr(cached_module, "build", None)
            if builder is None or not callable(builder):
                raise ModelBuildError(f"{model_path} must define a callable build() function.")
            _refresh_builder_metadata(builder)
            start_time = time.monotonic()
            last_build_time = None
            previous_scene = None
        return cached_module

    def factory() -> object:
        nonlocal last_build_time, previous_scene
        module = _load_cached_or_fresh_module()
        builder = getattr(module, "build", None)
        if builder is None or not callable(builder):
            raise ModelBuildError(f"{model_path} must define a callable build() function.")
        now = time.monotonic()
        kwargs: dict[str, object] = {}
        if accepts_kwargs or "elapsed_seconds" in accepted_kw_names:
            kwargs["elapsed_seconds"] = now - start_time
        if (accepts_kwargs or "dt_seconds" in accepted_kw_names) and last_build_time is not None:
            kwargs["dt_seconds"] = now - last_build_time
        if (accepts_kwargs or "previous_scene" in accepted_kw_names) and previous_scene is not None:
            kwargs["previous_scene"] = previous_scene
        scene = builder(**kwargs) if kwargs else builder()
        last_build_time = now
        previous_scene = scene
        return scene

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
    control_file: pathlib.Path | None = typer.Option(
        None,
        "--control-file",
        help="Optional control file for switching preview targets (default: ./.impression-preview).",
    ),
    force_window: bool = typer.Option(
        False,
        "--force-window",
        help="Force a new preview window even if a live control file exists.",
    ),
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

    model_state = {"path": model}
    auto_rebuild_state: dict[str, float | None] = {"interval": None}
    preview_chrome_state: dict[str, bool] = {"show_bounds": True, "show_axes": True}

    def _module_bool(module: ModuleType, name: str, default: bool) -> bool:
        value = getattr(module, name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
        return default

    def _on_module_loaded(module: ModuleType) -> None:
        interval = getattr(module, "ANIMATE_INTERVAL_SECONDS", None)
        if interval is None:
            auto_rebuild_state["interval"] = None
            return
        try:
            interval_val = float(interval)
        except (TypeError, ValueError):
            auto_rebuild_state["interval"] = None
        else:
            auto_rebuild_state["interval"] = interval_val if interval_val > 0 else None
        preview_chrome_state["show_bounds"] = _module_bool(module, "PREVIEW_SHOW_BOUNDS", True)
        preview_chrome_state["show_axes"] = _module_bool(module, "PREVIEW_SHOW_AXES", True)

    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _read_control_header(path: pathlib.Path) -> int | None:
        try:
            text = path.read_text().splitlines()
        except OSError:
            return None
        if not text:
            return None
        if not text[0].startswith("# impression-preview pid="):
            return None
        match = re.search(r"pid=(\d+)", text[0])
        if not match:
            return None
        return int(match.group(1))

    def _write_control_file(path: pathlib.Path) -> pathlib.Path | None:
        header = f"# impression-preview pid={os.getpid()}\n"
        try:
            path.write_text(header + str(model) + "\n")
        except OSError:
            return None
        return path

    def _ensure_control_file() -> pathlib.Path | None:
        if not opts.watch:
            return None
        path = control_file or (pathlib.Path.cwd() / ".impression-preview")
        if path.exists():
            existing_pid = _read_control_header(path)
            if existing_pid is not None and _pid_alive(existing_pid):
                if not force_window:
                    path.write_text(f"# impression-preview pid={existing_pid}\n{model}\n")
                    console.print(f"[cyan]Sent {model} to running preview (pid {existing_pid}).[/cyan]")
                    raise typer.Exit()
                return _write_control_file(path)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return _write_control_file(path)

    scene_factory_cache: dict[str, pathlib.Path | Callable[[], object] | None] = {
        "path": None,
        "factory": None,
    }

    def _get_scene_factory(path: pathlib.Path) -> Callable[[], object]:
        current_path = scene_factory_cache["path"]
        factory = scene_factory_cache["factory"]
        resolved = path.resolve()
        if current_path is None or factory is None or resolved != current_path:
            factory = _scene_factory_from_module(
                path,
                on_module_loaded=_on_module_loaded,
                cache_module=True,
            )
            scene_factory_cache["path"] = resolved
            scene_factory_cache["factory"] = factory
        return factory  # type: ignore[return-value]

    def scene_factory() -> object:
        return _get_scene_factory(model_state["path"])()
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
    control_path: pathlib.Path | None = None
    if opts.watch:
        console.print("[cyan]Watching for changes — save to hot reload, close the window to stop.[/cyan]")
        control_path = _ensure_control_file()
        if control_path is not None:
            console.print(
                f"[cyan]Switch file: {control_path} (write a new path to auto-reload; SIGUSR1 optional).[/cyan]"
            )
    interval = auto_rebuild_state["interval"]
    if interval is not None:
        console.print(f"[cyan]Animation timer active: rebuild every {interval:.2f}s.[/cyan]")

    previewer = PyVistaPreviewer(console=console)
    _log_active_units(previewer)
    try:
        previewer.show(
            scene_factory=scene_factory,
            initial_scene=initial_scene,
            model_path=model,
            model_path_state=model_state,
            watch_files=opts.watch,
            target_fps=opts.target_fps,
            screenshot_path=screenshot,
            show_edges=show_edges,
            face_edges=face_edges,
            show_bounds=preview_chrome_state["show_bounds"],
            show_axes=preview_chrome_state["show_axes"],
            control_file=control_path,
            auto_rebuild_interval_getter=lambda: auto_rebuild_state["interval"],
        )
    except PreviewBackendError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def export(
    model: pathlib.Path = typer.Argument(..., help="Model module to export."),
    output: pathlib.Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to the STL file that will be produced (defaults to model filename with .stl).",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow replacing an existing STL."),
    ascii: bool = typer.Option(False, "--ascii", help="Write ASCII STL instead of binary."),
) -> None:
    """
    Convert the provided model into a merged mesh and save it as an STL file.
    """

    if not model.exists():
        raise typer.BadParameter(f"Model path {model} does not exist.")

    requested_output = output if output is not None else model.with_suffix(".stl")
    final_output = requested_output
    if requested_output.exists():
        if not overwrite:
            final_output = _next_available_path(requested_output)
            if final_output != requested_output:
                console.print(
                    f"[yellow]Output {requested_output} exists; writing to {final_output} instead.[/yellow]"
                )

    try:
        scene_factory = _scene_factory_from_module(model)
        initial_scene = scene_factory()
    except ModelBuildError as exc:
        raise typer.BadParameter(str(exc)) from exc

    previewer = PyVistaPreviewer(console=console)
    _log_active_units(previewer)
    try:
        datasets = previewer.collect_datasets(initial_scene)
        merged = previewer.combine_to_mesh(datasets)
    except PreviewBackendError as exc:
        raise typer.BadParameter(str(exc)) from exc

    final_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_stl(merged, final_output, ascii=ascii)
    except Exception as exc:  # pragma: no cover - STL I/O failure
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
