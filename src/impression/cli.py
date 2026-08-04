from __future__ import annotations

import importlib.util
import importlib
import pathlib
from dataclasses import dataclass
from types import ModuleType
import sys
import traceback
import inspect
import time
import threading
from typing import Callable, Iterable

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
import stat

from impression.io import write_stl_atomically
from impression.mesh import Mesh, MeshAnalysis, analyze_mesh
from impression import __version__
from impression.preview import PyVistaPreviewer, PreviewBackendError

console = Console()
app = typer.Typer(help="Experiment with parametric models and preview pipelines.")
WatchPathsCallback = Callable[[tuple[pathlib.Path, ...]], None]
_USER_MODEL_OWNED_MODULE_PATHS: dict[str, pathlib.Path] = {}


@dataclass(frozen=True)
class PreviewOptions:
    watch: bool
    target_fps: int


def _validate_manufacturing_mesh(mesh: Mesh) -> MeshAnalysis:
    """Require the default STL contract before any output-path mutation."""

    analysis = analyze_mesh(mesh)
    issues: list[str] = []
    if analysis.n_vertices == 0:
        issues.append("empty mesh (0 vertices)")
    if analysis.n_faces == 0:
        issues.append("empty mesh (0 faces)")
    issues.extend(analysis.issues())
    if issues:
        raise PreviewBackendError(
            "Manufacturing STL integrity check failed: " + "; ".join(issues) + "."
        )
    return analysis


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


def _drop_owned_user_model_modules() -> None:
    for name, owned_path in tuple(_USER_MODEL_OWNED_MODULE_PATHS.items()):
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None) if module is not None else None
        if module_file is not None:
            try:
                current_path = pathlib.Path(module_file).resolve()
            except OSError:
                current_path = None
            if current_path == owned_path:
                sys.modules.pop(name, None)
                cached_path = getattr(module, "__cached__", None)
                if cached_path:
                    try:
                        pathlib.Path(cached_path).unlink(missing_ok=True)
                    except OSError:
                        pass
    _USER_MODEL_OWNED_MODULE_PATHS.clear()
    importlib.invalidate_caches()


def _record_user_model_modules(
    model_path: pathlib.Path,
    module_names: Iterable[str],
) -> None:
    _USER_MODEL_OWNED_MODULE_PATHS.update(
        _tracked_preview_module_paths(model_path, module_names)
    )


def _load_module(path: pathlib.Path) -> ModuleType:
    module_name = "impression_user_model"
    _drop_owned_user_model_modules()
    before_names = set(sys.modules)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"Unable to import model at {path}")

    module = importlib.util.module_from_spec(spec)
    # Register module so features relying on sys.modules (e.g., dataclasses) work.
    sys.modules[module_name] = module
    model_dir = str(path.resolve().parent)
    added_model_dir = model_dir not in sys.path
    if added_model_dir:
        sys.path.insert(0, model_dir)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        _record_user_model_modules(path, set(sys.modules) - before_names | {module_name})
        _drop_owned_user_model_modules()
        raise
    finally:
        if added_model_dir:
            try:
                sys.path.remove(model_dir)
            except ValueError:
                pass
    _record_user_model_modules(path, set(sys.modules) - before_names | {module_name})
    return module


def _path_is_relative_to(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _tracked_preview_module_paths(
    model_path: pathlib.Path,
    module_names: Iterable[str],
) -> dict[str, pathlib.Path]:
    """Return local Python modules that should hot-reload with the preview model."""

    resolved_model = model_path.resolve()
    package_root = pathlib.Path(__file__).resolve().parent
    runtime_roots = {pathlib.Path(sys.prefix).resolve(), pathlib.Path(sys.base_prefix).resolve()}
    tracked: dict[str, pathlib.Path] = {"impression_user_model": resolved_model}

    for name in module_names:
        module = sys.modules.get(name)
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            path = pathlib.Path(module_file).resolve()
        except OSError:
            continue
        if path.suffix not in {".py", ".pyw"}:
            continue
        if path == resolved_model:
            tracked[name] = path
            continue
        if _path_is_relative_to(path, package_root):
            continue
        if any(_path_is_relative_to(path, root) for root in runtime_roots):
            continue
        if "site-packages" in path.parts or "dist-packages" in path.parts:
            continue
        tracked[name] = path
    return tracked


def _watch_mtimes(paths: Iterable[pathlib.Path]) -> dict[pathlib.Path, int | None]:
    mtimes: dict[pathlib.Path, int | None] = {}
    for path in paths:
        try:
            mtimes[path.resolve()] = path.stat().st_mtime_ns
        except OSError:
            mtimes[path.resolve()] = None
    return mtimes


class ModelBuildError(RuntimeError):
    """Raised when a model module cannot provide a usable scene."""


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc))


@dataclass(frozen=True)
class _ValidatedArchiveMember:
    info: zipfile.ZipInfo
    parts: tuple[str, ...]
    is_directory: bool


def _validate_archive_member(member: zipfile.ZipInfo) -> _ValidatedArchiveMember:
    raw_name = member.orig_filename
    if "\x00" in raw_name:
        raise typer.BadParameter(f"Unsafe docs archive member: {raw_name!r} (NUL byte).")

    normalized = raw_name.replace("\\", "/")
    if (
        normalized.startswith("/")
        or normalized.startswith("//")
        or re.match(r"^[A-Za-z]:", normalized)
    ):
        raise typer.BadParameter(f"Unsafe docs archive member: {raw_name!r} (absolute path).")

    parts = tuple(part for part in pathlib.PurePosixPath(normalized).parts if part not in {"", "."})
    if not parts or ".." in parts:
        raise typer.BadParameter(f"Unsafe docs archive member: {raw_name!r} (invalid path).")

    unix_mode = member.external_attr >> 16
    file_type = stat.S_IFMT(unix_mode)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise typer.BadParameter(f"Unsafe docs archive member: {raw_name!r} (link-like type).")

    is_directory = member.is_dir() or file_type == stat.S_IFDIR
    return _ValidatedArchiveMember(member, parts, is_directory)


def _replace_directory_atomically(staged: pathlib.Path, destination: pathlib.Path) -> None:
    backup: pathlib.Path | None = None
    if destination.exists():
        backup = pathlib.Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent)
        )
        backup.rmdir()
        os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except BaseException:
        if backup is not None and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def _extract_docs_archive(data: bytes, destination: pathlib.Path, clean: bool) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = tuple(_validate_archive_member(member) for member in archive.infolist())
        if not members:
            raise typer.BadParameter("Downloaded archive is empty.")

        prefixes: list[tuple[str, ...]] = []
        for member in members:
            for index, part in enumerate(member.parts):
                if part == "docs" and (index < len(member.parts) - 1 or member.is_directory):
                    prefixes.append(member.parts[: index + 1])
                    break
        if not prefixes:
            raise typer.BadParameter("Docs folder not found in the downloaded archive.")
        docs_prefix = min(prefixes, key=lambda prefix: (len(prefix), prefix))

        destination = destination.expanduser()
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            raise typer.BadParameter(f"Docs destination is not a directory: {destination}")
        destination = destination.resolve(strict=False)

        selected: list[tuple[_ValidatedArchiveMember, pathlib.Path]] = []
        for member in members:
            if member.parts[: len(docs_prefix)] != docs_prefix:
                continue
            relative_parts = member.parts[len(docs_prefix):]
            if not relative_parts:
                continue
            target = destination.joinpath(*relative_parts).resolve(strict=False)
            if not _path_is_relative_to(target, destination):
                raise typer.BadParameter(
                    f"Unsafe docs archive member: {member.info.orig_filename!r} (outside destination)."
                )
            selected.append((member, target))

        if not any(not member.is_directory for member, _ in selected):
            raise typer.BadParameter("Docs folder not found in the downloaded archive.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = pathlib.Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.staged-", dir=destination.parent)
        )
        try:
            if destination.exists() and not clean:
                shutil.copytree(destination, staged, dirs_exist_ok=True, symlinks=True)

            for member, intended_target in selected:
                relative = intended_target.relative_to(destination)
                staged_target = staged / relative
                if member.is_directory:
                    staged_target.mkdir(parents=True, exist_ok=True)
                    continue
                staged_target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member.info) as source, staged_target.open("wb") as target_file:
                    shutil.copyfileobj(source, target_file)

            _replace_directory_atomically(staged, destination)
        except BaseException:
            if staged.exists():
                shutil.rmtree(staged)
            raise
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
    on_watch_paths_changed: WatchPathsCallback | None = None,
    cache_module: bool = False,
) -> Callable[[], object]:
    cached_module: ModuleType | None = None
    cached_mtimes: dict[pathlib.Path, int | None] = {}
    cached_module_names: set[str] = set()
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

    def _notify_watch_paths_changed(paths: Iterable[pathlib.Path]) -> None:
        if on_watch_paths_changed is None:
            return
        on_watch_paths_changed(tuple(sorted({path.resolve() for path in paths})))

    def _drop_cached_modules() -> None:
        for name in sorted(cached_module_names):
            module = sys.modules.pop(name, None)
            cached_path = getattr(module, "__cached__", None) if module is not None else None
            if cached_path:
                try:
                    pathlib.Path(cached_path).unlink(missing_ok=True)
                except OSError:
                    pass
        importlib.invalidate_caches()

    def _finish_module_load(module: ModuleType, loaded_module_names: Iterable[str]) -> ModuleType:
        nonlocal start_time, last_build_time, previous_scene, cached_module_names, cached_mtimes
        if on_module_loaded is not None:
            on_module_loaded(module)
        builder = getattr(module, "build", None)
        if builder is None or not callable(builder):
            raise ModelBuildError(f"{model_path} must define a callable build() function.")
        _refresh_builder_metadata(builder)
        tracked_paths_by_module = _tracked_preview_module_paths(
            model_path,
            set(loaded_module_names) | {module.__name__},
        )
        cached_module_names = set(tracked_paths_by_module)
        cached_mtimes = _watch_mtimes(tracked_paths_by_module.values())
        _notify_watch_paths_changed(cached_mtimes)
        start_time = time.monotonic()
        last_build_time = None
        previous_scene = None
        return module

    def _load_cached_or_fresh_module() -> ModuleType:
        nonlocal cached_module
        if not cache_module:
            before_names = set(sys.modules)
            module = _load_module(model_path)
            return _finish_module_load(
                module,
                set(sys.modules) - before_names | set(_USER_MODEL_OWNED_MODULE_PATHS),
            )

        current_mtimes = _watch_mtimes(cached_mtimes or [model_path])
        needs_reload = cached_module is None or current_mtimes != cached_mtimes
        if needs_reload:
            _drop_cached_modules()
            before_names = set(sys.modules)
            cached_module = _load_module(model_path)
            _finish_module_load(
                cached_module,
                set(sys.modules) - before_names | set(_USER_MODEL_OWNED_MODULE_PATHS),
            )
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


def _scene_factory_from_impress(model_path: pathlib.Path) -> Callable[[], object]:
    def factory() -> object:
        from impression.io import load_impress
        from impression.modeling import preview_tessellation_request, tessellate_surface_body

        loaded = load_impress(model_path)
        request = preview_tessellation_request(require_watertight=False)
        return tuple(tessellate_surface_body(body, request).mesh for body in loaded.bodies)

    return factory


def _scene_factory_from_path(
    model_path: pathlib.Path,
    *,
    on_module_loaded: Callable[[ModuleType], None] | None = None,
    on_watch_paths_changed: WatchPathsCallback | None = None,
    cache_module: bool = False,
) -> Callable[[], object]:
    if model_path.suffix.lower() == ".impress":
        if on_watch_paths_changed is not None:
            on_watch_paths_changed((model_path.resolve(),))
        return _scene_factory_from_impress(model_path)
    return _scene_factory_from_module(
        model_path,
        on_module_loaded=on_module_loaded,
        on_watch_paths_changed=on_watch_paths_changed,
        cache_module=cache_module,
    )


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
    model: pathlib.Path = typer.Argument(..., help="Path to a Python module or .impress document to preview."),
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
        None,
        "--screenshot",
        help="Render once off-screen, save a PNG, and exit without redirecting a running preview.",
    ),
    show_edges: bool = typer.Option(False, "--show-edges/--hide-edges", help="Toggle triangle edge rendering."),
    face_edges: bool = typer.Option(
        False,
        "--face-edges/--no-face-edges",
        help="Overlay detected face edges (feature edges) for hard-outline visuals.",
    ),
) -> None:
    """
    Load a Python model module or .impress document, then open an interactive
    preview or write a one-shot PNG with --screenshot.
    """

    if not model.exists():
        raise typer.BadParameter(f"Model path {model} does not exist.")

    opts = PreviewOptions(watch=watch, target_fps=target_fps)
    screenshot_mode = screenshot is not None
    effective_watch = opts.watch and not screenshot_mode

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
    watched_paths_lock = threading.Lock()
    watched_paths: set[pathlib.Path] = {model.resolve()}

    def _on_watch_paths_changed(paths: tuple[pathlib.Path, ...]) -> None:
        nonlocal watched_paths
        with watched_paths_lock:
            watched_paths = {path.resolve() for path in paths}

    def _get_watch_paths() -> tuple[pathlib.Path, ...]:
        with watched_paths_lock:
            return tuple(watched_paths)

    def _get_scene_factory(path: pathlib.Path) -> Callable[[], object]:
        current_path = scene_factory_cache["path"]
        factory = scene_factory_cache["factory"]
        resolved = path.resolve()
        if current_path is None or factory is None or resolved != current_path:
            factory = _scene_factory_from_path(
                path,
                on_module_loaded=_on_module_loaded,
                on_watch_paths_changed=_on_watch_paths_changed,
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
        if effective_watch:
            panel = Panel.fit(_format_exception(exc), title="Initial build failed — watching for changes", style="red")
            console.print(panel)
            initial_scene = None
        else:
            raise typer.BadParameter(f"Model execution failed: {exc}") from exc

    console.rule("Impression Preview")
    console.print(f"Using model [green]{model}[/green]")
    control_path: pathlib.Path | None = None
    if effective_watch:
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
            watch_files=effective_watch,
            target_fps=opts.target_fps,
            screenshot_path=screenshot,
            show_edges=show_edges,
            face_edges=face_edges,
            show_bounds=preview_chrome_state["show_bounds"],
            show_axes=preview_chrome_state["show_axes"],
            control_file=control_path,
            watch_paths_getter=_get_watch_paths,
            auto_rebuild_interval_getter=lambda: auto_rebuild_state["interval"],
        )
        if screenshot is not None:
            console.print(f"[green]Saved preview PNG to {screenshot.resolve()}[/green]")
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
        from impression.modeling import export_tessellation_request

        datasets = previewer.collect_datasets(
            initial_scene,
            tessellation_request=export_tessellation_request(),
        )
        merged = previewer.combine_to_mesh(datasets)
        _validate_manufacturing_mesh(merged)
    except PreviewBackendError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        write_stl_atomically(merged, final_output, ascii=ascii)
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
