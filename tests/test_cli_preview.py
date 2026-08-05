from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import click
import pytest
from typer.testing import CliRunner

import impression.cli as cli
from impression.cli import _scene_factory_from_path
from impression.io import save_impress
from impression.mesh import Mesh
from impression.modeling import make_box
from impression.preview import PreviewBackendError


def test_preview_scene_factory_loads_impress_document(tmp_path: Path) -> None:
    path = tmp_path / "box.impress"
    save_impress([make_box(size=(1, 1, 1))], path)

    scene = _scene_factory_from_path(path)()

    assert isinstance(scene, tuple)
    assert len(scene) == 1
    assert isinstance(scene[0], Mesh)
    assert scene[0].n_faces > 0


def test_preview_scene_factory_keeps_python_module_hook(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text(
        "import numpy as np\n"
        "from impression.mesh import Mesh\n\n"
        "def build():\n"
        "    return Mesh(\n"
        "        vertices=np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float),\n"
        "        faces=np.asarray(((0, 1, 2),), dtype=int),\n"
        "    )\n"
    )
    loaded = []

    scene = _scene_factory_from_path(source, on_module_loaded=loaded.append)()

    assert len(loaded) == 1
    assert isinstance(scene, Mesh)


def test_preview_scene_factory_tracks_transitive_local_includes(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    include_a = tmp_path / "include_a.py"
    include_b = tmp_path / "include_b.py"
    include_b.write_text(
        "import numpy as np\n"
        "from impression.mesh import Mesh\n\n"
        "def make_mesh():\n"
        "    return Mesh(\n"
        "        vertices=np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float),\n"
        "        faces=np.asarray(((0, 1, 2),), dtype=int),\n"
        "    )\n"
    )
    include_a.write_text("from include_b import make_mesh\n")
    source.write_text(
        "from include_a import make_mesh\n\n"
        "def build():\n"
        "    return make_mesh()\n"
    )
    watched = []

    scene = _scene_factory_from_path(source, on_watch_paths_changed=watched.append)()

    assert isinstance(scene, Mesh)
    assert watched
    assert {source.resolve(), include_a.resolve(), include_b.resolve()} <= set(watched[-1])


def test_preview_scene_factory_reloads_cached_transitive_includes(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    include_a = tmp_path / "reload_include_a.py"
    include_b = tmp_path / "reload_include_b.py"
    include_a.write_text("from reload_include_b import vertex_x\n")
    include_b.write_text("vertex_x = 1.0\n")
    source.write_text(
        "import numpy as np\n"
        "from impression.mesh import Mesh\n"
        "from reload_include_a import vertex_x\n\n"
        "def build():\n"
        "    return Mesh(\n"
        "        vertices=np.asarray(((0, 0, 0), (vertex_x, 0, 0), (0, 1, 0)), dtype=float),\n"
        "        faces=np.asarray(((0, 1, 2),), dtype=int),\n"
        "    )\n"
    )

    factory = _scene_factory_from_path(source, cache_module=True)
    first_scene = factory()
    time.sleep(0.01)
    include_b.write_text("vertex_x = 2.0\n")
    second_scene = factory()

    assert isinstance(first_scene, Mesh)
    assert isinstance(second_scene, Mesh)
    assert first_scene.vertices[1, 0] == 1.0
    assert second_scene.vertices[1, 0] == 2.0


def test_preview_scene_factory_manual_reload_bypasses_unchanged_mtime(tmp_path: Path) -> None:
    source = tmp_path / "manual_reload_model.py"
    include = tmp_path / "manual_reload_include.py"
    include.write_text("vertex_x = 1.0\n")
    source.write_text(
        "import numpy as np\n"
        "from impression.mesh import Mesh\n"
        "from manual_reload_include import vertex_x\n\n"
        "def build():\n"
        "    return Mesh(\n"
        "        vertices=np.asarray(((0, 0, 0), (vertex_x, 0, 0), (0, 1, 0)), dtype=float),\n"
        "        faces=np.asarray(((0, 1, 2),), dtype=int),\n"
        "    )\n"
    )
    reload_generation = 0

    factory = _scene_factory_from_path(
        source,
        cache_module=True,
        reload_generation_getter=lambda: reload_generation,
    )
    first_scene = factory()
    original_mtime = include.stat().st_mtime_ns
    include.write_text("vertex_x = 2.0\n")
    os.utime(include, ns=(original_mtime, original_mtime))
    cached_scene = factory()
    reload_generation += 1
    reloaded_scene = factory()

    assert isinstance(first_scene, Mesh)
    assert isinstance(cached_scene, Mesh)
    assert isinstance(reloaded_scene, Mesh)
    assert first_scene.vertices[1, 0] == 1.0
    assert cached_scene.vertices[1, 0] == 1.0
    assert reloaded_scene.vertices[1, 0] == 2.0


def test_preview_scene_factory_failed_generation_retries_until_success(tmp_path: Path) -> None:
    source = tmp_path / "retry_model.py"
    source.write_text("def build():\n    return 1\n")
    reload_generation = 0
    factory = _scene_factory_from_path(
        source,
        cache_module=True,
        reload_generation_getter=lambda: reload_generation,
    )
    assert factory() == 1

    original_mtime = source.stat().st_mtime_ns
    reload_generation += 1
    source.write_text("def build(:\n    return 2\n")
    os.utime(source, ns=(original_mtime, original_mtime))
    with pytest.raises(SyntaxError):
        factory()

    source.write_text("def build():\n    return 2\n")
    os.utime(source, ns=(original_mtime, original_mtime))

    assert factory() == 2


def test_preview_scene_factory_forced_reload_rediscovers_transitive_watch_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rediscover_model.py"
    first_helper = tmp_path / "rediscover_first.py"
    second_helper = tmp_path / "rediscover_second.py"
    first_helper.write_text("value = 1\n")
    second_helper.write_text("value = 2\n")
    source.write_text(
        "from rediscover_first import value\n\n"
        "def build():\n"
        "    return value\n"
    )
    watched: list[tuple[Path, ...]] = []
    reload_generation = 0
    factory = _scene_factory_from_path(
        source,
        cache_module=True,
        reload_generation_getter=lambda: reload_generation,
        on_watch_paths_changed=watched.append,
    )
    assert factory() == 1

    original_mtime = source.stat().st_mtime_ns
    source.write_text(
        "from rediscover_second import value\n\n"
        "def build():\n"
        "    return value\n"
    )
    os.utime(source, ns=(original_mtime, original_mtime))
    reload_generation += 1

    assert factory() == 2
    assert second_helper.resolve() in watched[-1]
    assert first_helper.resolve() not in watched[-1]


def test_preview_command_wires_forced_reload_generation_into_live_scene_factory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "command_model.py"
    helper = tmp_path / "command_helper.py"
    control_file = tmp_path / "preview.control"
    helper.write_text("value = 1\n")
    source.write_text(
        "from command_helper import value\n\n"
        "def build():\n"
        "    return value\n"
    )
    observed: dict[str, object] = {}

    class FakePreviewer:
        unit_scale_to_mm = 1.0
        unit_name = "millimeters"
        unit_label = "mm"

        def __init__(self, console) -> None:
            self.console = console

        def show(self, **kwargs: object) -> None:
            scene_factory = kwargs["scene_factory"]
            force_scene_reload = kwargs["force_scene_reload"]
            watch_paths_getter = kwargs["watch_paths_getter"]
            assert callable(scene_factory)
            assert callable(force_scene_reload)
            assert callable(watch_paths_getter)
            original_mtime = helper.stat().st_mtime_ns
            helper.write_text("value = 2\n")
            os.utime(helper, ns=(original_mtime, original_mtime))
            observed["cached"] = scene_factory()
            force_scene_reload()
            observed["forced"] = scene_factory()
            observed["watched"] = watch_paths_getter()

    monkeypatch.setattr(cli, "PyVistaPreviewer", FakePreviewer)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "preview",
            str(source),
            "--control-file",
            str(control_file),
            "--force-window",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed["cached"] == 1
    assert observed["forced"] == 2
    assert helper.resolve() in observed["watched"]


def _write_triangle_model(path: Path) -> None:
    path.write_text(
        "import numpy as np\n"
        "from impression.mesh import Mesh\n\n"
        "def build():\n"
        "    return Mesh(\n"
        "        vertices=np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float),\n"
        "        faces=np.asarray(((0, 1, 2),), dtype=int),\n"
        "    )\n"
    )


def test_preview_help_describes_one_shot_png_export() -> None:
    result = CliRunner().invoke(
        cli.app,
        ["preview", "--help"],
        env={"COLUMNS": "160", "FORCE_COLOR": "1"},
        color=False,
    )
    normalized_output = " ".join(click.unstyle(result.output).replace("│", " ").split())

    assert result.exit_code == 0, result.output
    assert "--screenshot" in normalized_output
    assert "Render once off-screen, save a PNG, and exit" in normalized_output
    assert "without redirecting a running preview" in normalized_output


def test_preview_screenshot_bypasses_live_control_file(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    output = tmp_path / "captures" / "model.png"
    control_file = tmp_path / ".impression-preview"
    _write_triangle_model(model)
    control_contents = f"# impression-preview pid={os.getpid()}\n{tmp_path / 'other.py'}\n"
    control_file.write_text(control_contents)
    calls: list[dict[str, object]] = []

    class FakePreviewer:
        unit_scale_to_mm = 1.0
        unit_name = "millimeters"
        unit_label = "mm"

        def __init__(self, console) -> None:
            self.console = console

        def show(self, **kwargs: object) -> None:
            calls.append(kwargs)
            screenshot_path = kwargs["screenshot_path"]
            assert isinstance(screenshot_path, Path)
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(cli, "PyVistaPreviewer", FakePreviewer)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli.app,
        ["preview", str(model), "--screenshot", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert output.read_bytes() == b"\x89PNG\r\n\x1a\n"
    assert control_file.read_text() == control_contents
    assert len(calls) == 1
    assert calls[0]["watch_files"] is False
    assert calls[0]["control_file"] is None
    assert calls[0]["screenshot_path"] == output
    assert "Saved preview PNG to" in result.output
    assert "captures/model.png" in result.output.replace("\n", "")


def test_preview_screenshot_does_not_report_success_after_renderer_failure(
    monkeypatch, tmp_path: Path
) -> None:
    model = tmp_path / "model.py"
    output = tmp_path / "model.png"
    _write_triangle_model(model)

    class FailingPreviewer:
        unit_scale_to_mm = 1.0
        unit_name = "millimeters"
        unit_label = "mm"

        def __init__(self, console) -> None:
            self.console = console

        def show(self, **kwargs: object) -> None:
            raise PreviewBackendError("capture failed")

    monkeypatch.setattr(cli, "PyVistaPreviewer", FailingPreviewer)

    result = CliRunner().invoke(
        cli.app,
        ["preview", str(model), "--screenshot", str(output)],
    )

    assert result.exit_code != 0
    assert "capture failed" in result.output
    assert "Saved preview PNG" not in result.output
    assert not output.exists()


@pytest.mark.preview
def test_preview_screenshot_command_writes_decodable_png(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    output = tmp_path / "captures" / "model.png"
    control_file = tmp_path / ".impression-preview"
    _write_triangle_model(model)
    control_contents = f"# impression-preview pid={os.getpid()}\n{tmp_path / 'other.py'}\n"
    control_file.write_text(control_contents)
    env = os.environ.copy()
    env["PYVISTA_OFF_SCREEN"] = "true"
    source_root = Path(__file__).resolve().parents[1] / "src"
    inherited_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(source_root)
        if not inherited_pythonpath
        else os.pathsep.join((str(source_root), inherited_pythonpath))
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from impression.cli import app; app()",
            "preview",
            str(model),
            "--screenshot",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert control_file.read_text() == control_contents
    assert "Saved preview PNG to" in result.stdout
    assert "captures/model.png" in result.stdout.replace("\n", "")
