from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

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
        env={"COLUMNS": "160"},
        color=False,
    )
    normalized_output = " ".join(result.output.replace("│", " ").split())

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
