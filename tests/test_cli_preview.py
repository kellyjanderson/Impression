from __future__ import annotations

import time
from pathlib import Path

from impression.cli import _scene_factory_from_path
from impression.io import save_impress
from impression.mesh import Mesh
from impression.modeling import make_box


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
