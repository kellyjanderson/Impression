from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from impression.cli import app
from impression.io import write_stl_atomically
from impression.mesh import Mesh


def _write_mesh_model(path: Path, vertices: list[list[float]], faces: list[list[int]]) -> None:
    vertex_literal = repr(vertices).replace("nan", 'float("nan")')
    path.write_text(
        "import numpy as np\n"
        "from impression.mesh import Mesh\n\n"
        "def build():\n"
        f"    return Mesh(np.asarray({vertex_literal}, dtype=float), "
        f"np.asarray({faces!r}, dtype=int))\n"
    )


def _invoke_export(model: Path, output: Path, *options: str):
    return CliRunner().invoke(
        app,
        ("export", str(model), "--output", str(output), *options),
        terminal_width=200,
    )


@pytest.fixture
def tetrahedron() -> tuple[list[list[float]], list[list[int]]]:
    return (
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    )


@pytest.mark.parametrize("ascii_output", [False, True])
def test_export_writes_valid_manufacturing_stl_atomically(
    tmp_path: Path,
    tetrahedron: tuple[list[list[float]], list[list[int]]],
    ascii_output: bool,
) -> None:
    model = tmp_path / "valid.py"
    output = tmp_path / ("valid-ascii.stl" if ascii_output else "valid-binary.stl")
    _write_mesh_model(model, *tetrahedron)

    result = _invoke_export(model, output, *(('--ascii',) if ascii_output else ()))

    assert result.exit_code == 0, result.output
    payload = output.read_bytes()
    if ascii_output:
        assert payload.startswith(b"solid impression")
        assert payload.rstrip().endswith(b"endsolid impression")
    else:
        assert len(payload) == 84 + 50 * 4
        assert int.from_bytes(payload[80:84], "little") == 4
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


@pytest.mark.parametrize(
    ("vertices", "faces", "diagnostic"),
    [
        ([], [], "empty mesh"),
        ([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0]], [[0, 1, 2]], "invalid vertices"),
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0, 1, 2]], "degenerate faces"),
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0, 1, 2]], "boundary edges (not watertight)"),
        (
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0, 1, 2], [1, 0, 3], [0, 1, 4]],
            "non-manifold edges",
        ),
    ],
)
def test_export_refuses_invalid_mesh_without_mutating_target(
    tmp_path: Path,
    vertices: list[list[float]],
    faces: list[list[int]],
    diagnostic: str,
) -> None:
    model = tmp_path / "invalid.py"
    output = tmp_path / "sentinel.stl"
    sentinel = b"existing validated artifact"
    output.write_bytes(sentinel)
    _write_mesh_model(model, vertices, faces)

    result = _invoke_export(model, output, "--overwrite")

    assert result.exit_code != 0
    assert "Manufacturing STL integrity check failed" in result.output
    assert diagnostic in result.output
    assert output.read_bytes() == sentinel
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_atomic_stl_replace_failure_preserves_existing_target(
    tmp_path: Path,
    tetrahedron: tuple[list[list[float]], list[list[int]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import impression.io.stl as stl_module

    output = tmp_path / "sentinel.stl"
    sentinel = b"existing validated artifact"
    output.write_bytes(sentinel)
    mesh = Mesh(np.asarray(tetrahedron[0]), np.asarray(tetrahedron[1]))

    def refuse_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated replace refusal")

    monkeypatch.setattr(stl_module.os, "replace", refuse_replace)

    with pytest.raises(OSError, match="simulated replace refusal"):
        write_stl_atomically(mesh, output)

    assert output.read_bytes() == sentinel
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_export_uses_surface_policy_through_real_command(tmp_path: Path) -> None:
    model = tmp_path / "surface.py"
    output = tmp_path / "surface.stl"
    model.write_text(
        "from impression.modeling import make_box\n\n"
        "def build():\n"
        "    return make_box(size=(2.0, 3.0, 4.0))\n"
    )

    result = _invoke_export(model, output)

    assert result.exit_code == 0, result.output
    assert output.stat().st_size > 84
