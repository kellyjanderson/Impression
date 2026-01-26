from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("PYVISTA_OFF_SCREEN", "true")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("IMPRESSION_PYVISTA_SAFE", "0")
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{pythonpath}" if pythonpath else str(PROJECT_ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _format_failure(result: subprocess.CompletedProcess[str]) -> str:
    return f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.preview
def test_pyvista_show_without_impression():
    code = dedent(
        """
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(pv.Box())
        plotter.show(auto_close=True, interactive=False)
        plotter.close()
        """
    )
    result = _run_python(code)
    assert result.returncode == 0, _format_failure(result)


@pytest.mark.preview
def test_pyvista_show_after_impression_import():
    code = dedent(
        """
        import impression
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(pv.Box())
        plotter.show(auto_close=True, interactive=False)
        plotter.close()
        """
    )
    result = _run_python(code)
    assert result.returncode == 0, _format_failure(result)


@pytest.mark.preview
def test_pyvista_show_after_modeling_import():
    code = dedent(
        """
        import impression.modeling
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(pv.Box())
        plotter.show(auto_close=True, interactive=False)
        plotter.close()
        """
    )
    result = _run_python(code)
    assert result.returncode == 0, _format_failure(result)


@pytest.mark.preview
def test_internal_mesh_show_after_modeling_import():
    code = dedent(
        """
        from impression.modeling import make_box
        from impression.mesh import mesh_to_pyvista
        import pyvista as pv

        mesh = mesh_to_pyvista(make_box())
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh)
        plotter.show(auto_close=True, interactive=False)
        plotter.close()
        """
    )
    result = _run_python(code)
    assert result.returncode == 0, _format_failure(result)
