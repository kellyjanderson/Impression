from __future__ import annotations

import subprocess
import sys


def _run(label: str, code: str) -> int:
    try:
        result = subprocess.run([sys.executable, "-c", code], timeout=5)
    except subprocess.TimeoutExpired:
        print(f"{label}: exit=timeout")
        return 124
    print(f"{label}: exit={result.returncode}")
    return result.returncode


def main() -> int:
    failures = 0

    failures += _run(
        "pyvista_plotter_plain",
        "import pyvista as pv; pv.Plotter(off_screen=True).close()",
    )

    failures += _run(
        "impression_backend_init",
        "from impression.preview import PyVistaPreviewer; from rich.console import Console; "
        "PyVistaPreviewer(Console())._ensure_backend()",
    )

    failures += _run(
        "old_mesh_pyvista",
        "import pyvista as pv; pv.Box(bounds=(-1,1,-1,1,-1,1))",
    )

    failures += _run(
        "new_mesh_to_pyvista",
        "from impression.modeling import make_box; "
        "from impression.mesh import mesh_to_pyvista; "
        "mesh_to_pyvista(make_box())",
    )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
