"""Interactive artifact preview launched from the review workbench."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from impression._vtk_runtime import ensure_vtk_runtime


@dataclass(frozen=True)
class InteractivePreviewLaunch:
    accepted: bool
    diagnostic: str | None = None
    process: subprocess.Popen[bytes] | None = field(default=None, compare=False)


def interactive_preview_command(artifact_path: Path, *, title: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "impression.devtools.reference_review.ui.interactive_preview",
        "--stl",
        str(artifact_path),
        "--title",
        title,
    ]


def launch_interactive_stl_preview(artifact_path: Path, *, title: str) -> InteractivePreviewLaunch:
    artifact_path = Path(artifact_path)
    if artifact_path.suffix.lower() != ".stl":
        return InteractivePreviewLaunch(False, "unsupported-artifact-kind")
    if not artifact_path.is_file():
        return InteractivePreviewLaunch(False, "missing-artifact")
    process = subprocess.Popen(
        interactive_preview_command(artifact_path, title=title),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return InteractivePreviewLaunch(True, process=process)


def show_interactive_stl_preview(artifact_path: Path, *, title: str) -> None:
    ensure_vtk_runtime()
    import pyvista as pv

    mesh = pv.read(artifact_path)
    pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=(1280, 800))
    plotter.set_background("#090c10", top="#1b2333")
    try:
        plotter.enable_eye_dome_lighting()
        plotter.add_axes(interactive=True)
        plotter.show_bounds(grid="front", location="outer")
    except Exception:
        pass
    plotter.add_mesh(
        mesh,
        color="#6ab0ff",
        smooth_shading=True,
        specular=0.2,
        show_edges=False,
    )
    _reset_camera(plotter, mesh.bounds)
    home_camera = plotter.camera_position

    def go_home() -> None:
        plotter.camera_position = home_camera
        plotter.reset_camera_clipping_range()
        plotter.render()

    plotter.add_key_event("v", go_home)
    plotter.show(title=title, auto_close=True)


def _reset_camera(plotter, bounds) -> None:
    x_center = (bounds[0] + bounds[1]) / 2.0
    y_center = (bounds[2] + bounds[3]) / 2.0
    z_center = (bounds[4] + bounds[5]) / 2.0
    diag = math.sqrt(
        (bounds[1] - bounds[0]) ** 2
        + (bounds[3] - bounds[2]) ** 2
        + (bounds[5] - bounds[4]) ** 2
    )
    distance = max(diag, 1.0) * 1.4
    plotter.camera_position = [
        (
            x_center + distance * 0.6,
            y_center + distance * 0.6,
            z_center + distance * 0.5,
        ),
        (x_center, y_center, z_center),
        (0.0, 0.0, 1.0),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open an interactive STL preview.")
    parser.add_argument("--stl", required=True, type=Path)
    parser.add_argument("--title", default="Impression Preview")
    args = parser.parse_args(argv)
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "false")
    show_interactive_stl_preview(args.stl, title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
