from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

from rich.console import Console

from impression.preview import PyVistaPreviewer


def _load_demo_module(path: Path):
    spec = importlib.util.spec_from_file_location("impression_wow_demo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load demo module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_at"):
        raise RuntimeError(f"{path} must define build_at(elapsed_seconds: float)")
    return module


def run_demo(
    demo_file: Path,
    *,
    duration_seconds: float,
    interval_seconds: float,
) -> None:
    module = _load_demo_module(demo_file)
    console = Console()
    previewer = PyVistaPreviewer(console=console)
    pv = previewer._ensure_backend()

    plotter = pv.Plotter(window_size=(1500, 920))
    previewer._configure_plotter(plotter)
    start = time.monotonic()

    def render_frame(elapsed: float, align_camera: bool) -> None:
        scene = module.build_at(float(elapsed))
        datasets = previewer.collect_datasets(scene)
        previewer._apply_scene(
            plotter,
            datasets,
            show_edges=False,
            face_edges=False,
            align_camera=align_camera,
        )
        plotter.render()

    render_frame(0.0, align_camera=True)

    def tick() -> None:
        elapsed = time.monotonic() - start
        render_frame(min(elapsed, duration_seconds), align_camera=False)

    cleanup = previewer._install_timer_callback(
        plotter,
        tick,
        interval_seconds=max(0.05, float(interval_seconds)),
    )
    try:
        plotter.show(title="Impression WOW Demo (Standalone)")
    finally:
        if cleanup is not None:
            cleanup()
        plotter.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the WOW socks-off demo as a standalone PyVista app.")
    parser.add_argument(
        "--demo",
        type=Path,
        default=Path(__file__).with_name("off.py"),
        help="Path to a demo module that defines build_at(elapsed_seconds).",
    )
    parser.add_argument("--duration", type=float, default=180.0, help="Timeline duration in seconds.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between scene rebuilds.")
    args = parser.parse_args()
    run_demo(
        demo_file=args.demo.resolve(),
        duration_seconds=float(args.duration),
        interval_seconds=float(args.interval),
    )


if __name__ == "__main__":
    main()

