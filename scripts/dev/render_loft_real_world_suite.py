#!/usr/bin/env python3
"""Render + export loft real-world examples, then analyze images/STLs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "dist" / "loft-real-world"
RESULTS_FILE = OUT_DIR / "results.json"
CLI_BIN = Path(sys.executable).with_name("impression")

CASES = [
    ("hvac-round-to-rect", "docs/examples/loft/real_world/loft_hvac_round_to_rect_example.py"),
    ("ergonomic-handle", "docs/examples/loft/real_world/loft_ergonomic_handle_example.py"),
    ("bottle-nozzle", "docs/examples/loft/real_world/loft_bottle_nozzle_family_example.py"),
    ("wearable-enclosure", "docs/examples/loft/real_world/loft_wearable_enclosure_example.py"),
    ("topology-vent", "docs/examples/loft/real_world/loft_topology_vent_transition_example.py"),
    ("hero-showcase", "docs/examples/loft/real_world/loft_hero_showcase_example.py"),
]


@dataclass
class CaseResult:
    name: str
    module: str
    preview_returncode: int
    export_returncode: int
    screenshot: str
    screenshot_exists: bool
    stl: str
    stl_exists: bool
    watertight: bool | None
    open_edges: int | None
    n_points: int | None
    n_cells: int | None
    non_dark_ratio: float | None
    label_pixel_ratio: float | None
    readable_label: bool
    preview_stdout: str
    preview_stderr: str
    export_stdout: str
    export_stderr: str


def _run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def _analyze_image(path: Path) -> tuple[float, float, bool]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    non_dark = float((luminance > 18).mean())

    # Label color target: #ff00ff (allowing for shading in preview).
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    label_mask = (r > 160) & (g < 120) & (b > 160)
    label_ratio = float(label_mask.mean())
    readable = label_ratio > 0.002
    return non_dark, label_ratio, readable


def _analyze_stl(path: Path) -> tuple[bool, int, int, int]:
    mesh = pv.read(path)
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        non_manifold_edges=True,
        manifold_edges=False,
    )
    open_edges = int(edges.n_cells)
    watertight = open_edges == 0
    return watertight, open_edges, int(mesh.n_points), int(mesh.n_cells)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYVISTA_OFF_SCREEN", "true")
    if not CLI_BIN.exists():
        raise SystemExit(f"Missing CLI executable at {CLI_BIN}. Activate the project venv first.")

    results: list[CaseResult] = []
    failures: list[str] = []

    for name, module in CASES:
        screenshot = OUT_DIR / f"{name}.png"
        stl = OUT_DIR / f"{name}.stl"
        if screenshot.exists():
            screenshot.unlink()
        if stl.exists():
            stl.unlink()
        preview_cmd = [
            str(CLI_BIN),
            "preview",
            module,
            "--no-watch",
            "--screenshot",
            str(screenshot),
        ]
        export_cmd = [
            str(CLI_BIN),
            "export",
            module,
            "--output",
            str(stl),
            "--overwrite",
        ]

        preview = _run(preview_cmd, env)
        export = _run(export_cmd, env)

        non_dark_ratio = None
        label_ratio = None
        readable_label = False
        if screenshot.exists():
            non_dark_ratio, label_ratio, readable_label = _analyze_image(screenshot)

        watertight = None
        open_edges = None
        n_points = None
        n_cells = None
        if stl.exists() and export.returncode == 0:
            watertight, open_edges, n_points, n_cells = _analyze_stl(stl)

        result = CaseResult(
            name=name,
            module=module,
            preview_returncode=preview.returncode,
            export_returncode=export.returncode,
            screenshot=str(screenshot.relative_to(PROJECT_ROOT)),
            screenshot_exists=screenshot.exists(),
            stl=str(stl.relative_to(PROJECT_ROOT)),
            stl_exists=stl.exists(),
            watertight=watertight,
            open_edges=open_edges,
            n_points=n_points,
            n_cells=n_cells,
            non_dark_ratio=non_dark_ratio,
            label_pixel_ratio=label_ratio,
            readable_label=readable_label,
            preview_stdout=preview.stdout.strip(),
            preview_stderr=preview.stderr.strip(),
            export_stdout=export.stdout.strip(),
            export_stderr=export.stderr.strip(),
        )
        results.append(result)

        if preview.returncode != 0:
            failures.append(f"{name}: preview failed")
        if export.returncode != 0:
            failures.append(f"{name}: export failed")
        if not screenshot.exists():
            failures.append(f"{name}: screenshot missing")
        if not stl.exists():
            failures.append(f"{name}: stl missing")
        if watertight is False:
            failures.append(f"{name}: non-watertight (open_edges={open_edges})")
        if not readable_label:
            failures.append(f"{name}: label readability check failed (ratio={label_ratio})")

    RESULTS_FILE.write_text(
        json.dumps(
            {
                "suite": "loft-real-world",
                "cases": [asdict(r) for r in results],
                "failures": failures,
            },
            indent=2,
        )
    )

    print(f"Wrote {RESULTS_FILE.relative_to(PROJECT_ROOT)}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[PASS] loft real-world render/export suite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
