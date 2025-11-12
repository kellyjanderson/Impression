#!/usr/bin/env python3
"""Focused regression suite for text rendering/export."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pyvista as pv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist" / "text-suite"
PREVIEW_DIR = DIST_DIR / "previews"
RESULTS_FILE = DIST_DIR / "results.json"

PREVIEW_CASES = [
    {"name": "text-basic", "module": "docs/examples/text/text_basic.py"},
    {"name": "text-emoji", "module": "docs/examples/text/text_emoji.py"},
    {"name": "text-logo", "module": "docs/examples/logo/impression_mark.py"},
]

STL_CASES = [
    {"name": "stl-text-basic", "module": "docs/examples/text/text_basic.py"},
    {"name": "stl-text-emoji", "module": "docs/examples/text/text_emoji.py"},
]


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def run_preview_case(case: dict) -> dict:
    screenshot = PREVIEW_DIR / f"{case['name']}.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "impression",
        "preview",
        case["module"],
        "--no-watch",
        "--screenshot",
        str(screenshot),
        "--hide-edges",
    ]
    env = os.environ.copy()
    env.setdefault("PYVISTA_OFF_SCREEN", "true")
    started = datetime.now(UTC)
    start_perf = time.perf_counter()
    print(f"preview {case['name']} - {_iso(started)}")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    duration = time.perf_counter() - start_perf
    ended = datetime.now(UTC)
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"{status} ({duration:.2f}s)\n")
    return {
        **case,
        "kind": "preview",
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "screenshot": str(screenshot.relative_to(PROJECT_ROOT)),
        "screenshot_exists": screenshot.exists(),
        "started_at": _iso(started),
        "ended_at": _iso(ended),
        "duration_seconds": duration,
    }


def _read_mesh(path: Path) -> pv.DataSet:
    mesh = pv.read(path)
    mesh.clean(inplace=True)
    return mesh


def _check_watertight(mesh: pv.DataSet) -> tuple[bool, int]:
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        non_manifold_edges=True,
        manifold_edges=False,
    )
    count = edges.n_cells
    return count == 0, count


def run_stl_case(case: dict) -> dict:
    output = DIST_DIR / f"{case['name']}.stl"
    cmd = [
        "impression",
        "export",
        case["module"],
        "--output",
        str(output),
        "--overwrite",
    ]
    env = os.environ.copy()
    started = datetime.now(UTC)
    start_perf = time.perf_counter()
    print(f"stl {case['name']} - {_iso(started)}")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    duration = time.perf_counter() - start_perf
    ended = datetime.now(UTC)

    watertight = None
    open_edges = None
    n_points = None
    n_cells = None
    analysis_error = None
    if proc.returncode == 0 and output.exists():
        try:
            mesh = _read_mesh(output)
            n_points = mesh.n_points
            n_cells = mesh.n_cells
            watertight, open_edges = _check_watertight(mesh)
        except Exception as exc:  # pragma: no cover - PyVista read failure
            analysis_error = str(exc)
            watertight = False

    status = "PASS" if proc.returncode == 0 and watertight else "FAIL"
    print(f"{status} ({duration:.2f}s)")
    if watertight is False and open_edges is not None:
        print(f"  open edges: {open_edges}")
    if analysis_error:
        print(f"  analysis error: {analysis_error}")
    print()

    return {
        **case,
        "kind": "stl",
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "stl_path": str(output.relative_to(PROJECT_ROOT)),
        "stl_exists": output.exists(),
        "watertight": watertight,
        "open_edge_count": open_edges,
        "analysis_error": analysis_error,
        "n_points": n_points,
        "n_cells": n_cells,
        "started_at": _iso(started),
        "ended_at": _iso(ended),
        "duration_seconds": duration,
    }


def main() -> int:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    suite_start = datetime.now(UTC)
    print(f"Starting text suite at {_iso(suite_start)}\n")

    results: list[dict] = []
    for case in PREVIEW_CASES:
        results.append(run_preview_case(case))
    for case in STL_CASES:
        results.append(run_stl_case(case))

    suite_end = datetime.now(UTC)
    payload = {
        "suite": "text-suite",
        "suite_started_at": _iso(suite_start),
        "suite_ended_at": _iso(suite_end),
        "cases": results,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2))

    failures = [
        case
        for case in results
        if case["returncode"] != 0
        or (case["kind"] == "preview" and not case.get("screenshot_exists", False))
        or (case["kind"] == "stl" and not case.get("watertight", False))
    ]
    status = "PASS" if not failures else "FAIL"
    print(f"Suite finished at {_iso(suite_end)} [{status}]\nResults: {RESULTS_FILE}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
