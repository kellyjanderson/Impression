#!/usr/bin/env python3
"""Export models to STL and verify watertightness and basic metrics."""

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
DIST_DIR = PROJECT_ROOT / "dist" / "stl-tests"
RESULTS_FILE = DIST_DIR / "results.json"
SUITE_NAME = "stl-tests"

CASES = [
    {"name": "stl-box", "module": "docs/examples/primitives/box_example.py"},
    {"name": "stl-csg-difference", "module": "docs/examples/csg/difference_example.py"},
    {"name": "stl-csg-union-mapping", "module": "docs/examples/csg/union_meshes_example.py"},
    {
        "name": "stl-csg-tooth-parts",
        "module": "docs/examples/csg/tooth_parts_example.py",
        "expected_volume": 99.3,
        "volume_tol": 0.5,
    },
    {
        "name": "stl-csg-tooth",
        "module": "docs/examples/csg/tooth_example.py",
        "expected_volume": 79.6,
        "volume_tol": 0.5,
        "expect_manifold": True,
    },
    {"name": "stl-csg-union-teeth", "module": "docs/examples/csg/teeth_union_example.py"},
]


def _isoformat(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _is_watertight(mesh: pv.DataSet) -> tuple[bool, int]:
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        non_manifold_edges=True,
        manifold_edges=False,
    )
    open_edges = edges.n_cells
    return open_edges == 0, open_edges


def _volume(mesh: pv.DataSet) -> float:
    vol = getattr(mesh, "volume", None)
    return float(vol) if vol is not None else 0.0


def run_case(case: dict, verbose: bool = False) -> dict:
    output = DIST_DIR / f"{case['name']}.stl"
    output.parent.mkdir(parents=True, exist_ok=True)
    impression_cmd = [sys.executable, '-m', 'impression.cli']
    cmd = impression_cmd + [
        'export',
        case['module'],
        '--output',
        str(output),
        '--overwrite',
    ]
    env = os.environ.copy()
    started_at = datetime.now(UTC)
    start_monotonic = time.perf_counter()
    if verbose:
        print(f"{case['name']} - {_isoformat(started_at)}")
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    ended_at = datetime.now(UTC)
    duration = time.perf_counter() - start_monotonic

    watertight = None
    open_edges = None
    analysis_error = None
    n_points = None
    n_cells = None
    volume = None
    manifold = None

    if proc.returncode == 0 and output.exists():
        try:
            mesh = pv.read(output)
            n_points = mesh.n_points
            n_cells = mesh.n_cells
            watertight, open_edges = _is_watertight(mesh)
            volume = _volume(mesh)
            manifold = getattr(mesh, "is_manifold", None)
        except Exception as exc:  # pragma: no cover - pyvista read failure
            watertight = False
            analysis_error = str(exc)

    success = proc.returncode == 0 and watertight is True

    expected_volume = case.get("expected_volume")
    volume_tol = case.get("volume_tol", 0.0)
    if success and expected_volume is not None and volume is not None:
        if abs(volume - expected_volume) > volume_tol:
            success = False
            analysis_error = f"volume {volume:.3f} outside expected {expected_volume}±{volume_tol}"

    expect_manifold = case.get("expect_manifold")
    if success and expect_manifold and manifold is False:
        success = False
        analysis_error = "mesh is non-manifold"

    if verbose:
        status = "PASS" if success else "FAIL"
        print(f"{status} - {_isoformat(ended_at)} ({duration:.2f}s)")
        if watertight is False:
            detail = analysis_error or f"open edges: {open_edges}"
            print(f"  {detail}")
        elif analysis_error:
            print(f"  {analysis_error}")
        print()

    return {
        "name": case["name"],
        "module": case["module"],
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
        "volume": volume,
        "started_at": _isoformat(started_at),
        "ended_at": _isoformat(ended_at),
        "duration_seconds": duration,
    }


def main() -> int:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    suite_start = datetime.now(UTC)
    print(f"Starting test suite {SUITE_NAME}")
    print(f"time: {_isoformat(suite_start)}")
    print("--")
    results = [run_case(case, verbose=True) for case in CASES]
    suite_end = datetime.now(UTC)
    payload = {
        "suite": SUITE_NAME,
        "suite_started_at": _isoformat(suite_start),
        "suite_ended_at": _isoformat(suite_end),
        "timestamp": _isoformat(suite_end),
        "cases": results,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2))

    failures = [
        case
        for case in results
        if case["returncode"] != 0
        or case["watertight"] is not True
        or case.get("analysis_error")
        or not case.get("stl_exists", False)
    ]
    status_text = "PASS" if not failures else "FAIL"
    print(f"suite end - {_isoformat(suite_end)} ({status_text})")
    print(f"Wrote results to {RESULTS_FILE}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
