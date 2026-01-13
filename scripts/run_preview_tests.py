#!/usr/bin/env python3
"""Run preview CLI tests and capture screenshots/results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist" / "preview-tests"
RESULTS_FILE = DIST_DIR / "results.json"

SUITE_NAME = "preview-tests"

CASES = [
    {"name": "primitive-box", "module": "docs/examples/primitives/box_example.py"},
    {"name": "primitive-cylinder", "module": "docs/examples/primitives/cylinder_example.py"},
    {"name": "primitive-sphere", "module": "docs/examples/primitives/sphere_example.py"},
    {"name": "primitive-cone", "module": "docs/examples/primitives/cone_example.py"},
    {"name": "primitive-prism", "module": "docs/examples/primitives/prism_example.py"},
    {"name": "primitive-color-dual", "module": "docs/examples/primitives/color_dual_example.py"},
    {"name": "primitive-color-transparency", "module": "docs/examples/primitives/color_transparency_example.py"},
    {"name": "text-basic", "module": "docs/examples/text/text_basic.py"},
    {"name": "text-emoji", "module": "docs/examples/text/text_emoji.py"},
    {"name": "drafting-line-plane", "module": "docs/examples/drafting/line_plane_example.py"},
    {"name": "drafting-dimension", "module": "docs/examples/drafting/dimension_example.py"},
    {"name": "primitive-torus", "module": "docs/examples/primitives/torus_example.py"},
    {"name": "csg-union", "module": "docs/examples/csg/union_example.py"},
    {"name": "csg-union-mapping", "module": "docs/examples/csg/union_meshes_example.py"},
    {"name": "csg-tooth-parts", "module": "docs/examples/csg/tooth_parts_example.py"},
    {"name": "csg-tooth", "module": "docs/examples/csg/tooth_example.py"},
    {"name": "csg-union-teeth", "module": "docs/examples/csg/teeth_union_example.py"},
    {"name": "csg-intersection", "module": "docs/examples/csg/intersection_example.py"},
    {"name": "path-polyline", "module": "docs/examples/paths/path_example.py"},
]


def _isoformat(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def run_case(case: dict, verbose: bool = False) -> dict:
    screenshot = DIST_DIR / f"{case['name']}.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    impression_cmd = [sys.executable, '-m', 'impression.cli']
    cmd = impression_cmd + [
        'preview',
        case['module'],
        '--no-watch',
        '--screenshot',
        str(screenshot),
    ]
    env = os.environ.copy()
    env.setdefault("PYVISTA_OFF_SCREEN", "true")
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
    end_monotonic = time.perf_counter()
    ended_at = datetime.now(UTC)
    duration = end_monotonic - start_monotonic
    if verbose:
        status = "PASS" if proc.returncode == 0 else "FAIL"
        print(f"{status} - {_isoformat(ended_at)} ({duration:.2f}s)")
        print()
    return {
        "name": case["name"],
        "module": case["module"],
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "screenshot": str(screenshot.relative_to(PROJECT_ROOT)),
        "screenshot_exists": screenshot.exists(),
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
        if case["returncode"] != 0 or not case.get("screenshot_exists", False)
    ]
    for case in failures:
        reasons = []
        if case["returncode"] != 0:
            reasons.append(f"returncode={case['returncode']}")
        if not case.get("screenshot_exists", False):
            reasons.append("missing screenshot")
        reason_text = ", ".join(reasons)
        print(f"[FAIL] {case['name']} ({case['module']}) [{reason_text}]", file=sys.stderr)
    status_text = "PASS" if not failures else "FAIL"
    print(f"suite end - {_isoformat(suite_end)} ({status_text})")
    print(f"Wrote results to {RESULTS_FILE}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
