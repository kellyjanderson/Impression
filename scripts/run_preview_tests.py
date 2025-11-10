#!/usr/bin/env python3
"""Run preview CLI tests and capture screenshots/results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist" / "preview-tests"
RESULTS_FILE = DIST_DIR / "results.json"

CASES = [
    {
        "name": "primitive-box",
        "module": "docs/examples/primitives/box_example.py",
    },
    {
        "name": "primitive-cylinder",
        "module": "docs/examples/primitives/cylinder_example.py",
    },
    {
        "name": "primitive-sphere",
        "module": "docs/examples/primitives/sphere_example.py",
    },
    {
        "name": "primitive-torus",
        "module": "docs/examples/primitives/torus_example.py",
    },
    {
        "name": "csg-union",
        "module": "docs/examples/csg/union_example.py",
    },
    {
        "name": "csg-difference",
        "module": "docs/examples/csg/difference_example.py",
    },
    {
        "name": "csg-intersection",
        "module": "docs/examples/csg/intersection_example.py",
    },
    {
        "name": "path-polyline",
        "module": "docs/examples/paths/path_example.py",
    },
]


def run_case(case: dict) -> dict:
    screenshot = DIST_DIR / f"{case['name']}.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "impression",
        "preview",
        case["module"],
        "--no-watch",
        "--screenshot",
        str(screenshot),
    ]
    env = os.environ.copy()
    env.setdefault("PYVISTA_OFF_SCREEN", "true")
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "name": case["name"],
        "module": case["module"],
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "screenshot": str(screenshot.relative_to(PROJECT_ROOT)),
    }


def main() -> int:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_case(case) for case in CASES]
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cases": results,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2))
    failures = [case for case in results if case["returncode"] != 0]
    for case in failures:
        print(f"[FAIL] {case['name']} ({case['module']})", file=sys.stderr)
    print(f"Wrote results to {RESULTS_FILE}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
