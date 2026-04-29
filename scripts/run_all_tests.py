#!/usr/bin/env python3
"""Run every Impression verification suite (preview + STL)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUITES = [
    ("preview", PROJECT_ROOT / "scripts" / "run_preview_tests.py"),
    ("stl", PROJECT_ROOT / "scripts" / "run_stl_tests.py"),
]


def run_suite(name: str, path: Path) -> int:
    print(f"=== Running {name} suite ===")
    cmd = [sys.executable, str(path)]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    print(f"=== {name} suite exited with {proc.returncode} ===\n")
    return proc.returncode


def main() -> int:
    failures = []
    for name, path in SUITES:
        if not path.exists():
            print(f"[WARN] Skipping {name} suite ({path} missing)")
            continue
        code = run_suite(name, path)
        if code != 0:
            failures.append((name, code))

    if failures:
        print("Some suites failed:")
        for name, code in failures:
            print(f" - {name}: exit {code}")
        return 1

    print("All suites passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
