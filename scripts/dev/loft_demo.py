#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SHOWCASE_MODEL = REPO_ROOT / "docs/examples/loft/loft_showcase.py"
SELECTOR_FILE = REPO_ROOT / "docs/examples/loft/.loft-showcase"
DEMO_ORDER = (
    "caps-lab",
    "path-choreo",
    "topology-transitions",
    "text-sculpt",
)


def _write_selector(name: str) -> None:
    SELECTOR_FILE.write_text(name.strip().lower() + "\n")


def _read_selector() -> str:
    if not SELECTOR_FILE.exists():
        return DEMO_ORDER[0]
    value = SELECTOR_FILE.read_text().strip().lower()
    return value or DEMO_ORDER[0]


async def _play(interval: float, loops: int) -> None:
    if loops <= 0:
        raise ValueError("loops must be >= 1")
    total = loops * len(DEMO_ORDER)
    for i in range(total):
        name = DEMO_ORDER[i % len(DEMO_ORDER)]
        _write_selector(name)
        print(f"[loft-demo] switched to: {name}")
        await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Switch/play loft showcase demos.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available demos.")
    sub.add_parser("current", help="Show currently selected demo.")

    set_parser = sub.add_parser("set", help="Set active demo.")
    set_parser.add_argument("name", choices=[*DEMO_ORDER, "auto"])

    play_parser = sub.add_parser("play", help="Auto-cycle demos.")
    play_parser.add_argument("--interval", type=float, default=6.0, help="Seconds between switches.")
    play_parser.add_argument("--loops", type=int, default=2, help="Number of full cycles through the demo list.")

    args = parser.parse_args()

    if args.command == "list":
        print("Available demos:")
        for name in DEMO_ORDER:
            print(f"- {name}")
        print("- auto")
        print()
        print("Preview command:")
        print(f"impression preview {SHOWCASE_MODEL}")
        return

    if args.command == "current":
        print(_read_selector())
        return

    if args.command == "set":
        _write_selector(args.name)
        print(f"[loft-demo] selected: {args.name}")
        return

    if args.command == "play":
        asyncio.run(_play(interval=float(args.interval), loops=int(args.loops)))
        return


if __name__ == "__main__":
    main()

