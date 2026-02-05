#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import zipfile


def _iter_docs_files(root: pathlib.Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == ".DS_Store":
            continue
        if "__pycache__" in path.parts:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Package docs/ into a release zip asset.")
    parser.add_argument("--ref", required=True, help="Release ref/tag (e.g. v0.0.1a4).")
    parser.add_argument(
        "--docs-dir",
        default="docs",
        help="Path to docs directory (default: docs).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output zip path (default: dist/impression-docs-<ref>.zip).",
    )
    args = parser.parse_args()

    docs_dir = pathlib.Path(args.docs_dir).resolve()
    if not docs_dir.exists() or not docs_dir.is_dir():
        raise SystemExit(f"docs directory not found: {docs_dir}")

    output = pathlib.Path(args.output or f"dist/impression-docs-{args.ref}.zip").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in _iter_docs_files(docs_dir):
            rel = file_path.relative_to(docs_dir)
            archive.write(file_path, arcname=str(pathlib.Path("docs") / rel))

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
