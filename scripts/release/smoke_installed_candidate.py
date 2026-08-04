#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import impression
from impression.cli import _extract_docs_archive, _scene_factory_from_path
from impression.preview import PyVistaPreviewer
from rich.console import Console


MODEL_SOURCE = """\
from impression.modeling import make_box

def build():
    return make_box(size=(2.0, 3.0, 4.0))
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke an installed Impression candidate.")
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--docs-archive", type=Path, required=True)
    args = parser.parse_args()

    if impression.__version__ != args.expected_version:
        raise SystemExit(
            f"installed version {impression.__version__!r} != {args.expected_version!r}"
        )

    with tempfile.TemporaryDirectory(prefix="impression-release-smoke-") as temporary:
        root = Path(temporary)
        model = root / "candidate_model.py"
        model.write_text(MODEL_SOURCE)

        scene = _scene_factory_from_path(model)()
        previewer = PyVistaPreviewer(console=Console())
        datasets = previewer.collect_datasets(scene)
        merged = previewer.combine_to_mesh(datasets)
        if merged.n_faces <= 0:
            raise SystemExit("installed preview consumption produced no faces")

        output = root / "candidate.stl"
        subprocess.run(
            (
                sys.executable,
                "-m",
                "impression.cli",
                "export",
                str(model),
                "--output",
                str(output),
            ),
            cwd=root,
            check=True,
        )
        if not output.is_file() or output.stat().st_size <= 84:
            raise SystemExit("installed export route did not produce a non-empty STL")

        docs_destination = root / "docs"
        _extract_docs_archive(args.docs_archive.read_bytes(), docs_destination, clean=True)
        if not (docs_destination / "cli.md").is_file():
            raise SystemExit("installed docs route did not produce cli.md")

    print(f"installed candidate smoke passed: Impression {impression.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
