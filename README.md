# Impression

Impression is an experimental parametric 3D modeling platform for rapid spatial ideation. The goal is to support both dynamic (Python-driven) and declarative pipelines for building, previewing, and exporting solid geometry that is suitable for fabrication workflows such as 3D printing.

## Current focus

- Command-line preview tool that can hot-reload model definitions.
- Modular architecture that lets us experiment with popular geometric kernels and renderers (e.g., Manifold, PyVista, pygfx).
- Foundation for exporting watertight meshes to STL and other CAD-friendly formats.
- Mesh primitives, CSG helpers, and path abstractions exposed via `impression.modeling`.
- Documentation index: `docs/index.md` lists available features and runnable examples.
- CLI manual: `docs/cli.md` describes `impression preview` / `export` options and usage.
- VS Code helper extension under `ide/vscode-extension/` for launching previews, exports, and regression tests from the editor.

## Roadmap highlights

1. **Preview CLI** – invoke `impression preview path/to/model.py` to render and manipulate scenes (orbit, pan, zoom, strafe).
2. **Dynamic runtime** – let Python scripts define parameterized solids with runtime overrides (CLI flags, config files, or sockets).
3. **Declarative mode** – ingest declarative scene graphs (JSON/TOML) that map to the same geometry pipeline.
4. **Exporters** – transform the canonical geometry representation into STL, AMF, and STEP variants.

## Getting Started

```bash
git clone https://github.com/kellyjanderson/Impression.git
cd Impression
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

After installation you can run `impression --help` from anywhere in that virtual environment.

The CLI also writes `~/.impression/env` with an `IMPRESSION_PY` export that the VS Code
extension (and other tooling) can source. Add the following line to your shell config if you
haven't already:

```bash
source ~/.impression/env
```

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
impression --help
```

The CLI now opens a PyVista window for interactive previewing; the renderer backend will evolve as we integrate additional geometric kernels.

## Preview workflow

1. Define a `build()` function in your model module that returns one or more [PyVista](https://docs.pyvista.org/) datasets (e.g., `pv.Cube()`, `pv.Sphere()`, or a list of meshes).
2. Run the previewer:

```bash
impression preview examples/hello_cube.py

# color-aware example
impression preview docs/examples/primitives/color_dual_example.py

# text primitive demo
impression preview docs/examples/text/text_basic.py

# drafting helpers
impression preview docs/examples/drafting/line_plane_example.py

# Impression mark
impression preview docs/examples/logo/impression_mark.py --hide-edges

# programmatic modeling with primitives/CSG/paths
python - <<'PY'
from impression.modeling import make_box, make_cylinder, boolean_union, Path
box = make_box(size=(2, 2, 1))
post = make_cylinder(radius=0.4, height=2.0)
result = boolean_union([box, post])
path = Path.from_points([(0, 0, 0), (4, 0, 0), (4, 2, 0)], closed=False)
print("cells:", result.n_cells, "path length:", path.length())
PY

# preview test suite (saves screenshots/results under dist/preview-tests)
scripts/run_preview_tests.py
```

The PyVista window supports orbit, pan, and zoom out of the box. Files are watched by default, so saving changes triggers a hot reload in the same window (disable with `--no-watch` if you just want a single render).

## Export workflow

```bash
impression export examples/hello_cube.py --output artifacts/hello.stl --overwrite
```

The exporter loads the same `build()` entry point, merges all returned PyVista datasets, and writes a watertight STL in binary format (use `--ascii` for text output). Existing files are protected unless `--overwrite` is specified.
- Strategic roadmap: see `docs/project-plan.md` for primitives, CSG, CAD integration, and helper utilities (e.g., auto-rounding sharp faces).
