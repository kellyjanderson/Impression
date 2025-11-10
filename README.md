# Impression

Impression is an experimental parametric 3D modeling platform for rapid spatial ideation. The goal is to support both dynamic (Python-driven) and declarative pipelines for building, previewing, and exporting solid geometry that is suitable for fabrication workflows such as 3D printing.

## Current focus

- Command-line preview tool that can hot-reload model definitions.
- Modular architecture that lets us experiment with popular geometric kernels and renderers (e.g., Manifold, PyVista, pygfx).
- Foundation for exporting watertight meshes to STL and other CAD-friendly formats.

## Roadmap highlights

1. **Preview CLI** – invoke `impression preview path/to/model.py` to render and manipulate scenes (orbit, pan, zoom, strafe).
2. **Dynamic runtime** – let Python scripts define parameterized solids with runtime overrides (CLI flags, config files, or sockets).
3. **Declarative mode** – ingest declarative scene graphs (JSON/TOML) that map to the same geometry pipeline.
4. **Exporters** – transform the canonical geometry representation into STL, AMF, and STEP variants.

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
```

The PyVista window supports orbit, pan, and zoom out of the box. Files are watched by default, so saving changes triggers a hot reload in the same window (disable with `--no-watch` if you just want a single render).

## Export workflow

```bash
impression export examples/hello_cube.py --output artifacts/hello.stl --overwrite
```

The exporter loads the same `build()` entry point, merges all returned PyVista datasets, and writes a watertight STL in binary format (use `--ascii` for text output). Existing files are protected unless `--overwrite` is specified.
- Strategic roadmap: see `docs/project-plan.md` for primitives, CSG, CAD integration, and helper utilities (e.g., auto-rounding sharp faces).
