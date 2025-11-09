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

The CLI scaffolding currently stubs preview behavior; future iterations will integrate real-time renderers and file watchers for live editing.
