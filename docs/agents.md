# Impression Agent Bootstrap Guide

This guide is for coding agents working in the Impression repo. The goals are:

1) Boot quickly into a working environment.
2) Know the existing features so you do not reimplement them or pull in new libraries.
3) Keep modeling internal and viewer-agnostic.

## Quick Bootstrap

```bash
git clone https://github.com/kellyjanderson/Impression.git
cd Impression
python3 -m venv .venv
source .venv/bin/activate
scripts/dev/install_impression.sh --local
```

Verify the CLI:

```bash
impression --help
```

## Core Rules (Non-Negotiable)

- Use Impression internal meshes and helpers. Do not return PyVista datasets from `build()`.
- PyVista is a view layer only. Modeling, boolean ops, and mesh generation stay in Impression.
- Reuse existing utilities before adding dependencies or new geometry code.

## What Impression Already Supports

Use these features instead of writing new implementations:

- **Primitives**: box, cylinder, cone, sphere, torus, prism, polyhedron
- **CSG**: union, difference, intersection (manifold3d backend)
- **2D Drawing**: Path2D, Profile2D, arcs, beziers, polylines, polygon/rect/circle/ngon helpers
- **Extrusions**: linear_extrude, rotate_extrude
- **Paths**: Path (polyline-based), Path3D (true curves)
- **Morph**: profile-to-profile morphing (same hole count)
- **Loft**: between multiple profiles, optional path
- **Transforms**: translate, rotate
- **Groups**: MeshGroup
- **Drafting**: lines, planes, arrows, dimensions (text labels currently disabled)
- **Preview/Export**: CLI preview and STL export

## Known Limitations

- **Text modeling** is disabled. Do not use `make_text` for now.
- If you see a missing feature, check [`docs/modeling/`](modeling/) and [`docs/examples/`](examples/) before adding code.

## Project Structure

- `src/impression/` - core package
- `src/impression/modeling/` - modeling API (use this first)
- `src/impression/preview.py` - PyVista viewer integration
- [`docs/`](./) - guides, tutorials, and examples
- [`docs/examples/`](examples/) - runnable example scripts (all define `build()`)
- `scripts/dev/` - setup and install helpers

## Common Commands

```bash
# Preview a model
impression preview docs/examples/primitives/box_example.py

# Export to STL
impression export docs/examples/csg/union_example.py --output dist/union.stl --overwrite

```

## Adding New Features

Before writing new code:

1) Check `docs/modeling/` and `docs/examples/` for existing helpers.
2) Search the codebase for similar utilities (`rg <term>`).
3) Prefer extending internal mesh APIs over adding external libraries.

If a new dependency is unavoidable, document it in [`docs/index.md`](index.md) and update
`requirements.txt` + `pyproject.toml` in the same change.
