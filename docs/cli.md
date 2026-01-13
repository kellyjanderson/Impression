# Impression CLI Reference

## NAME

`impression` — preview and export parametric models built with Impression.

## SYNOPSIS

```
impression preview [OPTIONS] MODEL
impression export [OPTIONS] MODEL
```

## COMMANDS

### `preview`

Render a model and optionally watch it for changes.

**Arguments**

- `MODEL` — path to a Python module containing a `build()` function.

**Options**

- `--watch / --no-watch` (default: `--watch`): enable file watching for hot reload.
- `--target-fps INTEGER` (default: `60`): polling frequency for watch mode.
- `--screenshot PATH`: render off-screen and save a PNG instead of opening a window.
- `--show-edges / --hide-edges` (default: `--hide-edges`): draw every triangle edge (useful for mesh debugging).
- `--face-edges / --no-face-edges` (default: `--no-face-edges`): overlay feature edges for crisp object outlines without the full triangle soup.
- Camera defaults: +Z up, +X right, +Y toward the camera. Resetting the scene (reload/first render) keeps this orientation consistent.

Environment tip: set `PYVISTA_OFF_SCREEN=true` when running in a headless environment (e.g., CI).

### `export`

Generate an STL from a model.

**Arguments**

- `MODEL` — path to a `build()` script.

**Options**

- `-o, --output PATH` (default: `model.stl`): destination STL path.
- `--overwrite / --no-overwrite` (default: `--no-overwrite`): guard existing files.
- `--ascii / --binary` (default: binary): choose STL encoding.

## WORKFLOW

1. Write a module that defines `build()` and returns internal meshes.
2. Preview: `impression preview path/to/module.py`
3. Export: `impression export path/to/module.py --output artifacts/model.stl`
4. Run screenshot regression tests: `scripts/run_preview_tests.py`
5. Validate STL exports + watertightness: `scripts/run_stl_tests.py`

See `docs/examples/` for ready-to-run modules covering primitives, drafting helpers, text, and more.
