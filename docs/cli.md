# Impression CLI Reference

## NAME

`impression` - preview, export, and explore parametric models built with Impression.

## SYNOPSIS

```
impression preview [OPTIONS] MODEL
impression export [OPTIONS] MODEL
impression --get-docs [OPTIONS]
impression studio [OPTIONS]
```

## COMMANDS

### `preview`

Render a model and optionally watch it for changes.

**Arguments**

- `MODEL` - path to a Python module containing a `build()` function.

**Requirements**

- `build()` must return internal meshes (`Mesh`, `MeshGroup`, `Polyline`, `Path2D`, `Profile2D`, or lists of them).
- PyVista is used as a viewer only; do not return PyVista datasets directly.

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

- `MODEL` - path to a `build()` script.

**Options**

- `-o, --output PATH` (default: `model.stl`): destination STL path.
- `--overwrite / --no-overwrite` (default: `--no-overwrite`): guard existing files.
- `--ascii / --binary` (default: binary): choose STL encoding.

### `studio`

Launch the Impression Studio desktop app (examples + docs + live preview).

**Options**

- `-w, --workspace PATH` (default: current directory): workspace root containing `docs/` and `docs/examples/`.

## `--get-docs`

Download the documentation bundle without cloning the full repository.

**Options**

- `--get-docs / --getDocs`: trigger the download and exit.
- `--docs-dest PATH`: destination folder (default: `./impression-docs`).
- `--docs-repo URL`: GitHub repo URL (default: `https://github.com/kellyjanderson/Impression`).
- `--docs-ref REF`: git ref to download from (default: `main`).
- `--docs-clean`: delete destination before downloading.

## UNITS

Impression reads `~/.impression/impression.cfg` (JSON) to determine the default units.
Valid values are `millimeters` (default), `meters`, and `inches` (case-insensitive). These
units are echoed in preview axes labels and export summaries.

## WORKFLOW

1. Write a module that defines `build()` and returns internal meshes.
2. Preview: `impression preview path/to/module.py`
3. Export: `impression export path/to/module.py --output artifacts/model.stl`
4. Studio: `impression studio`
5. Run screenshot regression tests: `scripts/run_preview_tests.py`
6. Validate STL exports + watertightness: `scripts/run_stl_tests.py`

See `docs/examples/` for ready-to-run modules covering primitives, drafting helpers, text, and more.
