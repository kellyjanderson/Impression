# Impression CLI Reference

## NAME

`impression` - preview, export, and explore parametric models built with Impression.

## SYNOPSIS

```
impression preview [OPTIONS] MODEL
impression export [OPTIONS] MODEL
impression --get-docs [OPTIONS]
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

**Hot reload controls**

- Press `r` in the preview window to force a reload.
- Press `v` to reset the camera.
- Send `SIGUSR1` to the preview process to force a reload (macOS/Linux):

```bash
kill -USR1 <pid>
```

**Switching preview targets (no restart)**

Provide a control file to switch models. By default, Impression writes a control file named
`.impression-preview` in the folder where you launched the preview and includes the preview
process ID in the first line. Updating this file automatically reloads the preview; `SIGUSR1`
is an optional manual trigger.

If a live preview already exists in that folder, running `impression preview other.py` will
update the control file and exit (reusing the existing window). To force a second window, use
`--force-window`.

```bash
impression preview docs/examples/loft/saddle_example.py
```

Write the new model path into the control file:

```bash
echo "/absolute/path/to/other_model.py" > ./.impression-preview
```

The preview will reload automatically; you can also send `SIGUSR1` to force it.

**Mesh analysis on demand**

Both `preview` and `export` print mesh analysis (degenerate faces, boundary edges, watertightness).
If you just made a change, run either command to see whether the model fails.

### `export`

Generate an STL from a model.

**Arguments**

- `MODEL` - path to a `build()` script.

**Options**

- `-o, --output PATH` (default: `model.stl`): destination STL path.
- `--overwrite / --no-overwrite` (default: `--no-overwrite`): guard existing files.
- `--ascii / --binary` (default: binary): choose STL encoding.

## `--get-docs`

Download the documentation bundle without cloning the full repository.

**Options**

- `--get-docs / --getDocs`: trigger the download and exit.
- `--docs-dest PATH`: destination folder (default: `./impression-docs`).
- `--docs-repo URL`: GitHub repo URL (default: `https://github.com/kellyjanderson/Impression`).
- `--docs-ref REF`: release/tag/branch ref (default: installed release tag).
- `--docs-clean`: delete destination before downloading.

By default, `--get-docs` fetches `impression-docs-<installed-tag>.zip` from the
matching GitHub release, then falls back to that same ref's source archive if
the docs asset is unavailable.

## UNITS

Impression reads `~/.impression/impression.cfg` (JSON) to determine the default units.
Valid values are `millimeters` (default), `meters`, and `inches` (case-insensitive). These
units are echoed in preview axes labels and export summaries.

## WORKFLOW

1. Write a module that defines `build()` and returns internal meshes.
2. Preview: `impression preview path/to/module.py`
3. Export: `impression export path/to/module.py --output artifacts/model.stl`
4. Run screenshot regression tests: `scripts/run_preview_tests.py`
5. Validate STL exports + watertightness: `scripts/run_stl_tests.py`

See [`docs/examples/`](examples/) for ready-to-run modules covering primitives, drafting helpers, text, and more.
