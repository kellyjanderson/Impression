# Modeling - Threading

Imported from `impression.modeling`:

```python
from impression.modeling import (
    ThreadSpec,
    ThreadFitPreset,
    ThreadSurfaceAssembly,
    ThreadSurfaceRepresentation,
    MeshQuality,
    lookup_standard_thread,
    apply_fit,
    paired_fit,
    validate_thread,
    estimate_mesh_cost,
    make_external_thread,
    make_internal_thread,
    make_hex_nut,
    make_round_nut,
    make_runout_relief,
    make_tapped_hole_cutter,
    prepare_surface_thread_representation,
    make_threaded_rod,
    clear_thread_cache,
)
```

## Surface-First Status

Threading is in an active surface-first migration. The legacy mesh generators still exist, but external and internal threads now also expose a deterministic surfaced preparation path:

```python
from impression.modeling import lookup_standard_thread, make_external_thread

spec = lookup_standard_thread("metric", "M6x1", length=12.0)
thread = make_external_thread(spec, backend="surface")
```

That surfaced result is a `ThreadSurfaceRepresentation`, not a tessellated mesh. It captures:

- normalized axis origin, direction, and orthonormal basis
- major diameter, minor diameter, pitch, and resolved thread depth
- starts, handedness, taper, runout, and end-treatment metadata
- deterministic pitch schedule and sampled thread-profile shape
- a stable identity derived from canonical thread geometry

This keeps thread truth explicit before any future surfaced thread executor turns it into patches or tessellation.

Surface convenience builders are also starting to migrate. Today, `backend="surface"` on:

- `make_threaded_rod(...)`
- `make_tapped_hole_cutter(...)`
- `make_hex_nut(...)`
- `make_round_nut(...)`
- `make_runout_relief(...)`

returns a `ThreadSurfaceAssembly`. That assembly records surfaced primitive and thread operands plus the intended composition (`standalone`, `union`, or `difference`) without collapsing back to mesh-first truth.

Fit presets still change canonical geometry on the surfaced path, because they change the actual compensated thread dimensions. Mesh-quality controls do not: they are currently ignored by surfaced thread preparation and only matter on the legacy mesh generator path.

## Migration Posture

The threading API now has three explicit surfaced layers:

- canonical thread representation via `ThreadSurfaceRepresentation`
- convenience helper composition via `ThreadSurfaceAssembly`
- legacy executable mesh generation for the existing print/export lane

This keeps the migration boundary honest:

- surfaced thread truth is structured and deterministic
- surfaced convenience builders do not silently collapse back to mesh-first assembly
- legacy mesh execution remains available while surfaced execution is still incomplete

## Reference Readiness

Before surfaced threading can be considered render-ready, this doc requires durable reference artifacts for at least these representative cases:

- `surfacebody/thread_external_metric_m6`
- `surfacebody/thread_hex_nut_m6`
- `surfacebody/thread_runout_relief_metric`

Each of those cases should eventually gain:

- dirty and clean reference images
- dirty and clean reference STL files
- surfaced public examples or tests proving consumer compatibility through the standard handoff boundary

## Core Flow

1. Create a base spec (`lookup_standard_thread(...)` or direct `ThreadSpec(...)`).
2. Apply fit compensation (`paired_fit(...)` or `apply_fit(...)`).
3. Generate geometry (`make_external_thread`, `make_internal_thread`, nuts/cutters).

## Standards and Profiles

`lookup_standard_thread(...)` supports:

- Metric / ISO (`"metric"`, `"iso"`, `"m"`)
- Unified (`"unified"`, `"unc"`, `"unf"`, `"unef"`)
- ACME (`"acme"`)
- Trapezoidal (`"trapezoidal"`, `"tr"`)
- Pipe-like tapered (`"pipe"`, `"npt-like"`)

Supported profile names include:
`iso`, `unified`, `whitworth`, `trapezoidal`, `acme`, `square`, `buttress`, `pipe`, `rounded`, and `custom`.

## Fit Presets

Built-in fit presets:

- `fdm_default`
- `fdm_tight`
- `fdm_loose`
- `sla_tight`
- `cnc_nominal`

Use `paired_fit(spec, "fdm_default")` to generate matched male/female specs.

## Validation and Cost

- `validate_thread(spec, quality=...)` checks geometry constraints.
- `estimate_mesh_cost(spec, quality=...)` predicts vertex/face counts before generation.

## Part Generators

- `make_threaded_rod(...)`
- `make_tapped_hole_cutter(...)`
- `make_hex_nut(...)`
- `make_round_nut(...)`
- `make_runout_relief(...)`

## Examples

- Basic threads: `docs/examples/threading/threading_basic_example.py`
- Threaded nut + rod: `docs/examples/threading/threading_nut_example.py`
- Print test plate (paired families): `docs/examples/threading/thread_mated_pairs_test_plate.py`

Preview:

```bash
impression preview docs/examples/threading/threading_basic_example.py
```
