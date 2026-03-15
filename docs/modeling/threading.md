# Modeling - Threading

Imported from `impression.modeling`:

```python
from impression.modeling import (
    ThreadSpec,
    ThreadFitPreset,
    MeshQuality,
    lookup_standard_thread,
    apply_fit,
    paired_fit,
    validate_thread,
    estimate_mesh_cost,
    make_external_thread,
    make_internal_thread,
    make_threaded_rod,
    make_tapped_hole_cutter,
    make_hex_nut,
    make_round_nut,
    make_runout_relief,
    clear_thread_cache,
)
```

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
