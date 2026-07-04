# Modeling - Hinges

Imported from `impression.modeling`:

```python
from impression.modeling import (
    HingeSurfaceAssembly,
    HingeSurfaceComponent,
    handoff_hinge_surface,
    make_traditional_hinge_leaf,
    make_traditional_hinge_pair,
    make_living_hinge,
    make_bistable_hinge,
)
```

## Surface-First Status

Hinges are now on an explicit surface-first migration path.

The legacy mesh builders still exist, but `backend="surface"` now exposes
deterministic surfaced hinge contracts instead of silently collapsing back to
mesh-first truth.

Today the surfaced path is split into two layers:

- direct surfaced leaf geometry for `make_traditional_hinge_leaf(..., backend="surface")`
- structured surfaced assemblies for:
  - `make_traditional_hinge_pair(..., backend="surface")`
  - `make_living_hinge(..., backend="surface")`
  - `make_bistable_hinge(..., backend="surface")`

Those structured results are returned as `HingeSurfaceAssembly`, which preserves:

- hinge family and assembly type
- deterministic assembly state such as `closed`, `opened`, `flat`, or `preloaded`
- surfaced component records (`HingeSurfaceComponent`) instead of baked mesh output
- consumer-facing color metadata
- stable identity from canonical hinge payload

## Public Handoff

Surfaced hinge results should move through the standard surface handoff boundary:

```python
from impression.modeling import (
    handoff_hinge_surface,
    make_traditional_hinge_pair,
)

hinge = make_traditional_hinge_pair(
    width=28.0,
    knuckle_count=5,
    opened_angle_deg=35.0,
    backend="surface",
)
collection = handoff_hinge_surface(hinge)
```

`handoff_hinge_surface(...)` returns a `SurfaceConsumerCollection`. That keeps
hinge outputs compatible with the standard preview/export path while surface
truth remains explicit up to tessellation time.

## Migration Posture

The hinge migration now has three explicit layers:

- surfaced primitive/body leaf generation
- structured hinge assembly truth
- legacy executable mesh generation

This keeps the migration honest:

- surfaced hinge APIs do not silently collapse back to mesh-first truth
- surfaced public outputs remain deterministic and inspectable
- legacy mesh execution remains available while surfaced hinge execution and
  richer assembly tooling continue maturing

## Traditional Hinges

`make_traditional_hinge_leaf(...)` creates one surfaced or mesh hinge leaf with
alternating knuckles and a pin bore region.

`make_traditional_hinge_pair(...)` creates a two-leaf assembly. On the surfaced
path it returns a `HingeSurfaceAssembly` with:

- two surfaced leaf components
- an optional pin component
- deterministic `opened` or `closed` state

## Living Hinges

`make_living_hinge(...)` creates a slit-pattern flex panel intended for
print-in-place bending.

On the surfaced path it returns a `HingeSurfaceAssembly` that preserves:

- panel dimensions
- hinge band width
- slit width and pitch
- bridge and edge margin
- deterministic slot count

## Bistable Hinges

`make_bistable_hinge(...)` creates a flexure-style over-center blank with
anchors, shuttle, and struts.

On the surfaced path it returns a `HingeSurfaceAssembly` that preserves:

- anchor dimensions
- shuttle dimensions
- ligament width
- preload offset
- deterministic `neutral` or `preloaded` state

## Reference Readiness

Before surfaced hinges can be considered fully render-ready, this doc requires
durable reference artifacts for at least these representative cases:

- `surfacebody/hinge_traditional_pair`
- `surfacebody/hinge_living_panel`
- `surfacebody/hinge_bistable_blank`

Each case should eventually gain:

- dirty and clean reference images
- dirty and clean reference STL files
- surfaced public examples or tests proving compatibility through the standard
  surface handoff boundary

## Examples

Existing example scenes still show the legacy executable lane:

- `docs/examples/hinges/traditional_hinge_example.py`
- `docs/examples/hinges/living_hinge_example.py`
- `docs/examples/hinges/bistable_hinge_example.py`
- `docs/examples/hinges/hinges_overview_example.py`

While those examples remain mesh-first for now, new public surfaced examples
should prefer `backend="surface"` plus `handoff_hinge_surface(...)`.
