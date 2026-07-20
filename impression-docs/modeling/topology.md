# Modeling - Topology

`impression.modeling.topology` is the shared planar-topology layer used by loft, extrusion, text profile assembly, and 2D operations.

Authoring boundary:

- `drawing2d` owns authored curves and profile construction ergonomics.
- `topology` owns planar loop/region/section algorithms and kernel classification/triangulation behavior.

## Core Primitives

- `Loop`: closed 2D loop wrapper with winding/area/resampling helpers
- `Region`: one outer loop plus zero or more hole loops
- `Section`: collection of one or more disconnected regions

Imported from `impression.modeling`:

```python
from impression.modeling import (
    Loop,
    Region,
    Section,
    as_section,
    as_sections,
    regions_from_paths,
    sections_from_paths,
)
```

## Core Helpers

- `signed_area(points)`
- `ensure_winding(points, clockwise=...)`
- `as_section(shape, ...)`
- `as_sections(shapes, ...)`
- `resample_loop(points, count)`
- `anchor_loop(loop)`
- `point_in_polygon(point, polygon)`
- `regions_from_paths(paths)`
- `sections_from_paths(paths)`
- `profile_loops(profile, ...)`
- `triangulate_loops(loops)`
- `triangulate_profile(profile, ...)`
- `classify_loops(loops, expected_holes=...)`
- `inset_profile_loops(profile, inset, join_type=..., hole_count=...)`
- `minimum_cost_loop_assignment(source, target, area_weight=...)`
- `minimum_cost_subset_assignment(source, target, area_weight=...)`
- `stable_loop_transition(source_loop, target_loop)`
- `split_merge_ambiguous(loop_a, loop_b)`

## Compatibility Notes

`as_section(shape)` accepts topology-native objects (`Section`, `Region`) and
closed drawing shapes (`Path2D`, `PlanarShape2D`).

## Transition Checkpoints

1. New kernel features must accept topology-native primitives first.
2. Existing user-facing `PlanarShape2D` flows remain supported through `as_section`.
