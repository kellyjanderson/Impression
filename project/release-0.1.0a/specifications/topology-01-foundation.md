> Status: Deprecated historical spec.
> Active work now lives in the surface-first specification tree and the current
> implementation. Retained for project history only.

# Topology Spec 01: Foundation Layer

## Goal

Introduce a shared planar-topology module that becomes the canonical home for loop/region/section operations used by loft, extrude, text assembly, and 2D ops.

## Scope

- Add `src/impression/modeling/topology.py`.
- Introduce first-class topology primitives:
  - `Loop`
  - `Region`
  - `Section`
- Move foundational planar helpers into `topology.py`:
  - loop sampling
  - signed area / winding normalization
  - resampling
  - point-in-polygon
  - loop classification
  - triangulation prep (`earcut`)
- Preserve compatibility through `legacy_planar_adapter.py` adapters.

## Non-Goals (for this phase)

- Public API breakage for `PlanarShape2D`.
- Full transition planner (hole birth/death, split/merge policy).
- Replacing all callers in one patch.

## Deliverables

1. New `topology.py` module with docstrings and invariants.
2. `legacy_planar_adapter.py` converted to compatibility adapters delegating to topology.
3. At least one caller migrated to use topology directly.

## Completion Criteria

- Existing tests pass without behavior regressions.
- Loft/extrude/text/ops can continue functioning through adapters.
- There is one obvious file to discover planar-topology capabilities.
