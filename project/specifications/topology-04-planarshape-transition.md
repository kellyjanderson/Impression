> Status: Deprecated historical spec.
> Active work now lives in the surface-first specification tree and the current
> implementation. Retained for project history only.

# Topology Spec 04: PlanarShape2D Transition Plan

## Goal

Move `PlanarShape2D` from kernel-center to compatibility adapter while preserving developer ergonomics.

## Current Problem

`PlanarShape2D` is used as a cross-feature topology carrier, but it was originally scoped for loft-profile workflows.
This causes topology logic to leak across modules and blocks topology-native APIs.

## Target State

- Kernel operations consume `Loop` / `Region` / `Section`.
- `PlanarShape2D` remains available for users during migration.
- Conversion is explicit and deterministic:
  - `as_section(profile) -> Section`
  - `section_from_paths(section) -> list[PlanarShape2D]`

## Migration Rules

1. No new kernel logic should be introduced directly on `PlanarShape2D`.
2. New features accept topology-native primitives first, with `PlanarShape2D` adapters.
3. Warnings are introduced before deprecating behavior.
4. Remove `PlanarShape2D` internals only after all kernel callers migrate.

## Deliverables

- Documented compatibility contract for `PlanarShape2D`.
- Conversion helpers in topology surface.
- Deprecation schedule and checkpoints.

## Completion Criteria

- `PlanarShape2D` is no longer required in internal loft/extrude/text/ops codepaths.
- Public UX remains stable through adapters.
