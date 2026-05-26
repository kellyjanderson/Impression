# Loft Spec 61: Surface Executor Correspondence Consumption (v1.0)

Date: 2026-05-25
Status: Proposed
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `Surface Executor Correspondence Consumption`

## Purpose

Update the surface loft executor so ruled surface patch construction consumes
validated sample correspondence records and refuses missing correspondence
instead of falling back to index-only matching.

## Scope

Owns:

- Surface executor validation for sample correspondence records.
- `RuledSurfacePatch` construction from correspondence-aligned samples.
- Diagnostics for missing, inconsistent, or semantically invalid surface
  correspondence input.

Does not own:

- Planner resampling and sample allocation.
- Mesh executor behavior.
- Surface body patch family implementation outside loft handoff.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - update `_loft_execute_plan_surface`
    and surface executor helpers.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` or current surface patch owner - use
    `RuledSurfacePatch`.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - surface executor consumption path.
- Tests:
  - `tests/test_loft_surface_executor_correspondence.py` - ruled patch
    boundaries, lifecycle supports, and missing-record refusal.

## Chosen Defaults / Parameters

- Surface executor requires validated `ResampledLoopCorrespondence`.
- Surface executor refuses missing sample records and never silently falls back
  to mesh-first or index-only correspondence.
- Boundary construction preserves protected sample order exactly.
- Surface patch diagnostics include track and lifecycle references when a
  boundary cannot be emitted.

## Data Ownership

- Source of truth: planner owns correspondence truth; surface executor owns
  emitted surface patches.
- Read ownership: surface executor reads immutable resampled correspondence.
- Write ownership: surface executor writes surface body/shell/patch output and
  executor diagnostics only.
- Derived/cache data: surface patch boundaries derive from sample arrays and
  records.
- Privacy/logging constraints: diagnostics may log sample indices, track ids,
  lifecycle ids, and refusal reasons.

## Dependencies And Routes

- Domain/service dependencies:
  - `ResampledLoopCorrespondence`
  - `SampleCorrespondenceRecord`
  - `RuledSurfacePatch`
  - surface body/shell construction helpers
- Database dependencies:
  - none.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; use current loft execution path.

## Reuse And Extraction Plan

- Existing code to reuse:
  - `RuledSurfacePatch`
  - Existing surface loft executor route.
- Current reuse readiness:
  - update existing surface executor path.
- Extraction/wrapping needed:
  - wrap surface boundary construction to accept sample correspondence records.
- Additions to existing library/modules:
  - `emit_surface_patches_from_sample_correspondence(...)`
  - `validate_surface_executor_correspondence_input(...)`
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `SurfaceSampleEmissionDiagnostic` - fields: `loop_pair_id`,
    `sample_index`, `track_id`, `lifecycle_event_id`, `reason`.
- Functions/methods:
  - `validate_surface_executor_correspondence_input(resampled) -> None`
  - `emit_surface_patches_from_sample_correspondence(resampled,
    surface_builder) -> SurfaceBody`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Surface boundary emission is O(s) per loop pair.
- Validation is O(s).
- Surface executor must not resample, infer, or repair correspondence.

## Error And State Behavior

- Missing sample records refuse with `missing_sample_correspondence`.
- Surface boundary sample mismatch refuses with `surface_boundary_sample_mismatch`.
- Missing lifecycle support referenced by a sample refuses with
  `missing_lifecycle_support`.
- Refusal happens before partial surface body output.

## Test Strategy

- Unit tests:
  - ruled patch boundaries follow sample correspondence records.
  - protected landmarks remain patch boundary samples.
  - birth/death supports are preserved in surface output.
  - missing records refuse.
  - surface executor does not call mesh fallback for correspondence failure.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Production-data rule:
  - Tests must not require the user's production database.

## Acceptance Criteria

- Surface loft execution consumes explicit sample correspondence records.
- Missing correspondence refuses before output, with no mesh fallback.
- Ruled surface boundaries preserve protected landmarks and lifecycle supports.

## Readiness Checklist

- [x] Implementation owner/module is named.
- [x] Existing code reuse/extraction decision is explicit.
- [x] Existing library/module additions or new reusable module boundaries are named, or marked not applicable.
- [x] UI fields/elements are listed, or marked not applicable.
- [x] Chosen defaults are explicit.
- [x] Data source of truth and write owner are explicit.
- [x] GUI/concurrency route is explicit, or marked not applicable.
- [x] Performance bounds are explicit, or marked not applicable.
- [x] Privacy/logging constraints are explicit, or marked not applicable.
- [x] Test strategy does not depend on production data.
- [x] Acceptance criteria are testable.
