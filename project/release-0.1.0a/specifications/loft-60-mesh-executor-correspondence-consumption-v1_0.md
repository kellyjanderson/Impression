# Loft Spec 60: Legacy Mesh Debug Correspondence Consumption (v1.1)

Date: 2026-05-25
Status: Retired as canonical modeled execution; retained only as legacy/debug
compatibility guidance.
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `Legacy Mesh Debug Correspondence Consumption`

Revision note: Surface Spec 167 revised this document after the mesh execution
tessellation-boundary audit. `LoftPlan -> SurfaceBody` is the canonical modeled
executor contract. Mesh face emission from loft correspondence is not canonical
authored geometry; it is allowed only as an explicit legacy compatibility,
debug, or tessellation-boundary route.

## Purpose

Document the retired mesh loft executor posture so any remaining explicit
mesh/debug path consumes validated sample correspondence records instead of
assuming equal sample indices represent the same semantic point.

## Scope

Owns:

- Legacy/debug mesh executor validation for sample correspondence records.
- Explicit mesh/debug face emission from correspondence-aligned samples.
- Diagnostics when correspondence records are missing or inconsistent.
- The migration note that this document is not canonical modeled execution.

Does not own:

- Planner resampling and sample allocation.
- Surface patch construction.
- Mesh fallback behavior for surface execution.
- Canonical loft execution; Surface Executor Correspondence Consumption owns
  the modeled `LoftPlan -> SurfaceBody` contract.
- Tessellation-boundary mesh generation; Surface Spec 168 owns relocation of
  remaining mesh emission out of canonical loft execution.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - legacy/debug mesh helper references
    only until Surface Spec 168 relocates or retires them.
- Supporting modules/files:
  - `src/impression/modeling/mesh.py` or current mesh emission owner - read
    existing mesh construction primitives.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - legacy/debug consumption path only.
- Tests:
  - `tests/test_loft_mesh_executor_correspondence.py` - stable
    correspondence, birth/death supports, and missing-record refusal.

## Chosen Defaults / Parameters

- Any retained explicit mesh/debug route requires a validated
  `ResampledLoopCorrespondence`.
- Equal-length sample arrays without sample records are invalid for new
  topology-correspondence plans.
- Legacy/debug emission trusts planner validation but performs defensive length and id
  consistency checks.
- Missing or mismatched correspondence records refuse execution; they do not
  fall back to index-only matching.
- Public loft APIs must not silently fall back from surface execution to this
  mesh/debug path.

## Data Ownership

- Source of truth: planner owns correspondence truth; surface executor owns
  canonical modeled loft output.
- Read ownership: retained mesh/debug emission reads immutable resampled
  correspondence.
- Write ownership: retained mesh/debug emission writes mesh vertices/faces and
  diagnostics only; those meshes are not canonical authored model state.
- Derived/cache data: mesh geometry derives from sample arrays and records.
- Privacy/logging constraints: diagnostics may log sample indices, track ids,
  lifecycle event ids, and refusal reason.

## Dependencies And Routes

- Domain/service dependencies:
  - `ResampledLoopCorrespondence`
  - `SampleCorrespondenceRecord`
  - existing mesh face emission helpers
- Database dependencies:
  - none.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; existing loft execution route remains synchronous unless
    broader executor policy changes.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Existing mesh face emission path only as legacy/debug compatibility until
    Surface Spec 168 relocates or retires it.
- Current reuse readiness:
  - retired from canonical execution; only explicit legacy/debug routes may
    reuse it.
- Extraction/wrapping needed:
  - wrap or move old index-based emission behind explicit debug/tessellation
    boundary functions that accept sample correspondence records.
- Additions to existing library/modules:
  - `emit_mesh_faces_from_sample_correspondence(...)`
  - `validate_mesh_executor_correspondence_input(...)`
- New reusable modules to expose:
  - none from this retired spec; Surface Spec 168 owns any tessellation/debug
    relocation.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `MeshSampleEmissionDiagnostic` - fields: `loop_pair_id`, `sample_index`,
    `track_id`, `reason`.
- Functions/methods:
  - `validate_mesh_executor_correspondence_input(resampled) -> None`
  - `emit_mesh_faces_from_sample_correspondence(resampled, mesh_builder) -> None`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Retained explicit mesh/debug emission remains O(s) per loop pair, where `s`
  is sample count.
- Defensive validation is O(s).
- No correspondence inference or resampling occurs inside the mesh executor.

## Error And State Behavior

- Missing sample records refuse with `missing_sample_correspondence`.
- Sample array length mismatch refuses with `sample_count_mismatch`.
- Record index mismatch refuses with `sample_record_index_mismatch`.
- Refusal happens before partial mesh emission.
- Any surface-executor failure must surface as a surface diagnostic, not a
  successful mesh result.

## Test Strategy

- Unit tests:
  - mesh face indices follow sample correspondence records.
  - birth support samples emit stable faces.
  - death support samples emit stable faces.
  - missing sample records refuse.
  - mismatched record indices refuse before partial output.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Production-data rule:
  - Tests must not require the user's production database.

## Acceptance Criteria

- This document is visibly retired as canonical modeled execution.
- Retained explicit mesh/debug loft emission consumes explicit sample
  correspondence records.
- Index-only semantic matching is not used for topology-correspondence plans.
- Invalid correspondence input refuses with diagnostics before mesh output.
- Canonical loft execution is documented as `LoftPlan -> SurfaceBody`, with
  mesh output restricted to legacy/debug/tessellation-boundary routes.

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
