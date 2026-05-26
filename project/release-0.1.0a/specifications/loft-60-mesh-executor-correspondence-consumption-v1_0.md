# Loft Spec 60: Mesh Executor Correspondence Consumption (v1.0)

Date: 2026-05-25
Status: Proposed
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `Mesh Executor Correspondence Consumption`

## Purpose

Update the mesh loft executor so it consumes validated sample correspondence
records instead of assuming equal sample indices represent the same semantic
point.

## Scope

Owns:

- Mesh executor validation for sample correspondence records.
- Mesh face emission from correspondence-aligned samples.
- Executor diagnostics when correspondence records are missing or inconsistent.

Does not own:

- Planner resampling and sample allocation.
- Surface patch construction.
- Mesh fallback behavior for surface execution.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - update `loft_execute_plan` and mesh
    executor helpers.
- Supporting modules/files:
  - `src/impression/modeling/mesh.py` or current mesh emission owner - read
    existing mesh construction primitives.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - executor consumption path.
- Tests:
  - `tests/test_loft_mesh_executor_correspondence.py` - stable
    correspondence, birth/death supports, and missing-record refusal.

## Chosen Defaults / Parameters

- Mesh executor requires a validated `ResampledLoopCorrespondence`.
- Equal-length sample arrays without sample records are invalid for new
  topology-correspondence plans.
- Executor trusts planner validation but performs defensive length and id
  consistency checks.
- Missing or mismatched correspondence records refuse execution; they do not
  fall back to index-only matching.

## Data Ownership

- Source of truth: planner owns correspondence truth; mesh executor owns emitted
  mesh geometry.
- Read ownership: mesh executor reads immutable resampled correspondence.
- Write ownership: mesh executor writes mesh vertices/faces and executor
  diagnostics only.
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
  - Existing mesh face emission path in `loft_execute_plan`.
- Current reuse readiness:
  - update existing mesh executor path.
- Extraction/wrapping needed:
  - wrap old index-based emission behind a function that accepts sample
    correspondence records.
- Additions to existing library/modules:
  - `emit_mesh_faces_from_sample_correspondence(...)`
  - `validate_mesh_executor_correspondence_input(...)`
- New reusable modules to expose:
  - none.
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

- Mesh emission remains O(s) per loop pair, where `s` is sample count.
- Defensive validation is O(s).
- No correspondence inference or resampling occurs inside the mesh executor.

## Error And State Behavior

- Missing sample records refuse with `missing_sample_correspondence`.
- Sample array length mismatch refuses with `sample_count_mismatch`.
- Record index mismatch refuses with `sample_record_index_mismatch`.
- Refusal happens before partial mesh emission.

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

- Mesh loft execution consumes explicit sample correspondence records.
- Index-only semantic matching is not used for topology-correspondence plans.
- Invalid correspondence input refuses with diagnostics before mesh output.

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
