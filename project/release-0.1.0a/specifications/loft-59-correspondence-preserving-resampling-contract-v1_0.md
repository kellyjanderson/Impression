# Loft Spec 59: Correspondence-Preserving Resampling Contract (v1.0)

Date: 2026-05-25
Status: Proposed
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `Correspondence-Preserving Resampling Contract`

## Purpose

Define the planner-side sample correspondence contract that turns semantic
point correspondence, protected landmarks, and lifecycle support samples into
executor-ready sample arrays.

## Scope

Owns:

- Protected sample allocation.
- Source/target sample arrays with parallel sample correspondence records.
- Explicit sample-count overflow refusal.
- Sample-to-track and sample-to-lifecycle associations.

Does not own:

- Mesh face emission.
- Surface patch construction.
- Rail priority, inference, or birth/death support resolution.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - add correspondence-preserving resampler
    and DTOs.
- Supporting modules/files:
  - `src/impression/modeling/topology.py` - read protected landmarks.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - reusable planner/executor handoff
    contract.
- Tests:
  - `tests/test_loft_correspondence_resampling.py` - protected landmarks,
    synthetic supports, explicit cap refusal, and sample-to-track mapping.

## Chosen Defaults / Parameters

- `sample_count="auto"` grows to include every protected landmark, required
  synthetic support sample, and minimum surviving-span samples.
- Explicit integer sample count is a hard cap.
- Too-low explicit sample count refuses with requested and minimum counts.
- Protected landmarks are exact samples whenever they are geometrically valid.
- Remaining samples are allocated by span arc length after protected points are
  placed.

## Data Ownership

- Source of truth: loop correspondence map owns semantic truth; resampled loop
  correspondence owns executable samples.
- Read ownership: mesh and surface executors read resampled correspondence.
- Write ownership: resampler writes sample arrays and sample correspondence
  records; executors must not modify them.
- Derived/cache data: sample arrays derive from loop geometry, protected
  landmarks, lifecycle events, and sample policy.
- Privacy/logging constraints: diagnostics may log sample counts, ids, and
  refusal reasons only.

## Dependencies And Routes

- Domain/service dependencies:
  - loop correspondence map
  - point lifecycle events
  - existing `resample_loop`
  - topology protected landmarks
- Database dependencies:
  - none.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; deterministic synchronous planning step.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Existing `resample_loop` as the low-level interpolation primitive where it
    can honor protected parameters.
  - Loop pairing records and lifecycle event records.
- Current reuse readiness:
  - add to loft planner/executor boundary.
- Extraction/wrapping needed:
  - wrap `resample_loop` so semantic protected samples are inserted before
    remaining arc-length samples.
- Additions to existing library/modules:
  - `ResampledLoopCorrespondence`
  - `SampleCorrespondenceRecord`
  - `resample_loop_correspondence(...)`
  - `validate_sample_correspondence(...)`
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `ResampledLoopCorrespondence` - fields: `source_samples`,
    `target_samples`, `sample_records`, `protected_indices`, `diagnostics`.
  - `SampleCorrespondenceRecord` - fields: `index`, `source_point_ref`,
    `target_point_ref`, `track_id`, `lifecycle_event_id`, `source_parameter`,
    `target_parameter`, `protected`.
- Functions/methods:
  - `resample_loop_correspondence(loop_pair, *, sample_count) ->
    ResampledLoopCorrespondence`
  - `validate_sample_correspondence(resampled) -> None`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Resampling is O(n + s), where `n` is protected/lifecycle samples and `s` is
  requested output samples.
- Explicit sample-count validation runs before allocation work.
- No global correspondence inference runs in this spec.

## Error And State Behavior

- Too-low explicit sample count refuses with `sample_count_too_low`.
- Missing lifecycle support refs refuse with `missing_lifecycle_support`.
- Non-monotonic protected parameters refuse with `protected_sample_order_conflict`.
- Executor inputs are not produced after refusal.

## Test Strategy

- Unit tests:
  - protected landmarks survive as exact sample records.
  - synthetic birth/death supports produce paired samples.
  - explicit sample-count cap refusal reports minimum count.
  - sample records align with source and target arrays.
  - unprotected spans receive length-proportional samples.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Production-data rule:
  - Tests must not require the user's production database.

## Acceptance Criteria

- Equal sample arrays no longer imply semantic equality by index alone.
- Every emitted sample has a correspondence record.
- Protected landmarks and lifecycle supports are preserved or validation
  refuses before execution.

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
