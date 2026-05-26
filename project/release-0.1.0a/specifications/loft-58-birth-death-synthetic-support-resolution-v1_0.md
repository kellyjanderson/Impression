# Loft Spec 58: Birth/Death Synthetic Support Resolution (v1.0)

Date: 2026-05-25
Status: Proposed
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `Birth/Death Synthetic Support Resolution`

## Purpose

Define how unmatched authored points are localized to stable parent spans and
converted into explicit point birth/death lifecycle events with synthetic
support samples.

## Scope

Owns:

- Parent-span lookup for unmatched points.
- Birth and death support projection.
- Creation of point lifecycle events and synthetic support references.
- Refusal behavior for ambiguous or collapsed point lifecycle transitions.

Does not own:

- Lifecycle DTO definitions.
- Sample allocation across all spans.
- Executor consumption of samples.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - add birth/death resolver near loop
    correspondence planning.
- Supporting modules/files:
  - `src/impression/modeling/topology.py` - read topology points/landmarks.
  - `src/impression/modeling/loft.py` tolerance policy records - add or consume
    `collapse_degeneracy.min_point_correspondence_span`.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - reusable lifecycle resolver.
- Tests:
  - `tests/test_loft_point_birth_death_resolution.py` - rectangle-to-rounded-L,
    birth, death, collapse refusal, and conflict refusal fixtures.

## Chosen Defaults / Parameters

- Birth/death requires stable neighboring tracks or an explicit parent span.
- Parent span projection uses normalized span parameter in `[0.0, 1.0]`.
- Collapse refusal uses `collapse_degeneracy.min_point_correspondence_span` from
  loft tolerance policy.
- Multiple equally plausible parent spans refuse and request rails.
- Events conflicting with explicit correspondence ids refuse as invalid input.

## Data Ownership

- Source of truth: loop correspondence map owns lifecycle events and synthetic
  support references.
- Read ownership: resampling and executors read lifecycle output.
- Write ownership: birth/death resolver writes derived events and support refs;
  it does not mutate source topology records.
- Derived/cache data: support coordinates derive from source/target span
  geometry and can be recomputed.
- Privacy/logging constraints: diagnostics may log point ids, span refs,
  parameters, and refusal reasons only.

## Dependencies And Routes

- Domain/service dependencies:
  - `PointLifecycleEvent`
  - topology landmarks and point tracks
  - loop correspondence map
  - loft tolerance policy
- Database dependencies:
  - none.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; deterministic planner step.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Existing loop pairing diagnostics.
  - Existing point projection or interpolation helpers where present.
- Current reuse readiness:
  - add to existing loft planner module.
- Extraction/wrapping needed:
  - wrap existing tolerance policy with `min_point_correspondence_span` if the
    field is absent.
- Additions to existing library/modules:
  - `resolve_point_birth_death_events(...)`
  - `locate_parent_span(...)`
  - `insert_synthetic_support_reference(...)`
  - `PointLifecycleRefusalDiagnostic`
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `ParentSpanMatch` - fields: `source_track_ref`, `target_track_ref`,
    `span_parameter`, `confidence`, `source`.
  - `PointLifecycleRefusalDiagnostic` - fields: `reason`, `point_ref`,
    `candidate_spans`, `required_rail_hint`.
- Functions/methods:
  - `resolve_point_birth_death_events(loop_pair, rail_result,
    tolerance_policy) -> PointLifecycleResolution`
  - `locate_parent_span(point_ref, stable_tracks) -> ParentSpanMatch`
  - `project_point_to_span(point, span) -> float`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Parent-span lookup is O(u * s), where `u` is unmatched points and `s` is
  stable spans in the loop pair.
- Projection is local to a candidate span and does not run global optimization.
- Candidate span diagnostics are bounded to plausible local spans.

## Error And State Behavior

- Missing stable neighboring tracks refuses with `missing_parent_span`.
- Multiple equally plausible spans refuse with `ambiguous_parent_span`.
- Span below collapse tolerance refuses with `collapsed_parent_span`.
- Order inversion refuses with `birth_death_order_inversion`.
- Explicit correspondence conflict refuses with `explicit_correspondence_conflict`.

## Test Strategy

- Unit tests:
  - rectangle to rounded-L creates point birth events on correct parent spans.
  - source-only point creates point death event and target support.
  - ambiguous parent span refuses.
  - collapsed span refuses through tolerance policy.
  - explicit correspondence conflict refuses.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Production-data rule:
  - Tests must not require the user's production database.

## Acceptance Criteria

- Born and dying points become explicit lifecycle events with support refs.
- Ambiguous point lifecycle changes refuse instead of shifting correspondence.
- The rectangle-to-rounded-L fixture has stable parent-span diagnostics.

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
