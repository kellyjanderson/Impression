# Loft Spec 57: Point Lifecycle Event Records (v1.0)

Date: 2026-05-25
Status: Proposed
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `Point Lifecycle Event Records`

## Purpose

Define durable point lifecycle records so point birth, point death, and
synthetic support samples are visible planner facts rather than hidden
resampling side effects.

## Scope

Owns:

- Point lifecycle state enum.
- Point birth/death event record.
- Synthetic support reference record.
- Validation and serialization-ready diagnostic shape for lifecycle events.

Does not own:

- Algorithms that locate parent spans or insert support samples.
- Resampling sample allocation.
- Mesh or surface executor consumption.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - add lifecycle DTOs near loop
    correspondence records.
- Supporting modules/files:
  - `src/impression/modeling/topology.py` - topology point refs used by events.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - reusable planner lifecycle records.
- Tests:
  - `tests/test_loft_point_lifecycle_records.py` - construction, validation,
    state values, provenance, and diagnostics.

## Chosen Defaults / Parameters

- Lifecycle states are `present`, `birth`, `death`,
  `synthetic_birth_support`, `synthetic_death_support`, and `inferred`.
- Event types are `point_birth` and `point_death`.
- Event source is one of `authored`, `generated`, or `inferred`.
- Events require station interval, loop reference, point/support reference,
  parent span reference, interpolation policy, and diagnostic provenance.

## Data Ownership

- Source of truth: loop correspondence map owns lifecycle events.
- Read ownership: support resolver, resampling contract, executors, and
  diagnostics may read lifecycle events.
- Write ownership: lifecycle resolver creates events; downstream stages must not
  mutate event identity.
- Derived/cache data: serialized diagnostics and sample associations derive from
  lifecycle records.
- Privacy/logging constraints: diagnostics may log ids, lifecycle state, and
  numeric span parameters only.

## Dependencies And Routes

- Domain/service dependencies:
  - loop correspondence map
  - topology point and landmark refs
- Database dependencies:
  - none.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; planner records are synchronous.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Existing loop pairing and diagnostics record conventions.
- Current reuse readiness:
  - add to existing loft planner module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `PointLifecycleState`
  - `PointLifecycleEvent`
  - `SyntheticSupportReference`
  - `validate_point_lifecycle_event(...)`
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `PointLifecycleEvent` - fields: `id`, `event_type`, `station_interval`,
    `loop_ref`, `point_ref`, `correspondence_id`, `parent_span_ref`, `source`,
    `interpolation_policy`, `diagnostics`.
  - `SyntheticSupportReference` - fields: `id`, `source_event_id`,
    `station_index`, `span_ref`, `span_parameter`, `coordinates`.
- Functions/methods:
  - `validate_point_lifecycle_event(event) -> None`
  - `event.lifecycle_state_for_station(station_index) -> PointLifecycleState`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Event validation is O(1) per event.
- Event lookup by id should be backed by maps when stored in correspondence
  plans.
- No geometry projection or sampling occurs in this spec.

## Error And State Behavior

- Missing station interval, loop reference, point/support reference, or parent
  span reference raises `ValueError`.
- Unknown lifecycle state or event type raises `ValueError`.
- Event ids must be unique inside a loop correspondence map.

## Test Strategy

- Unit tests:
  - all lifecycle states are accepted.
  - invalid event type/state rejected.
  - event ids unique within a map.
  - diagnostic provenance preserved.
  - birth/death event records serialize without geometry execution.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Production-data rule:
  - Tests must not require the user's production database.

## Acceptance Criteria

- Point birth/death is represented as explicit planner data.
- Synthetic support references are inspectable before resampling.
- Invalid lifecycle records fail independently of executor code.

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
