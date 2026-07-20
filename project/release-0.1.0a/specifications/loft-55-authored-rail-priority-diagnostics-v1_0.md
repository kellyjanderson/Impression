# Loft Spec 55: Authored Rail Priority and Diagnostics (v1.0)

Date: 2026-05-25
Status: Proposed
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `Authored Rail Priority And Diagnostics`

## Purpose

Define deterministic priority order and diagnostics for authored
correspondence rails before the loft planner considers high-confidence
inference.

## Scope

Owns:

- Rail priority resolution for explicit ids, landmark names, segment names,
  authored start/direction, and generated rails.
- Conflict diagnostics when authored rails contradict each other.
- A resolved rail map consumed by inference and loop correspondence planning.

Does not own:

- Inference scoring or acceptance thresholds.
- Point birth/death lifecycle events.
- Resampling or executor behavior.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - add rail priority resolver and
    diagnostics near loop correspondence planning.
- Supporting modules/files:
  - `src/impression/modeling/topology.py` - source topology identity records.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - reusable planner rail resolution
    boundary.
- Tests:
  - `tests/test_loft_authored_rail_priority.py` - each priority tier and
    conflict diagnostics.

## Chosen Defaults / Parameters

- Rail priority order is:
  1. explicit `correspondence_id`
  2. matching landmark names
  3. matching segment names plus segment-local parameters
  4. authored start and direction
  5. generated shape default rails
  6. inference
  7. refusal
- Higher-priority rails override lower-priority rails only when they do not
  conflict internally.
- Conflicting explicit `correspondence_id` rails are invalid input, not
  ambiguity.

## Data Ownership

- Source of truth: topology identity records own authored rails; loft planner
  owns the resolved rail map.
- Read ownership: loft planning and diagnostics read rail maps.
- Write ownership: rail resolver writes derived maps and diagnostics; it does
  not mutate topology records.
- Derived/cache data: rail maps are recomputable from normalized topology and
  identity records.
- Privacy/logging constraints: diagnostics may include ids/names/roles and
  conflict reasons only.

## Dependencies And Routes

- Domain/service dependencies:
  - `TopologyPath`
  - `TopologyPoint`
  - `TopologySegment`
  - `TopologyLandmark`
  - existing loft ambiguity diagnostics.
- Database dependencies:
  - none.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; deterministic synchronous planner step.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Existing loft ambiguity/diagnostic record patterns.
  - Existing loop correspondence planning entrypoints.
- Current reuse readiness:
  - add to existing loft planner module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `resolve_authored_rails(...)`
  - `RailResolutionResult`
  - `RailSource`
  - `RailConflictDiagnostic`
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `RailResolutionResult` - fields: `matches`, `source_by_match`,
    `conflicts`, `unmatched_source`, `unmatched_target`.
  - `RailSource` - enum values: `explicit_id`, `landmark_name`,
    `segment_name`, `authored_order`, `generated_rail`.
  - `RailConflictDiagnostic` - fields: `reason`, `source_ref`, `target_ref`,
    `priority_tier`.
- Functions/methods:
  - `resolve_authored_rails(source_loop, target_loop) -> RailResolutionResult`
  - `validate_rail_priority(result: RailResolutionResult) -> None`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Resolution is O(n) for explicit ids and names, plus O(s) for segment ranges.
- No geometric inference search runs in this spec.
- Diagnostics construction is bounded by the number of authored rail records.

## Error And State Behavior

- Duplicate explicit correspondence ids on the same side raise invalid input.
- Crossing authored rail order raises invalid input when caused by explicit
  rails; lower-priority generated rail conflicts degrade to diagnostics.
- If rails are insufficient but non-conflicting, the result is unresolved and
  may be passed to inference.

## Test Strategy

- Unit tests:
  - explicit ids beat names and generated rails.
  - landmark names beat authored order.
  - segment names partition ranges.
  - generated rails apply only after authored rails.
  - conflicting explicit rails produce invalid-input diagnostics.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Production-data rule:
  - Tests must not require the user's production database.

## Acceptance Criteria

- Authored correspondence intent resolves deterministically before inference.
- Conflicts are reported with priority-tier diagnostics.
- Unresolved but non-conflicting rail maps are explicitly marked for inference
  or refusal.

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
