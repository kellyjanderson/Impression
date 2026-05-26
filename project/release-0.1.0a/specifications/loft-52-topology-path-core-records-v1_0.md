# Loft Spec 52: Topology Path Core Records (v1.0)

Date: 2026-05-25
Status: Proposed
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `Topology Path Core Records`

## Purpose

Define the reusable topology path container that preserves authored path intent
before station normalization, loft planning, and resampling alter loop order or
sample count.

## Scope

Owns:

- `TopologyPath` record shape for closure, authored start, traversal direction,
  anchor policy, and sampling policy.
- Validation for path-level invariants that do not require segment, landmark,
  or planner correspondence logic.
- Conversion boundary from helpers/adapters into a path record suitable for
  `Section` construction.

Does not own:

- Segment and landmark identity records.
- `Path2D`, `BSpline2D`, or generated shape adapters.
- Loft rail resolution, inference, resampling, or executor behavior.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/topology.py` - add `TopologyPath` and
    `TopologyPathSamplingPolicy`.
- Supporting modules/files:
  - `src/impression/modeling/loft.py` - no direct behavior changes; later specs
    consume the record.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/topology.py` - public reusable topology path
    boundary.
- Tests:
  - `tests/test_topology_path_records.py` - path construction, defaults,
    validation, and section handoff.

## Chosen Defaults / Parameters

- Closed paths preserve authored start and authored direction.
- Unnamed authored point arrays use ordinal `0` as authored anchor.
- `anchor_policy` defaults to `authored`.
- `direction` defaults to `forward`.
- `sampling_policy.sample_count` defaults to `"auto"`.
- Winding normalization may reverse order later only when validity requires it
  and must preserve a mapping back to authored start.
- Invalid path records raise `ValueError` at construction time.

## Data Ownership

- Source of truth: `TopologyPath` owns path-level authored intent.
- Read ownership: topology adapters, section construction, and loft planner may
  read through the public dataclass/record API.
- Write ownership: constructors and builder/adapters create records; downstream
  normalization creates derived mapped records rather than mutating originals.
- Derived/cache data: normalized loop order and point maps are recomputable from
  the authored path plus normalization diagnostics.
- Privacy/logging constraints: diagnostics may include ids, names, roles, and
  numeric coordinates; no user files or external data are logged.

## Dependencies And Routes

- Domain/service dependencies:
  - `Loop`
  - `Section`
  - topology validation helpers in `impression.modeling.topology`
- Database dependencies:
  - none.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; pure synchronous modeling records.

## Reuse And Extraction Plan

- Existing code to reuse:
  - `src/impression/modeling/topology.py` - existing loop/section primitives.
  - Existing finite-coordinate and loop validation helpers where present.
- Current reuse readiness:
  - add to existing reusable topology module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `TopologyPath`
  - `TopologyPathSamplingPolicy`
  - `TopologyPath.validate()`
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `TopologyPath` - fields: `id`, `closed`, `anchor_id`,
    `anchor_policy`, `direction`, `sampling_policy`, `segments`, `metadata`.
  - `TopologyPathSamplingPolicy` - fields: `sample_count`, `min_span_samples`,
    `preserve_protected_landmarks`.
- Functions/methods:
  - `TopologyPath.validate() -> None` - validates path-level invariants.
  - `TopologyPath.to_section_loop() -> Loop` - creates a basic loop only after
    identity/adapters have supplied points.
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Construction and validation are O(n) in stored segments/points.
- Validation must not run geometric correspondence solving.
- No caching is required in this spec.

## Error And State Behavior

- Empty paths, non-finite coordinates, invalid direction, invalid anchor policy,
  and invalid sample count raise `ValueError`.
- Closed paths with insufficient path evidence raise invalid input, not
  underconstrained correspondence.
- Validation errors should identify the path id when available.

## Test Strategy

- Unit tests:
  - authored start and direction defaults.
  - explicit anchor policy validation.
  - `sample_count="auto"` and explicit positive sample count validation.
  - invalid empty, non-finite, and contradictory closed path inputs.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Production-data rule:
  - Tests must not require the user's production database.

## Acceptance Criteria

- `TopologyPath` can represent an authored closed path without rotating its
  start.
- Invalid path-level states fail before loft planning begins.
- The record can be constructed and read by later specs without depending on
  mesh or executor code.

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
