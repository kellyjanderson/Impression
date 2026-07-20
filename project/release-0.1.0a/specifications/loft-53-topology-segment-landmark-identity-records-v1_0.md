# Loft Spec 53: Topology Segment and Landmark Identity Records (v1.0)

Date: 2026-05-25
Status: Proposed
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `Topology Segment And Landmark Identity Records`

## Purpose

Define durable point, segment, and landmark identity records so authored names,
ids, roles, and correspondence rails survive topology normalization and become
available to loft planning.

## Scope

Owns:

- `TopologyPoint`, `TopologySegment`, and `TopologyLandmark` record fields.
- Identity/provenance validation for ids, names, correspondence ids, roles, and
  protection policy.
- Record-level duplicate and missing-reference diagnostics.

Does not own:

- Public builder syntax for creating these records.
- Input adapters from existing curve types.
- Rail priority, high-confidence inference, birth/death resolution, or
  executor consumption.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/topology.py` - add identity record classes and
    validation helpers.
- Supporting modules/files:
  - `src/impression/modeling/loft.py` - later planner reads records; no direct
    executor work in this spec.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/topology.py` - reusable identity record boundary.
- Tests:
  - `tests/test_topology_identity_records.py` - ids, names, roles, duplicate
    rejection, provenance, and protection policy behavior.

## Chosen Defaults / Parameters

- `name` is user-facing and optional.
- `id` is stable record identity; if omitted and `name` is present, helpers may
  derive `id` from `name` and must mark provenance as derived.
- `correspondence_id` is the authoritative planner rail key.
- `protection_policy` defaults to `protected` for corners, seams, explicit
  landmarks, and user-authored correspondence ids; otherwise it defaults to
  `sample`.
- Roles are plain string values initially; expected roles include `corner`,
  `seam`, `feature`, `tangent_transition`, `sample`, and `inferred_support`.

## Data Ownership

- Source of truth: topology identity records own authored point and landmark
  identity before normalization.
- Read ownership: adapters, station normalization, rail resolver, lifecycle
  resolver, and resampler may read identity records.
- Write ownership: topology builders/adapters create identity records;
  normalization writes mapped ids into derived correspondence structures.
- Derived/cache data: normalized ids and loop-local ordinals are derived from
  original identity records plus normalization diagnostics.
- Privacy/logging constraints: diagnostics may log ids, names, roles, and
  correspondence ids; no external file paths or user private content.

## Dependencies And Routes

- Domain/service dependencies:
  - `TopologyPath`
  - `Loop`
  - `Section`
  - station normalization path in `src/impression/modeling/loft.py`
- Database dependencies:
  - none.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; pure synchronous record validation.

## Reuse And Extraction Plan

- Existing code to reuse:
  - `Loop` and `Section` identity/reference handling where present.
- Current reuse readiness:
  - add to existing reusable topology module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `TopologyPoint`
  - `TopologySegment`
  - `TopologyLandmark`
  - `validate_topology_identity_records(...)`
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `TopologyPoint` - fields: `id`, `coordinates`, `ordinal`, `role`, `name`,
    `correspondence_id`, `protection_policy`, `provenance`.
  - `TopologySegment` - fields: `id`, `name`, `source_kind`, `start_ref`,
    `end_ref`, `curve`, `correspondence_id`, `provenance`.
  - `TopologyLandmark` - fields: `id`, `name`, `segment_id`, `parameter`,
    `point_ordinal`, `role`, `correspondence_id`, `protection_policy`,
    `provenance`.
- Functions/methods:
  - `validate_topology_identity_records(path: TopologyPath) -> None` - checks
    identity uniqueness and references.
  - `derive_stable_id(name: str) -> str` - deterministic helper id derivation.
- UI fields / visible data, if applicable:
  - public API fields: `id`, `name`, `role`, `correspondence_id`,
    `protection_policy`.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Validation is O(n) over points, segments, and landmarks.
- Duplicate checks use sets/maps and do not perform geometric search.
- Record creation must not trigger sampling or loft planning.

## Error And State Behavior

- Duplicate ids or correspondence ids in a scope that requires uniqueness raise
  `ValueError`.
- Missing segment references or invalid point ordinals raise `ValueError`.
- Conflicting protection policies on the same correspondence id raise
  `ValueError`.
- Diagnostics should include the owning path id and offending record id.

## Test Strategy

- Unit tests:
  - stable id derivation from names.
  - explicit ids overriding derived ids.
  - duplicate id and missing reference failures.
  - protection policy defaults by role and correspondence id.
  - provenance preservation through validation.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Production-data rule:
  - Tests must not require the user's production database.

## Acceptance Criteria

- Segment, landmark, and point records can represent named authored rails.
- Duplicate or broken identity records fail before loft planning.
- Records expose enough identity for station normalization and rail resolution
  without depending on sample indices.

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
