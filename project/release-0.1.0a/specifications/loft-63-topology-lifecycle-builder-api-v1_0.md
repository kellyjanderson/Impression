# Loft Spec 63: Topology Lifecycle Builder API (v1.0)

Date: 2026-05-25
Status: Proposed
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `Topology Lifecycle Builder API`

## Purpose

Define public builder helpers for authored point birth and death rails so users
can express lifecycle intent without constructing internal lifecycle event
records directly.

## Scope

Owns:

- `path.birth_span`, `path.birth_arc`, and `path.death_span` helper methods.
- Helper validation for explicit parent spans and authored lifecycle order.
- Translation from lifecycle helper calls into topology records consumed by the
  loft lifecycle resolver.

Does not own:

- Core point/segment builder helpers.
- Planner-side lifecycle event DTOs.
- Parent-span inference or synthetic support insertion.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/topology.py` - add lifecycle builder helpers.
- Supporting modules/files:
  - `src/impression/modeling/loft.py` - later consumes lifecycle-capable
    topology records.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/topology.py` - public lifecycle authoring API.
- Tests:
  - `tests/test_topology_lifecycle_builder_api.py` - birth span, birth arc,
    death span, parent validation, and provenance.

## Chosen Defaults / Parameters

- Lifecycle helper calls require explicit `parent` span references.
- Authored order inside `points` is preserved.
- `birth_span` inserts one or more born points into a parent span.
- `birth_arc` creates a curve segment and landmarks for a born arc.
- `death_span` marks an authored point or span as ending against a target parent
  span.
- Helpers create lifecycle-capable topology records; final lifecycle event
  creation remains planner-owned.

## Data Ownership

- Source of truth: topology builder owns authored lifecycle intent until
  finalized into topology records.
- Read ownership: loft lifecycle resolver reads finalized lifecycle intent.
- Write ownership: lifecycle builder methods append lifecycle requests; planner
  writes actual lifecycle events later.
- Derived/cache data: lifecycle requests derive from helper arguments and
  parent span refs.
- Privacy/logging constraints: diagnostics may include helper names, parent
  refs, point names, and coordinates only.

## Dependencies And Routes

- Domain/service dependencies:
  - `TopologyPathBuilder`
  - `TopologyPoint`
  - `TopologySegment`
  - lifecycle resolver DTOs by field compatibility only
- Database dependencies:
  - none.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; builder is synchronous and local.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Core topology builder state.
  - Topology point/segment/landmark identity records.
- Current reuse readiness:
  - add lifecycle helpers to existing topology module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `TopologyPathBuilder.birth_span(...)`
  - `TopologyPathBuilder.birth_arc(...)`
  - `TopologyPathBuilder.death_span(...)`
  - `TopologyLifecycleBuilderRequest`
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `TopologyLifecycleBuilderRequest` - fields: `request_type`, `parent`,
    `points`, `curve`, `name`, `radius`, `provenance`.
- Functions/methods:
  - `TopologyPathBuilder.birth_span(parent, *, points) -> TopologyPathBuilder`
  - `TopologyPathBuilder.birth_arc(name, *, parent, start, end, radius,
    correspond=None) -> TopologyPathBuilder`
  - `TopologyPathBuilder.death_span(parent, *, points=None,
    names=None) -> TopologyPathBuilder`
- UI fields / visible data, if applicable:
  - public arguments: `parent`, `points`, `name`, `radius`, `start`, `end`,
    `correspond`.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Lifecycle helper append operations are O(k) in supplied points.
- Parent span validation is structural only and does not solve correspondence.
- No synthetic support projection runs inside the builder.

## Error And State Behavior

- Missing parent span raises `ValueError`.
- Parent references to unknown point names fail during `build()`.
- `birth_arc` with invalid radius or endpoints raises `ValueError`.
- Death helpers with no points or names raise `ValueError`.

## Test Strategy

- Unit tests:
  - birth span preserves point order.
  - birth arc creates curve/landmark lifecycle intent.
  - death span records parent refs and target points.
  - unknown parent refs fail.
  - invalid arc radius fails.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Production-data rule:
  - Tests must not require the user's production database.

## Acceptance Criteria

- Users can express point birth and death intent with compact helper calls.
- Lifecycle helper output is inspectable topology data before loft planning.
- Helpers require enough parent-span rails to avoid ambiguous authored intent.

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
