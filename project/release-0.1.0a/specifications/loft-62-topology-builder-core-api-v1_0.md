# Loft Spec 62: Topology Builder Core API (v1.0)

Date: 2026-05-25
Status: Proposed
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `Topology Builder Core API`

## Purpose

Define the lightweight public builder API for ordinary topology points and
curve segments so users can author correspondence rails without constructing
internal records by hand.

## Scope

Owns:

- `TopologyPath.closed` builder entrypoint.
- `path.point` and `path.segment` builder methods.
- Helper alias normalization from `correspond=` to `correspondence_id`.
- Builder validation for point/segment authoring.

Does not own:

- Birth/death lifecycle builder helpers.
- Generated shape helpers.
- Loft planner rail resolution or inference.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/topology.py` - add builder object and core builder
    methods.
- Supporting modules/files:
  - `docs` or examples location if present - later examples may use these
    helpers, but docs are not required for this spec.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/topology.py` - public modeling API builder.
- Tests:
  - `tests/test_topology_builder_core_api.py` - builder chaining, alias
    normalization, validation, and output records.

## Chosen Defaults / Parameters

- `TopologyPath.closed(anchor=...)` creates a builder with `closed=True`.
- `path.point(name, coordinates, correspond=None, id=None, role=None)` creates
  a `TopologyPoint`.
- `path.segment(name, curve=None, landmarks=None, correspond=None)` creates a
  `TopologySegment` and optional landmarks.
- `correspond=` is accepted as authoring shorthand and normalized into
  `correspondence_id`.
- If `id` is omitted and `name` is present, the helper derives a stable id from
  the name and records derived provenance.

## Data Ownership

- Source of truth: builder owns temporary authoring state until finalized into
  topology records.
- Read ownership: user code reads returned builder/path objects through public
  API; planner reads finalized records.
- Write ownership: builder methods append records in authored order; downstream
  planner does not mutate builder state.
- Derived/cache data: finalized `TopologyPath` records derive from builder
  state.
- Privacy/logging constraints: validation diagnostics may include names, ids,
  coordinates, and helper argument names only.

## Dependencies And Routes

- Domain/service dependencies:
  - `TopologyPath`
  - `TopologyPoint`
  - `TopologySegment`
  - `TopologyLandmark`
- Database dependencies:
  - none.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; builder is synchronous and local.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Topology path and identity records from specs 52 and 53.
- Current reuse readiness:
  - add builder helpers to existing topology module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `TopologyPath.closed(...)`
  - `TopologyPathBuilder`
  - `TopologyPathBuilder.point(...)`
  - `TopologyPathBuilder.segment(...)`
  - `TopologyPathBuilder.build()`
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `TopologyPathBuilder` - fields: `path_id`, `closed`, `anchor`,
    `direction`, `points`, `segments`, `landmarks`.
- Functions/methods:
  - `TopologyPath.closed(*, anchor=None, direction="forward",
    sampling_policy=None) -> TopologyPathBuilder`
  - `TopologyPathBuilder.point(name, coordinates, *, id=None, correspond=None,
    role=None) -> TopologyPathBuilder`
  - `TopologyPathBuilder.segment(name, *, curve=None, landmarks=None,
    correspond=None) -> TopologyPathBuilder`
  - `TopologyPathBuilder.build() -> TopologyPath`
- UI fields / visible data, if applicable:
  - public arguments: `name`, `id`, `correspond`, `anchor`, `points`, `curve`,
    `landmarks`.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Builder append operations are O(1) excluding validation of supplied landmark
  lists.
- `build()` validation is O(n) in authored records.
- Builder methods must not run loft planning or correspondence inference.

## Error And State Behavior

- Duplicate point names or ids fail during `build()` or earlier if detectable.
- Invalid `correspond` values fail with `ValueError`.
- A segment with neither curve nor sufficient point references fails.
- Builder validation errors identify the method/argument when possible.

## Test Strategy

- Unit tests:
  - chained builder calls preserve authored order.
  - `correspond=` normalizes to `correspondence_id`.
  - derived ids from names are stable.
  - segment landmarks attach to segment ids.
  - invalid duplicate names and invalid segment definitions fail.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Production-data rule:
  - Tests must not require the user's production database.

## Acceptance Criteria

- Users can author named points and curve segments with compact builder calls.
- Builder output is the same topology record model used by lower-level APIs.
- Builder methods preserve authored order and do not perform hidden loft
  inference.

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
