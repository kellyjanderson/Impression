# Loft Spec 64: Generated Shape Default Rails (v1.0)

Date: 2026-05-25
Status: Proposed
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `Generated Shape Default Rails`

## Purpose

Define generated topology rails for common shape helpers so rectangles, circles,
and rounded rectangles are useful authored topology inputs instead of anonymous
point lists.

## Scope

Owns:

- `TopologyPath.named_rect`, `TopologyPath.named_rounded_rect`, and
  `TopologyPath.named_circle`.
- Default generated names, correspondence ids, roles, and provenance for
  common landmarks.
- Override behavior for user-supplied names or anchors.

Does not own:

- General point/segment builder syntax.
- Generated shape geometry primitives outside topology helper output.
- Loft planner rail priority or inference.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/topology.py` - add generated shape topology
    helpers and generated rail provenance.
- Supporting modules/files:
  - Existing generated shape helper modules - source dimensions/point
    generation where reusable.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/topology.py` - generated topology helper API.
- Tests:
  - `tests/test_topology_generated_shape_rails.py` - generated names, anchors,
    correspondence ids, provenance, and overrides.

## Chosen Defaults / Parameters

- Rectangles generate corner rails and side-midpoint rails.
- Circles generate authored start and quadrant rails.
- Rounded rectangles generate straight segment names, corner arc names, and
  tangent transition landmarks.
- `anchor` defaults to the generated start/corner appropriate to the shape.
- User-supplied names or correspondence ids override generated ones but keep
  generated provenance for non-overridden rails.
- `name_prefix` may namespace generated ids for multiple shapes.

## Data Ownership

- Source of truth: generated helper creates topology records with generated
  provenance.
- Read ownership: topology adapters and loft planner read generated rails like
  other topology identity records.
- Write ownership: generated helper writes records at construction; downstream
  planner writes derived correspondence.
- Derived/cache data: generated rails are recomputable from shape dimensions,
  anchor, and name prefix.
- Privacy/logging constraints: diagnostics may include generated ids, names,
  dimensions, and override names only.

## Dependencies And Routes

- Domain/service dependencies:
  - topology path core records
  - topology identity records
  - existing generated shape helpers
- Database dependencies:
  - none.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; helper construction is synchronous.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Existing rectangle, rounded rectangle, and circle point/curve generation
    helpers where present.
- Current reuse readiness:
  - add generated topology helpers to existing topology module.
- Extraction/wrapping needed:
  - wrap generated shape point output into topology points, segments, and
    landmarks with provenance.
- Additions to existing library/modules:
  - `TopologyPath.named_rect(...)`
  - `TopologyPath.named_rounded_rect(...)`
  - `TopologyPath.named_circle(...)`
  - `GeneratedRailProvenance`
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `GeneratedRailProvenance` - fields: `shape_kind`, `name_prefix`,
    `generated_role`, `source_parameter`, `overridden`.
- Functions/methods:
  - `TopologyPath.named_rect(width, height, *, anchor="bottom-left",
    name_prefix=None) -> TopologyPath`
  - `TopologyPath.named_rounded_rect(width, height, radius, *,
    anchor="bottom-left", name_prefix=None) -> TopologyPath`
  - `TopologyPath.named_circle(radius, *, anchor=None,
    name_prefix=None) -> TopologyPath`
- UI fields / visible data, if applicable:
  - public arguments: `width`, `height`, `radius`, `anchor`, `name_prefix`.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Generated helper construction is O(1) for each supported primitive.
- Helpers do not run planner inference or resampling.
- Generated rail override validation is O(n) in generated rail count.

## Error And State Behavior

- Non-positive dimensions or radius raise `ValueError`.
- Invalid anchor names raise `ValueError` and list valid generated anchors.
- Override collisions raise `ValueError`.
- Generated helpers record provenance so diagnostics can distinguish generated
  rails from authored rails.

## Test Strategy

- Unit tests:
  - rectangle generated corner and midpoint names.
  - circle generated quadrant rails and authored start.
  - rounded rectangle segment, arc, and tangent transition rails.
  - `name_prefix` namespaces generated ids.
  - invalid dimensions, invalid anchor, and override collision failures.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Production-data rule:
  - Tests must not require the user's production database.

## Acceptance Criteria

- Common generated shapes produce topology paths with useful default rails.
- Users can override generated names without losing provenance for other rails.
- Generated rails participate in the same identity model as authored rails.

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
