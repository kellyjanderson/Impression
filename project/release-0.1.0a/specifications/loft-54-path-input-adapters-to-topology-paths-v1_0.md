# Loft Spec 54: Path Input Adapters to Topology Paths (v1.0)

Date: 2026-05-25
Status: Proposed
Parent: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md` / `Path Input Adapters To Topology Paths`

## Purpose

Define how existing point arrays, `Path2D`, `BSpline2D`, and generated helper
output become topology paths while preserving authored starts, curve
parameters, landmarks, and provenance.

## Scope

Owns:

- Adapter factory methods on `TopologyPath`.
- Adapter provenance records and validation errors.
- Mapping from polyline points and curve parameters into topology identity
  records.

Does not own:

- Generated shape default rail names.
- Builder syntax beyond the factory methods named here.
- Loft planner rail resolution or inference.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/topology.py` - add `TopologyPath.from_points`,
    `TopologyPath.from_path2d`, and `TopologyPath.from_bspline`.
- Supporting modules/files:
  - `src/impression/modeling/paths.py` or current `Path2D` owner - read-only
    dependency for adapter input.
  - `src/impression/modeling/curves.py` or current `BSpline2D` owner - read-only
    dependency for curve input.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/topology.py` - reusable adapter boundary.
- Tests:
  - `tests/test_topology_path_adapters.py` - point, path, b-spline, provenance,
    and validation behavior.

## Chosen Defaults / Parameters

- `from_points(..., closed=True)` uses point index `0` as authored anchor unless
  `anchor` names another supplied point.
- Tuple form `("name", (x, y))` creates a point with `name`, derived `id`, and
  matching `correspondence_id` unless overridden.
- `from_bspline` attaches correspondence to parameters and landmarks, not to
  raw control-point indices.
- Adapter methods do not infer point correspondence across stations.
- Adapter methods accept `sampling_policy`, but do not allocate loft samples.

## Data Ownership

- Source of truth: source point/curve objects own raw geometry; resulting
  `TopologyPath` owns authored topology intent after conversion.
- Read ownership: adapters read source objects through their public API.
- Write ownership: adapters create new topology records and do not mutate
  source `Path2D` or `BSpline2D` instances.
- Derived/cache data: adapter provenance records can recompute source mapping
  for diagnostics.
- Privacy/logging constraints: diagnostics may include source type and ids, but
  not external file paths or private user content.

## Dependencies And Routes

- Domain/service dependencies:
  - `Path2D`
  - `BSpline2D`
  - `TopologyPath`
  - `TopologyPoint`
  - `TopologyLandmark`
- Database dependencies:
  - none.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; adapter conversion is synchronous.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Existing `Path2D` point iteration/evaluation APIs.
  - Existing `BSpline2D` evaluation and parameter APIs.
  - Generated shape helpers as input providers only.
- Current reuse readiness:
  - add adapter methods to existing topology module.
- Extraction/wrapping needed:
  - thin wrapper around curve parameter landmarks if `BSpline2D` has no native
    landmark object.
- Additions to existing library/modules:
  - `TopologyPath.from_points(...)`
  - `TopologyPath.from_path2d(...)`
  - `TopologyPath.from_bspline(...)`
  - `TopologyAdapterProvenance`
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `TopologyAdapterProvenance` - fields: `source_kind`, `source_id`,
    `source_ref`, `parameter_map`, `point_ordinal_map`.
- Functions/methods:
  - `TopologyPath.from_points(points, *, closed=True, anchor=None,
    direction="forward", landmarks=None, sampling_policy=None) -> TopologyPath`
  - `TopologyPath.from_path2d(path, *, closed=None, anchor=None,
    landmarks=None, sampling_policy=None) -> TopologyPath`
  - `TopologyPath.from_bspline(curve, *, closed=None, anchor=None,
    landmarks=None, sampling_policy=None) -> TopologyPath`
- UI fields / visible data, if applicable:
  - public factory arguments: `closed`, `anchor`, `direction`, `landmarks`,
    `sampling_policy`, `correspondence_id`.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Point-array and `Path2D` conversion are O(n) in source point count.
- `BSpline2D` conversion stores curve and landmarks without dense sampling.
- Adapter validation must not run cyclic-shift, reversal, or birth/death
  correspondence solving.

## Error And State Behavior

- Non-finite coordinates, invalid landmark parameters, invalid anchor names,
  and empty input raise `ValueError`.
- Unsupported curve types raise `TypeError` with the accepted adapter list.
- If a closed curve has no explicit anchor, the adapter uses the source start
  parameter and records that decision.

## Test Strategy

- Unit tests:
  - unnamed point-array defaults.
  - named tuple points deriving ids and correspondence ids.
  - `Path2D` conversion preserving authored order.
  - `BSpline2D` parameter landmark preservation.
  - invalid anchor and invalid parameter failures.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Production-data rule:
  - Tests must not require the user's production database.

## Acceptance Criteria

- Existing point and curve inputs can become topology paths without losing
  authored start or parameter landmarks.
- Adapter outputs are topology records consumed by later specs, not sampled
  meshes.
- Invalid adapter inputs fail before loft planning.

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
