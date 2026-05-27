# Loft Spec 66: Topology Segment Builder API (v1.0)

## Overview

Define lightweight user-facing builder methods for topology segments, segment
point arrays, curve references, landmarks, and closed-path construction.

## Backlink

- [Architecture: Loft Topology Point Correspondence Architecture](../architecture/loft-topology-point-correspondence-architecture.md)

## Scope

This specification promotes the manifest candidate `Topology Segment Builder
API` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - `TopologyPath.closed`
  - `path.segment`
  - segment builder validation
- Data structures/models:
  - topology segment builder state
- Dependencies/services:
  - topology path core records
  - segment identity records
- Returns/outputs/signals:
  - topology segment records
  - segment validation errors
- UI surfaces/components:
  - public modeling API builder
- UI fields/elements:
  - `name`
  - `id`
  - `points`
  - `curve`
  - `landmarks`
- Reusable code plan:
  - Existing code reused as-is: topology path records and segment records
  - Additions to existing reusable library/module: `impression.modeling.topology`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - segment validation avoids hidden expensive correspondence solving
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/topology.py`

Routes:

- segment builder calls to topology records to section to loft planner

Reuse/extraction decision:

- add segment helpers to the topology module as thin wrappers over records

UI field/control inventory:

- builder arguments: `name`, `id`, `points`, `curve`, `landmarks`

## Data And Defaults

Chosen defaults / parameters:

- segment order is authored order
- path start remains the default anchor
- `TopologyPath.closed` declares closure explicitly instead of relying on
  coincident start/end points

Data ownership:

- builder creates segment records
- planner owns later correspondence decisions

## Behavior

The implementation must:

- create topology segment records from authored point arrays or curve references
- preserve authored segment order and closure intent
- validate duplicate ids, insufficient points, conflicting `points` and `curve`
  payloads, invalid landmarks, and incompatible closed-path declarations
- report validation errors at the segment-builder boundary
- avoid lifecycle birth/death helpers, which remain in the lifecycle builder
  specification

## Verification

Test strategy:

- segment builder API tests
- closed path tests
- docs snippets for segment and closed-path authoring
- validation error tests for malformed points, invalid curves, and duplicate
  landmarks

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 5 x 1 = 5
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 22

Split decision:

- Review for split. Cohesion reason: this is one segment-authoring API surface
  after point authoring and lifecycle helpers were split out.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- public segment builder calls create validated topology segment records
- closed-path construction is explicit and deterministic
- planner handoff preserves authored segment order and landmark identity
