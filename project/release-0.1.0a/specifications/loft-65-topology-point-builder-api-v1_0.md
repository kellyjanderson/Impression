# Loft Spec 65: Topology Point Builder API (v1.0)

## Overview

Define lightweight user-facing builder methods for ordinary topology points,
anchors, stable names, and correspondence ids.

## Backlink

- [Architecture: Loft Topology Point Correspondence Architecture](../architecture/loft-topology-point-correspondence-architecture.md)

## Scope

This specification promotes the manifest candidate `Topology Point Builder API`
into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - `path.point`
  - point builder validation
- Data structures/models:
  - topology point builder state
- Dependencies/services:
  - topology path core records
  - landmark identity records
- Returns/outputs/signals:
  - topology point records
  - point validation errors
- UI surfaces/components:
  - public modeling API builder
- UI fields/elements:
  - `name`
  - `id`
  - `correspond`
  - `anchor`
- Reusable code plan:
  - Existing code reused as-is: topology path records and landmark records
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
  - point validation avoids hidden expensive correspondence solving
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/topology.py`

Routes:

- point builder calls to topology records to section to loft planner

Reuse/extraction decision:

- add point helpers to the topology module as thin wrappers over records

UI field/control inventory:

- builder arguments: `name`, `id`, `correspond`, `anchor`

## Data And Defaults

Chosen defaults / parameters:

- `correspond=` is a helper alias for `correspondence_id`
- omitted ids derive from stable names with provenance
- path start remains the default anchor unless the user authors another anchor

Data ownership:

- builder creates point records
- planner owns later correspondence decisions

## Behavior

The implementation must:

- create topology point records without requiring users to construct internal
  records manually
- preserve authored point order and anchor intent
- validate duplicate ids, invalid names, missing required coordinates, and
  incompatible correspondence ids
- report validation errors at the point-builder boundary
- avoid automatic correspondence solving while authoring point records

## Verification

Test strategy:

- point builder API tests
- docs snippets for authored topology points
- validation error tests for duplicate ids, invalid anchors, and malformed
  correspondence ids
- planner handoff smoke tests proving point records are consumed without
  changing authored order

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 4 x 1 = 4
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
- Total: 19

Split decision:

- Review for split. Cohesion reason: this is one point-authoring API surface
  after segment authoring was split out.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- public point builder calls create validated topology point records
- errors identify the exact authored field that failed
- planner handoff preserves authored point order and anchor intent
