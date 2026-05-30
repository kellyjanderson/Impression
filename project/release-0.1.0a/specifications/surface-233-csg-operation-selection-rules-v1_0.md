# Surface Spec 233: CSG Operation Selection Rules (v1.0)

## Overview

Convert fragment classifications into operation-specific survive/discard/cut-cap
selection rules for union, difference, and intersection.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification extracts operation semantics hidden inside Surface Spec 221.

## Responsibilities

- Functions/methods:
  - union fragment selector
  - difference fragment selector
  - intersection fragment selector
- Data structures/models:
  - operation selection record
  - cut-cap requirement record
- Dependencies/services:
  - fragment classification records
  - CSG operation payload
- Returns/outputs/signals:
  - selected surviving fragments
  - explicit empty-result signal
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: existing split role names
  - Additions to existing reusable library/module: operation rule helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG operation behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by classified fragment count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- CSG operation selection helpers

Routes:

- classifications to shell assembly

Reuse/extraction decision:

- add to existing CSG private helpers

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- operation rules are explicit tables, not implicit conditionals scattered
  through reconstruction

Data ownership:

- CSG owns selection records

## Behavior

The implementation must:

- define survive/discard rules for inside, outside, and on-boundary fragments
  for all three operations
- identify when cut caps are required
- produce deterministic empty-result signals
- avoid changing classification truth during operation selection

## Verification

Test strategy:

- selection table tests for union, difference, and intersection over all
  relation classes

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 18.5

Split decision:

- Review for split. Cohesion reason: this is the operation-selection boundary.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

