# Surface Spec 234: CSG Shell Assembly From Fragments (v1.0)

## Overview

Assemble selected surface fragments into result shells before seam rebuilding
and validity cleanup.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification extracts shell assembly hidden inside Surface Spec 221.

## Responsibilities

- Functions/methods:
  - fragment-to-patch assembler
  - shell grouping helper
  - empty/multi-shell result builder
- Data structures/models:
  - shell assembly record
  - fragment provenance map
- Dependencies/services:
  - operation selection records
  - `SurfaceBody` and `SurfaceShell`
- Returns/outputs/signals:
  - provisional result shells
  - empty-result classification
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface body/shell containers
  - Additions to existing reusable library/module: CSG shell assembly helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG result construction
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by selected fragment count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- CSG result assembly helpers

Routes:

- operation selection to seam rebuild

Reuse/extraction decision:

- reuse existing surface containers

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- shell grouping is deterministic and preserves source provenance

Data ownership:

- result `SurfaceBody` owns assembled shells after validity gates pass

## Behavior

The implementation must:

- build provisional shells from selected fragments without mesh rewrap
- preserve source patch identity and operation provenance
- represent empty and multi-shell results explicitly
- refuse incomplete fragment connectivity before shell truth is published

## Verification

Test strategy:

- shell assembly fixtures for single-shell, multi-shell, and empty outcomes

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

- Review for split. Cohesion reason: this is one shell assembly stage.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

