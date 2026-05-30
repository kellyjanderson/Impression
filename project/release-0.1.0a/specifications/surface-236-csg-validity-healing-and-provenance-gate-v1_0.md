# Surface Spec 236: CSG Validity, Healing, And Provenance Gate (v1.0)

## Overview

Validate, narrowly heal, and finalize provenance for CSG result bodies after
shell assembly and seam rebuild.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification extracts validity, healing, and provenance finalization
hidden inside Surface Spec 221.

## Responsibilities

- Functions/methods:
  - result validity gate
  - bounded healing pass
  - provenance finalizer
- Data structures/models:
  - validity diagnostic record
  - CSG provenance metadata record
- Dependencies/services:
  - seam rebuild output
  - surface body metadata model
- Returns/outputs/signals:
  - accepted `SurfaceBody`
  - invalid result diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface validation and metadata containers
  - Additions to existing reusable library/module: CSG validity/provenance helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes final CSG result acceptance
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded validation and cleanup
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- CSG result finalization helpers

Routes:

- seam rebuild to public `SurfaceBooleanResult`

Reuse/extraction decision:

- add to existing CSG module/private helpers

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- bounded healing may normalize topology but cannot invent materially different
  geometry

Data ownership:

- result `SurfaceBody` owns final topology; CSG owns operation provenance

## Behavior

The implementation must:

- validate trim loops, seam participation, shell classification, and metadata
- perform only deterministic cleanup such as zero-measure removal and loop
  orientation normalization
- reject invalid results with diagnostics rather than mesh repair or rewrap
- attach deterministic operation and operand provenance

## Verification

Test strategy:

- validity, bounded-healing, invalid-result, and provenance fixtures

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

- Review for split. Cohesion reason: validity, bounded cleanup, and provenance
  are one final acceptance gate.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

