# Surface Spec 235: CSG Seam And Adjacency Rebuild (v1.0)

## Overview

Rebuild seam, boundary-use, and adjacency truth for provisional CSG result
shells.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification extracts seam and adjacency rebuilding hidden inside Surface
Spec 221.

## Responsibilities

- Functions/methods:
  - cut-boundary seam builder
  - retained-boundary seam mapper
  - adjacency rebuild helper
- Data structures/models:
  - seam rebuild record
  - boundary-use provenance record
- Dependencies/services:
  - shell assembly records
  - seam/adjacency model
- Returns/outputs/signals:
  - rebuilt seams
  - adjacency records
  - seam validation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: seam validation and adjacency containers
  - Additions to existing reusable library/module: CSG seam rebuild helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes result topology construction
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by result boundary count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- CSG seam rebuild helpers

Routes:

- shell assembly to validity gate

Reuse/extraction decision:

- reuse existing seam validation and adjacency APIs

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- cut boundaries become durable seams or open boundaries before tessellation

Data ownership:

- result shell owns rebuilt seams and adjacency truth

## Behavior

The implementation must:

- create seam records for shared cut boundaries
- preserve retained operand seams where still valid
- classify open boundaries explicitly
- reject mismatched seam pairings or orientation ambiguity

## Verification

Test strategy:

- seam rebuild fixtures for retained seams, new cut seams, open boundaries, and
  orientation reversal

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 3 x 1 = 3
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
- Total: 19.5

Split decision:

- Review for split. Cohesion reason: seam and adjacency rebuild are one topology
  truth stage.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

