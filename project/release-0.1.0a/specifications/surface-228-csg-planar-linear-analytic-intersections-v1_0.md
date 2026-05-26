# Surface Spec 228: CSG Planar/Linear Analytic Intersections (v1.0)

## Overview

Implement exact CSG intersection discovery for planar and linear analytic patch
pairs before higher-order analytic pairs.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification splits the low-order analytic portion out of Surface Spec
219.

## Responsibilities

- Functions/methods:
  - plane/plane intersection
  - plane/ruled intersection gate
  - coincident/disjoint planar relation classifier
- Data structures/models:
  - analytic intersection record
  - planar relation diagnostic
- Dependencies/services:
  - CSG curve primitives
  - patch-local curve mapping
- Returns/outputs/signals:
  - exact line/segment curve records
  - coincident/disjoint/touching diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: planar patch payloads
  - Additions to existing reusable library/module: CSG analytic intersection helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG intersection behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded pairwise analytic computation
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- CSG analytic intersection helpers

Routes:

- prepared operands to intersection stage

Reuse/extraction decision:

- reuse Specs 226 and 227

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- low-order intersections execute exactly; ambiguous coincidence refuses unless
  a later operation rule handles it

Data ownership:

- CSG owns intersection records

## Behavior

The implementation must:

- emit exact planar/linear curve records for crossing cases
- classify disjoint, touching, parallel, and coincident cases deterministically
- map curves into patch-local coordinates
- avoid mesh clipping or sampled intersection as CSG truth

## Verification

Test strategy:

- fixtures for crossing, parallel, coincident, disjoint, and touching
  planar/linear pairs

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

- Review for split. Cohesion reason: this is the low-order analytic CSG set.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

