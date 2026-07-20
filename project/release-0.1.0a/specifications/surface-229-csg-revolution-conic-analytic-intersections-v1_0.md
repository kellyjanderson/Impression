# Surface Spec 229: CSG Revolution/Conic Analytic Intersections (v1.0)

## Overview

Implement exact or explicitly bounded analytic intersections for revolution and
conic patch pairs that are common in surface CSG.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification splits higher-order analytic revolution/conic intersections
out of Surface Spec 219.

## Responsibilities

- Functions/methods:
  - plane/cylinder and plane/cone intersection
  - plane/sphere intersection
  - axis-compatible cylinder/cylinder intersection gate
- Data structures/models:
  - revolution intersection record
  - conic diagnostic record
- Dependencies/services:
  - CSG curve primitives
  - patch-local curve mapping
  - revolution patch evaluators
- Returns/outputs/signals:
  - exact line, circle, arc, or conic records
  - explicit unsupported/degenerate diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: revolution patch payloads
  - Additions to existing reusable library/module: revolution/conic CSG helpers
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
  - bounded analytic computation
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

- execute exact supported revolution/conic cases; refuse unsupported
  general-case algebra before mesh fallback is reachable

Data ownership:

- CSG owns intersection records and diagnostics

## Behavior

The implementation must:

- handle the named revolution/conic pairs with deterministic exact records
- identify tangent, coincident, singular-axis, and periodic seam cases
- return explicit diagnostics when the requested pair exceeds the supported
  solver boundary
- keep sampled tessellation outside the CSG truth path

## Verification

Test strategy:

- unit fixtures for plane/cylinder, plane/cone, plane/sphere, and supported
  cylinder/cylinder cases plus degeneracies

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
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
- Total: 19.5

Split decision:

- Review for split. Cohesion reason: this is the revolution/conic analytic set.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

