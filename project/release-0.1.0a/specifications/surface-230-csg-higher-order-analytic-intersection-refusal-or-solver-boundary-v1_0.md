# Surface Spec 230: CSG Higher-Order Analytic Intersection Refusal Or Solver Boundary (v1.0)

## Overview

Define the explicit solver boundary and refusal behavior for higher-order CSG
intersection pairs such as torus, spline, sweep, subdivision, and implicit
surfaces.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification prevents unsupported advanced-family intersections from
hiding inside the analytic intersection library.

## Responsibilities

- Functions/methods:
  - higher-order pair classifier
  - solver-boundary diagnostic builder
- Data structures/models:
  - higher-order support record
  - refusal diagnostic record
- Dependencies/services:
  - CSG family capability matrix
  - patch family capability matrix
- Returns/outputs/signals:
  - explicit solver-boundary decision
  - unsupported result reason
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: family diagnostics
  - Additions to existing reusable library/module: higher-order CSG gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG diagnostics
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded support lookup
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- CSG support/diagnostic code

Routes:

- CSG family gate to intersection dispatch

Reuse/extraction decision:

- reuse Surface Spec 218 family support records

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- advanced pairs refuse unless an exact solver is explicitly declared

Data ownership:

- CSG owns operation support decisions

## Behavior

The implementation must:

- name every advanced-family intersection pair as supported, unsupported, or
  future solver work
- return explicit diagnostics with required future capability
- avoid approximate mesh/sampling fallback for unsupported pairs
- leave room for future exact solvers without changing the public refusal shape

## Verification

Test strategy:

- support-matrix tests for torus, spline, sweep, subdivision, implicit,
  heightmap, and displacement family pairs

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
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
- Total: 16.5

Split decision:

- Review for split. Cohesion reason: this is one advanced-family refusal boundary.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

