# Surface Spec 219: Surface CSG Analytic Patch Intersection Library (v1.0)

## Overview

Add exact intersection primitives for analytic patch families so common CSG
work does not stop at box/box or planar-only scope.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface CSG Analytic Patch
Intersection Library` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - plane/plane intersection
  - plane/cylinder and plane/cone intersection
  - plane/sphere and plane/torus intersection
  - cylinder/cylinder initial intersection
- Data structures/models:
  - surface intersection curve record
  - patch-local curve mapping record
- Dependencies/services:
  - analytic patch evaluators
  - CSG classifier
- Returns/outputs/signals:
  - exact curve segments
  - unsupported/degenerate diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: analytic patch evaluation
  - Additions to existing reusable library/module: CSG intersection helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG kernel behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded pairwise patch intersection
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/csg.py` or a CSG-private helper module

Routes:

- prepared operands to intersection discovery

Reuse/extraction decision:

- keep helpers CSG-private until reused by another kernel operation

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- exact analytic curves are preferred; ambiguous degeneracies refuse

Data ownership:

- CSG owns intersection records; patch families own evaluation

## Behavior

The implementation must:

- compute exact curve records for the named analytic pairs
- map every curve into participating patch-local coordinates
- refuse unsupported, tangential, or degenerate cases explicitly when exact
  classification would otherwise require guessing
- avoid mesh sampling or triangle clipping as CSG truth

## Constraints

- Spline, subdivision, sweep, and implicit intersections are not bundled here.
- The implementation must remain deterministic for equivalent inputs.
- Mesh output is permitted only after a surface result reaches tessellation.

## Verification

Test strategy:

- unit fixtures for each named analytic pair and degeneracy

Automated or review verification must prove exact records, deterministic
diagnostics, and absence of mesh boolean execution.

## Manifest Assessment

Score:

- Functions/methods: 4 x 2 = 8
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
- Total: 20.5

Split decision:

- Review for split. Cohesion reason: this is the analytic-only intersection set.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when the named analytic intersections produce
surface-native curve records or explicit diagnostics without mesh fallback.
