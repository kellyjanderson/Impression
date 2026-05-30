# Surface Spec 226: CSG Curve Primitive And Tolerance Policy (v1.0)

## Overview

Define reusable curve primitives and numeric tolerance rules for surface-native
CSG intersection, trimming, classification, and reconstruction.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification extracts the curve and tolerance foundation hidden inside
the CSG analytic intersection and trim-fragment specs.

## Responsibilities

- Functions/methods:
  - curve primitive constructors
  - curve equality and hashing helpers
  - tolerance normalization helper
- Data structures/models:
  - CSG curve primitive record
  - CSG tolerance policy record
- Dependencies/services:
  - surface patch evaluators
  - CSG intersection and trim stages
- Returns/outputs/signals:
  - deterministic curve payloads
  - explicit tolerance diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface evaluator coordinate conventions
  - Additions to existing reusable library/module: CSG-private curve/tolerance helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG internal representation
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - curve comparison must stay bounded and deterministic
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- CSG-private curve and tolerance helpers

Routes:

- intersection discovery to trim mapping, classification, and reconstruction

Reuse/extraction decision:

- keep private to CSG until another surface kernel operation requires it

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- one CSG tolerance policy governs snapping, equality, degeneracy, and
  diagnostic thresholds

Data ownership:

- CSG owns curve records and tolerance policy

## Behavior

The implementation must:

- represent line, circle/arc, conic, and sampled fallback curve records without
  storing mesh triangles as truth
- make all equality, deduplication, and ordering deterministic
- expose tolerance diagnostics when curves are too short, coincident,
  ambiguous, or outside patch domains
- avoid operation-specific tolerance constants scattered across CSG stages

## Verification

Test strategy:

- unit tests for curve canonical payloads, ordering, hashing, tolerance
  decisions, and degenerate diagnostics

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

- Review for split. Cohesion reason: curve records and tolerance policy are one
  shared CSG foundation.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

