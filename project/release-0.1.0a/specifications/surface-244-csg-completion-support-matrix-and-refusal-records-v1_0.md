# Surface Spec 244: CSG Completion Support Matrix And Refusal Records (v1.0)

## Overview

Define the authoritative CSG family-pair support matrix and refusal record
schema for unsupported or staged boolean combinations.

## Backlink

- [Architecture: Surface Body Completion Architecture](../architecture/surface-body-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `CSG Completion Support
Matrix And Refusal Records` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - CSG support matrix resolver
  - refusal record factory
  - family-pair diagnostic formatter
- Data structures/models:
  - CSG family-pair support record
  - CSG refusal record
  - operation support phase record
- Dependencies/services:
  - SurfaceBody CSG architecture
  - patch family capability matrix
  - no-hidden-mesh-fallback enforcement
- Returns/outputs/signals:
  - support verdict
  - structured refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG unsupported result posture
  - Additions to existing reusable library/module: CSG capability/refusal table
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes unsupported CSG reporting posture
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - constant-time matrix lookup
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/csg.py`

Routes:

- boolean API to support matrix to executor or refusal result

Reuse/extraction decision:

- extend CSG result diagnostics rather than adding caller-specific checks

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- unsupported family pairs refuse explicitly and never route to mesh

Data ownership:

- CSG owns operation support truth
- patch families own family availability

## Behavior

The implementation must:

- classify every promoted family pair for union, difference, and intersection
- distinguish exact, declared-tolerance, adapter, unsupported, and not-yet
  implemented states
- return structured refusals with operation, family pair, support phase, and
  required next dependency
- keep the matrix separate from caller inventory so producers do not duplicate
  CSG policy

## Verification

Test strategy:

- family-pair support matrix tests
- refusal record tests
- no-hidden-mesh-fallback tests for unsupported CSG pairs

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
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
- Total: 20.5

Split decision:

- Review for split. Cohesion reason: matrix and refusal records are one
  boundary contract used by all CSG solvers.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- every family pair has an operation support verdict
- unsupported verdicts produce structured refusals
- mesh is never selected as a substitute boolean result
