# Surface Spec 270: Analytic To Spline Surface Intersection Solvers (v1.0)

## Overview

Implement exact or declared-tolerance intersections for planar/ruled/
revolution surfaces against B-spline and NURBS surfaces.

## Backlink

- [Architecture: Exact Surface Intersection Kernel Architecture](../architecture/exact-surface-intersection-kernel-architecture.md)

## Scope

This specification promotes the manifest candidate `Analytic To Spline Surface Intersection Solvers` into a final implementation leaf.

This specification covers:

- Implement exact or declared-tolerance intersections for planar/ruled/
  revolution surfaces against B-spline and NURBS surfaces.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - analytic-spline solver
  - Newton/refinement routine
  - analytic seed generator
- Data structures/models:
  - solver iteration record
  - residual report
- Dependencies/services:
  - B-spline/NURBS evaluators
  - registry and result records
- Returns/outputs/signals:
  - intersection curve records
  - non-convergence diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: spline evaluation helpers
  - Additions to existing reusable library/module: solver routines
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - iteration and subdivision budgets
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface_intersections.py`

Routes:

- registry dispatch to solver to result record

Reuse/extraction decision:

- Existing code reused as-is: spline evaluation helpers
- Additions to existing reusable library/module: solver routines
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- deterministic seeds; bounded iteration; residual metadata required

Data ownership:

- solver owns numeric process; result record owns reusable curve truth

Open questions and resolved assumptions:

- revolution/spline seeding may need conic-aware initial guesses

Implementation prerequisites:

- spline derivative/evaluation coverage must be verified

## Behavior

The implementation must:

- Implement exact or declared-tolerance intersections for planar/ruled/
  revolution surfaces against B-spline and NURBS surfaces.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- planar/spline, ruled/spline, revolution/spline positive and refusal tests

Additional verification requirements:

- add focused unit coverage for the records and helpers introduced by this specification
- add regression coverage for deterministic diagnostic text or diagnostic keys
- ensure reference artifact behavior is updated when this leaf changes model-output evidence

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 17.5

Split decision:

- Review for split. Cohesion reason: analytic/spline intersections share
  analytic seeding and spline projection; spline/spline is split separately.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Analytic To Spline Surface Intersection Solvers` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
