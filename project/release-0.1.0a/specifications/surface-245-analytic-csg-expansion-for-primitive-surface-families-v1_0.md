# Surface Spec 245: Analytic CSG Expansion For Primitive Surface Families (v1.0)

## Overview

Expand executable surface CSG coverage for planar, ruled, and revolution
primitive-derived families without broadening into higher-order spline or
sampled solvers.

## Backlink

- [Architecture: Surface Body Completion Architecture](../architecture/surface-body-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Analytic CSG Expansion For
Primitive Surface Families` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - planar/ruled/revolution intersection dispatch
  - primitive analytic pair handlers
  - tolerance-aware curve classification
- Data structures/models:
  - analytic intersection curve record
  - primitive pair solver record
  - tolerance policy record
- Dependencies/services:
  - CSG support matrix
  - primitive surface families
  - seam/adjacency rebuild
- Returns/outputs/signals:
  - analytic intersection records
  - unsupported analytic pair diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current bounded CSG result types
  - Additions to existing reusable library/module: analytic CSG solver library
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes boolean execution coverage
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded pair solver tolerances and iteration limits
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/csg.py` and analytic intersection helpers

Routes:

- support matrix to analytic solver to trim/result reconstruction

Reuse/extraction decision:

- keep solver helpers reusable under CSG; do not place pair logic in primitives

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- analytic primitive family pairs execute when the support matrix marks them
  supported

Data ownership:

- analytic solver owns intersection records
- CSG owns result topology

## Behavior

The implementation must:

- support the primitive-derived analytic pairs declared by the CSG support
  matrix
- produce curve records suitable for trim splitting and shell reconstruction
- apply declared tolerances consistently across pair handlers
- return unsupported analytic pair diagnostics for unsupported combinations
- avoid tessellating operands to perform authored CSG

## Verification

Test strategy:

- primitive pair boolean fixtures
- tolerance-bound negative tests
- no-hidden-mesh-fallback tests for analytic primitive CSG

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

- Review for split. Cohesion reason: this is intentionally limited to analytic
  primitive families; higher-order and sampled policies are separate specs.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- declared analytic primitive pairs execute as surface CSG
- unsupported pairs refuse with structured diagnostics
- no analytic primitive CSG path uses mesh as the modeled truth
