# Surface Spec 252: Primitive Patch Producer Selection (v1.0)

## Overview

Make primitives choose exact surface patch families and refuse unsupported
primitive producer requests without hidden mesh substitution.

## Backlink

- [Architecture: Surface Body Completion Architecture](../architecture/surface-body-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Primitive Patch Producer
Selection` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - primitive family selector
  - unsupported primitive diagnostic
- Data structures/models:
  - primitive producer selection record
  - unsupported primitive producer result
- Dependencies/services:
  - primitive surface constructors
  - no-hidden-mesh-fallback gate
- Returns/outputs/signals:
  - selected patch family
  - surface truth result
  - explicit unsupported diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: primitive modules
  - Additions to existing reusable library/module: primitive producer selection
    helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes primitive authored output routing
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - selection should be deterministic and constant-time per primitive
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- primitive constructors

Routes:

- public primitive API to producer selector to SurfaceBody output

Reuse/extraction decision:

- share primitive selection helper across primitive constructors

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- exact analytic families are preferred when available
- mesh is never a hidden substitute

Data ownership:

- primitives own authored family choice
- surface body owns emitted geometry

## Behavior

The implementation must:

- route box, prism, polyhedron, cylinder, cone, ngon, sphere, torus, and other
  primitive constructors to appropriate surface families
- prefer exact analytic families over sampled approximations when exact
  families exist
- return explicit unsupported diagnostics for primitive requests that cannot
  produce surface truth
- never call mesh construction as hidden substitute geometry

## Verification

Test strategy:

- producer selection tests for all public primitives
- no-hidden-mesh-fallback tests
- unsupported primitive diagnostic tests

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
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
- Total: 17.5

Split decision:

- Review for split. Cohesion reason: this is one primitive-family routing
  contract after feature-builder handoff was split out.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- primitives report their selected patch families
- unsupported primitive surface requests refuse explicitly
- hidden mesh substitute paths are covered by failing tests
