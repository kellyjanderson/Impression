# Surface Spec 142: NURBS Surface Patch (v1.0)

## Overview

Define rational NURBS surface payload and evaluation as an extension over the shared B-spline basis infrastructure.

## Backlink

- [Architecture: Full Surface Patch Family Architecture](../architecture/full-surface-patch-family-architecture.md)

## Scope

This specification promotes the manifest candidate `NURBS Surface Patch` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - NURBS patch constructor
  - rational evaluation
  - weighted derivative evaluation
- Data structures/models:
  - `NURBSSurfacePatch`
  - weight grid
  - rational control net view
- Dependencies/services:
  - B-spline basis utilities
  - surface patch base contract
  - tessellation
- Returns/outputs/signals:
  - evaluated point/derivative
  - weight validation diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: B-spline basis utilities
  - Additions to existing reusable library/module: surface patch family module
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds new patch family
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - rational evaluation should share cached basis values
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- patch family module selected by implementation spec

Routes:

- `SurfacePatch` API to rational evaluator to tessellation

Reuse/extraction decision:

- reuse B-spline basis; add family class

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- NURBS is a separate patch class over shared B-spline basis utilities

Data ownership:

- patch payload owns knots/control net/weights

## Behavior

The implementation must:

- satisfy every function, data-structure, dependency, and output responsibility listed above
- preserve the architecture boundary named in the backlink
- reject unsupported or ambiguous states with explicit diagnostics rather than silent fallback behavior
- keep mesh data outside canonical authored-surface state unless this spec explicitly names a tessellation, compatibility, or mesh-utility boundary
- expose only the public API surface needed by downstream specs and tests

## Constraints

- The implementation must remain deterministic for equivalent inputs.
- The implementation must keep metadata and stable identity behavior explicit when the leaf touches persisted or reusable surface state.
- The implementation must not introduce hidden mesh execution in authored modeling paths.
- The implementation must not broaden industry interchange, patch-family, or mesh compatibility scope beyond what this leaf names.

## Verification

Test strategy:

- unit tests for rational validation, evaluation, derivative behavior,
    tessellation, and `.impress` payload

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Split decision:

- Review for split.
- Cohesion reason: rational payload and rational evaluation are inseparable for
  a usable NURBS patch.

Open questions / nuance resolved for implementation:

- Weight zero/negative policy must be explicit; default is reject non-positive
  weights unless a later spec allows signed rational forms.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- all manifest responsibilities are implemented or explicitly refused by the leaf
- owner/module, routes, data ownership, reuse, UI inventory, defaults, and test strategy are represented in code or verification artifacts
- related progression items can be checked without relying on unstated architecture assumptions
- downstream specs can cite this leaf instead of re-reading the manifest candidate
