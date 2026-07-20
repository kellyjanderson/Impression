# Surface Spec 292: NURBS Exact Conic Producer Helpers (v1.0)

## Overview

Implement `NURBS Exact Conic Producer Helpers` as a final surface-body availability or patch-family completion leaf.

## Backlink

- [Architecture: Advanced Patch Family Implementation Completion Architecture](../architecture/advanced-patch-family-implementation-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `NURBS Exact Conic Producer Helpers` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, operation rows, and evidence named by the
  owning manifest candidate
- the exact implementation boundary required by the owning architecture
- deterministic refusal behavior for unsupported, unsafe, unavailable, or
  under-evidenced states

## Responsibilities

- Functions/methods:
  - exact circular arc rational control-net builder
  - exact ellipse/conic profile helper for rational section construction
  - helper for producing NURBS patch/profile payloads with conic provenance
- Data structures/models:
  - conic construction request
  - conic construction diagnostic
- Dependencies/services:
  - shared spline basis utilities
  - NURBS rational evaluation helper
- Returns/outputs/signals:
  - NURBS-compatible control points, weights, knots, and metadata
  - validation diagnostics for unsupported conic requests
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: rational evaluator and knot validation helpers
  - Additions to existing reusable library/module: conic helper functions in
    surface modeling
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
  - constant-size conic helper construction
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`

Routes:

- conic helper to NURBS rational evaluator to `.impress` codec

Reuse/extraction decision:

- keep helpers on top of the rational evaluator; do not duplicate basis math

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- support exact circular arcs and ellipses first; unsupported conic kinds
  return structured diagnostics rather than approximate silently

Data ownership:

- helper owns construction metadata; resulting patch/profile owns the
  generated rational payload

Open questions and resolved assumptions:

- Hyperbola and parabola helpers are explicit unsupported diagnostics unless a
  later manifest candidate promotes them; circle and ellipse cover the current
  exact-conic completion gap.

Implementation prerequisites:

- shared spline basis utilities
- NURBS rational evaluation helper

## Behavior

The implementation must:

- satisfy every responsibility listed above with explicit records, helpers,
  diagnostics, or operation-matrix entries
- preserve authored surface truth and never use mesh as a hidden fallback
- keep unavailable, unsupported, unsafe, or non-applicable states explicit and
  inspectable
- make readiness and availability evidence deterministic enough for release
  progression and future completion reports

## Verification

Test strategy:

- exact circle/ellipse helper tests, malformed request tests, and NURBS
  round-trip metadata tests

Additional verification requirements:

- add focused unit coverage for each new record, helper, diagnostic, and matrix
  row introduced by this leaf
- add negative coverage for malformed, unsupported, unsafe, missing-evidence, or
  non-applicable states named by this leaf
- include no-hidden-mesh-fallback assertions where the leaf touches authoring,
  operation selection, CSG, seams, tessellation, or reference evidence
- update reference or diagnostic fixtures when this leaf changes visible model
  output or durable refusal behavior

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 0 x 2 = 0
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 2 x 2 = 4
- Total: 21

Split decision:

- Review for split. Cohesion reason: circle and ellipse exact-conic helpers
  share one rational construction pipeline and depend on the same NURBS
  evaluator; unsupported conic kinds are explicit diagnostics, not hidden
  approximation work.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `NURBS Exact Conic Producer Helpers` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers,
  diagnostics, operation rows, or evidence gates
- unsupported, unsafe, unavailable, or non-applicable cases fail with
  deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired
  test specification created from this leaf
