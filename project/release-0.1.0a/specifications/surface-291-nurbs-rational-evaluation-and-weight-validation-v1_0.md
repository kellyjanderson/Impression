# Surface Spec 291: NURBS Rational Evaluation And Weight Validation (v1.0)

## Overview

Implement rational homogeneous evaluation, derivative correction, and weight validation on top of shared spline basis utilities.

## Backlink

- [Architecture: Advanced Patch Family Implementation Completion Architecture](../architecture/advanced-patch-family-implementation-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `NURBS Rational Evaluation And Weight Validation` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, operation rows, and evidence named by the
  owning manifest candidate
- the exact implementation boundary required by the owning architecture
- deterministic refusal behavior for unsupported, unsafe, unavailable, or
  under-evidenced states

## Responsibilities

- Functions/methods:
  - rational evaluator
  - rational derivative evaluator
  - weight validator
- Data structures/models:
  - weight validation diagnostic
  - rational evaluation metadata
- Dependencies/services:
  - shared spline basis utilities
  - NURBS patch payload
- Returns/outputs/signals:
  - rational point/derivative values
  - malformed weight diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `NURBSSurfacePatch`
  - Additions to existing reusable library/module: rational evaluation helper
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
  - bounded rational evaluation
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`

Routes:

- NURBS patch evaluator to shared spline basis to rational helper

Reuse/extraction decision:

- keep rational layer separate from non-rational B-spline evaluator

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- finite positive weights; reject zero, negative, non-finite, and shape-mismatched weights

Data ownership:

- NURBS patch owns weights; rational helper owns homogeneous evaluation

Open questions and resolved assumptions:

- Exact conic producer helpers are split into their own manifest candidate so
  rational evaluation does not hide construction-helper work.

Implementation prerequisites:

- shared spline basis utilities

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

- rational point, derivative, malformed weights, and conic metadata tests

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
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 17.5

Split decision:

- Review for split. Cohesion reason: this candidate owns only rational math and
  validation.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `NURBS Rational Evaluation And Weight Validation` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers,
  diagnostics, operation rows, or evidence gates
- unsupported, unsafe, unavailable, or non-applicable cases fail with
  deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired
  test specification created from this leaf
