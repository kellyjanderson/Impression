# Surface Spec 289: Shared Spline Basis And Knot Infrastructure (v1.0)

## Overview

Build reusable spline basis, knot validation, derivative, and control-net canonicalization utilities shared by B-spline and NURBS surfaces.

## Backlink

- [Architecture: Advanced Patch Family Implementation Completion Architecture](../architecture/advanced-patch-family-implementation-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Shared Spline Basis And Knot Infrastructure` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, operation rows, and evidence named by the
  owning manifest candidate
- the exact implementation boundary required by the owning architecture
- deterministic refusal behavior for unsupported, unsafe, unavailable, or
  under-evidenced states

## Responsibilities

- Functions/methods:
  - knot vector validator
  - basis evaluator
  - basis derivative evaluator
  - control-net canonicalizer
- Data structures/models:
  - spline basis diagnostic
  - knot policy record
- Dependencies/services:
  - B-spline patch record
  - NURBS patch record
- Returns/outputs/signals:
  - evaluated basis values
  - validation diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current spline patch records
  - Additions to existing reusable library/module: shared spline utility layer
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
  - bounded basis evaluation and finite validation
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`

Routes:

- B-spline/NURBS patch evaluators to shared spline utility layer

Reuse/extraction decision:

- required shared utility, not family-local duplication

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- finite clamped/open knots first; periodic support behind explicit flags

Data ownership:

- shared spline utilities own basis math; patch classes own geometry payloads

Open questions and resolved assumptions:

- periodic knot support can be separate future scope if not needed for promotion.

Implementation prerequisites:

- none

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

- knot validation, basis values, derivative values, malformed control nets

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

- Functions/methods: 4 x 2 = 8
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
- Readiness blockers: 0 x 2 = 0
- Total: 17.5

Split decision:

- Review for split. Cohesion reason: this candidate owns only shared spline
  math and validation utilities.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Shared Spline Basis And Knot Infrastructure` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers,
  diagnostics, operation rows, or evidence gates
- unsupported, unsafe, unavailable, or non-applicable cases fail with
  deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired
  test specification created from this leaf
