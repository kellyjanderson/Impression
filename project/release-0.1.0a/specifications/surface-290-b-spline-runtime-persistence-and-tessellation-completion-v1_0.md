# Surface Spec 290: B-spline Runtime Persistence And Tessellation Completion (v1.0)

## Overview

Complete B-spline patch evaluation, boundary extraction, tessellation, `.impress`, seams, CSG matrix participation, loft output, diagnostics, and evidence using shared spline utilities.

## Backlink

- [Architecture: Advanced Patch Family Implementation Completion Architecture](../architecture/advanced-patch-family-implementation-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `B-spline Runtime Persistence And Tessellation Completion` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, operation rows, and evidence named by the
  owning manifest candidate
- the exact implementation boundary required by the owning architecture
- deterministic refusal behavior for unsupported, unsafe, unavailable, or
  under-evidenced states

## Responsibilities

- Functions/methods:
  - B-spline evaluator
  - B-spline boundary extractor
  - B-spline tessellation adapter
- Data structures/models:
  - B-spline patch payload
  - B-spline promotion evidence record
- Dependencies/services:
  - shared spline basis utilities
  - `.impress` codec
  - CSG and intersection registries
- Returns/outputs/signals:
  - implemented B-spline readiness
  - malformed payload diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `BSplineSurfacePatch`
  - Additions to existing reusable library/module: B-spline family adapters
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - capability matrix update after evidence
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - adaptive tessellation bounds
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`,
  `src/impression/io/impress.py`

Routes:

- loft/authoring to B-spline patch to store/codec/tessellation/CSG

Reuse/extraction decision:

- consume shared spline utilities

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- non-rational tensor-product B-spline using shared finite knot policy

Data ownership:

- B-spline patch owns geometry payload

Open questions and resolved assumptions:

- smooth loft producer may be its own spec if it affects loft planning broadly.

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

- evaluation, boundary, tessellation, `.impress`, seam, CSG matrix, loft producer

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
- Readiness blockers: 1 x 2 = 2
- Total: 21.5

Split decision:

- Review for split. Cohesion reason: this candidate is one family adapter layer
  after shared spline utilities exist.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `B-spline Runtime Persistence And Tessellation Completion` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers,
  diagnostics, operation rows, or evidence gates
- unsupported, unsafe, unavailable, or non-applicable cases fail with
  deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired
  test specification created from this leaf
