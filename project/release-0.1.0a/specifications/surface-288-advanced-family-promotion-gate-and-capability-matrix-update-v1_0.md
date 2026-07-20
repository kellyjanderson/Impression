# Surface Spec 288: Advanced Family Promotion Gate And Capability Matrix Update (v1.0)

## Overview

Define the code gate that moves B-spline, NURBS, sweep, subdivision, implicit, heightmap, and displacement from `planned` to `implemented` only when evidence exists.

## Backlink

- [Architecture: Advanced Patch Family Implementation Completion Architecture](../architecture/advanced-patch-family-implementation-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Advanced Family Promotion Gate And Capability Matrix Update` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, operation rows, and evidence named by the
  owning manifest candidate
- the exact implementation boundary required by the owning architecture
- deterministic refusal behavior for unsupported, unsafe, unavailable, or
  under-evidenced states

## Responsibilities

- Functions/methods:
  - promotion gate evaluator
  - capability matrix updater/assertion helper
- Data structures/models:
  - advanced family promotion gate record
  - promotion evidence record
- Dependencies/services:
  - `PATCH_FAMILY_CAPABILITY_MATRIX`
  - promotion readiness audit
  - completion evidence gate
- Returns/outputs/signals:
  - promotion report
  - blocking diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current capability and readiness records
  - Additions to existing reusable library/module: surface completion gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - updates release capability truth
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded matrix evaluation
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`

Routes:

- readiness audit to capability matrix to completion gate

Reuse/extraction decision:

- extend existing gate records rather than adding a second matrix

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- no family phase changes without evidence

Data ownership:

- capability matrix owns family phase truth

Open questions and resolved assumptions:

- `implemented` and `available` may need separate matrix fields.

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

- promotion gate tests for passing and missing-evidence states

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

- Functions/methods: 2 x 2 = 4
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
- Readiness blockers: 0 x 2 = 0
- Total: 17.5

Split decision:

- Review for split. Cohesion reason: this spec owns only the cross-family gate,
  not the family implementations.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Advanced Family Promotion Gate And Capability Matrix Update` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers,
  diagnostics, operation rows, or evidence gates
- unsupported, unsafe, unavailable, or non-applicable cases fail with
  deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired
  test specification created from this leaf
