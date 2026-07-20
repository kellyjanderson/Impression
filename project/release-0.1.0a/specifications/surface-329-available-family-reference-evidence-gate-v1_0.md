# Surface Spec 329: Available-Family Reference Evidence Gate (v1.0)

## Overview

Implement `Available-Family Reference Evidence Gate` as a final surface-body availability or patch-family completion leaf.

## Backlink

- [Architecture: Advanced Family Availability Producer Architecture](../architecture/advanced-family-availability-producer-architecture.md)

## Scope

This specification promotes the manifest candidate `Available-Family Reference Evidence Gate` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, operation rows, and evidence named by the
  owning manifest candidate
- the exact implementation boundary required by the owning architecture
- deterministic refusal behavior for unsupported, unsafe, unavailable, or
  under-evidenced states

## Responsibilities

- Functions/methods:
  - reference evidence collector
  - promoted evidence gate checker
  - dirty-artifact refusal diagnostic builder
- Data structures/models:
  - reference evidence summary
  - missing-reference diagnostic
- Dependencies/services:
  - reference artifact promotion gates
  - operation completeness verifier
- Returns/outputs/signals:
  - reference evidence pass/fail result
  - missing/dirty artifact diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: reference gates
  - Additions to existing reusable library/module: availability reference gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference/evidence fixture writes in tests
- Security/privacy-sensitive behavior:
  - preserves refusal reasons without executing unsafe payloads
- Performance-sensitive behavior:
  - bounded evidence scan
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`

Routes:

- operation verifier to reference gates

Reuse/extraction decision:

- reuse reference artifact promotion records

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- dirty artifacts never count as availability evidence

Data ownership:

- reference system owns artifacts; gate owns evidence result

Open questions and resolved assumptions:

- Negative diagnostic references should count only for refusal-path evidence,
  not positive model-output evidence.

Implementation prerequisites:

- operation matrix completeness verifier
- reference artifact promotion gates

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

- missing reference, dirty reference, promoted reference, and diagnostic
  reference tests

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
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 2 x 2 = 4
- Total: 23.5

Split decision:

- Review for split. Cohesion reason: this candidate owns evidence acceptance
  rules and reference artifact lifecycle integration.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Available-Family Reference Evidence Gate` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers,
  diagnostics, operation rows, or evidence gates
- unsupported, unsafe, unavailable, or non-applicable cases fail with
  deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired
  test specification created from this leaf
