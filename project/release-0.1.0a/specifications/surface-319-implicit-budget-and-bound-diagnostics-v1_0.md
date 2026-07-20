# Surface Spec 319: Implicit Budget And Bound Diagnostics (v1.0)

## Overview

Implement `Implicit Budget And Bound Diagnostics` as a final surface-body availability or patch-family completion leaf.

## Backlink

- [Architecture: Advanced Family Availability Producer Architecture](../architecture/advanced-family-availability-producer-architecture.md)

## Scope

This specification promotes the manifest candidate `Implicit Budget And Bound Diagnostics` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, operation rows, and evidence named by the
  owning manifest candidate
- the exact implementation boundary required by the owning architecture
- deterministic refusal behavior for unsupported, unsafe, unavailable, or
  under-evidenced states

## Responsibilities

- Functions/methods:
  - budget-refusal diagnostic builder
  - bounded-domain validator wrapper
  - extraction-budget locator
- Data structures/models:
  - budget diagnostic
  - bound diagnostic
- Dependencies/services:
  - implicit extraction budget policy
  - implicit evaluation/extraction adapter
- Returns/outputs/signals:
  - structured budget/bounds diagnostic
  - non-executable producer result
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: implicit budget policy
  - Additions to existing reusable library/module: authoring budget diagnostics
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - diagnostic fixture writes in tests
- Security/privacy-sensitive behavior:
  - avoids executing over-budget payloads
- Performance-sensitive behavior:
  - bounded tree and extraction budget validation
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`

Routes:

- builder to budget gate to refusal diagnostic

Reuse/extraction decision:

- reuse extraction budget policy rather than inventing builder-only limits

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- over-budget authoring returns diagnostics before extraction

Data ownership:

- producer diagnostics own budget and bound failure reasons

Open questions and resolved assumptions:

- Budget diagnostics must name the first exceeded limit and the family.

Implementation prerequisites:

- implicit extraction budget policy

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

- unbounded domain refusal, budget exhaustion, depth limit, node-count limit,
  and diagnostic payload tests

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
- Performance-sensitive behavior: 2 x 1 = 2
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:

- Review for split. Cohesion reason: this candidate owns bounded evaluation
  refusal and shares limits with extraction.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Implicit Budget And Bound Diagnostics` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers,
  diagnostics, operation rows, or evidence gates
- unsupported, unsafe, unavailable, or non-applicable cases fail with
  deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired
  test specification created from this leaf
