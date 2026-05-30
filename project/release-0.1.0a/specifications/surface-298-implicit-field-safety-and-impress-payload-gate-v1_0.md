# Surface Spec 298: Implicit Field Safety And Impress Payload Gate (v1.0)

## Overview

Make implicit payloads safe, declarative, and persistable without executable callbacks or unbounded field definitions.

## Backlink

- [Architecture: Advanced Patch Family Implementation Completion Architecture](../architecture/advanced-patch-family-implementation-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Implicit Field Safety And Impress Payload Gate` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, operation rows, and evidence named by the
  owning manifest candidate
- the exact implementation boundary required by the owning architecture
- deterministic refusal behavior for unsupported, unsafe, unavailable, or
  under-evidenced states

## Responsibilities

- Functions/methods:
  - implicit field validator
  - implicit payload encoder
  - implicit payload decoder
- Data structures/models:
  - field safety policy record
  - unsafe payload diagnostic
- Dependencies/services:
  - implicit field node allow-list
  - `.impress` patch payload dispatch
- Returns/outputs/signals:
  - safe payload round-trip
  - unsafe payload refusal
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current implicit field node records
  - Additions to existing reusable library/module: implicit `.impress` safety gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - file-format fixture writes in tests
- Security/privacy-sensitive behavior:
  - executable field payload refusal
- Performance-sensitive behavior:
  - bounded payload size
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`, `src/impression/io/impress.py`

Routes:

- implicit patch to `.impress` writer to reader to implicit patch

Reuse/extraction decision:

- extend existing implicit payload validation

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- allow-listed declarative field nodes only

Data ownership:

- implicit patch owns field graph; `.impress` owns serialized form

Open questions and resolved assumptions:

- External field references are explicitly refused by this candidate. Embedded,
  allow-listed declarative field graphs are the canonical implemented payload.

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

- safe round-trip, malformed payload, callback refusal, and determinism tests

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
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Split decision:

- Review for split. Cohesion reason: this candidate owns only safe implicit
  persistence, not extraction or CSG.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Implicit Field Safety And Impress Payload Gate` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers,
  diagnostics, operation rows, or evidence gates
- unsupported, unsafe, unavailable, or non-applicable cases fail with
  deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired
  test specification created from this leaf
