# Surface Spec 305: Displacement Source Payload And Impress Identity Policy (v1.0)

## Overview

Define displacement source-surface persistence and identity rules without serializing derived mesh output.

## Backlink

- [Architecture: Advanced Patch Family Implementation Completion Architecture](../architecture/advanced-patch-family-implementation-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Displacement Source Payload And Impress Identity Policy` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, operation rows, and evidence named by the
  owning manifest candidate
- the exact implementation boundary required by the owning architecture
- deterministic refusal behavior for unsupported, unsafe, unavailable, or
  under-evidenced states

## Responsibilities

- Functions/methods:
  - displacement payload encoder
  - displacement payload decoder
  - source patch identity validator
- Data structures/models:
  - source patch reference record
  - displacement payload version record
  - identity diagnostic
- Dependencies/services:
  - source patch codec dispatch
  - SurfaceBodyStore identity policy
- Returns/outputs/signals:
  - round-trip displacement patch
  - identity/refusal diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current patch payload dispatch
  - Additions to existing reusable library/module: recursive source patch codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - file-format fixture writes in tests
- Security/privacy-sensitive behavior:
  - external references explicitly refused unless identity policy supports them
- Performance-sensitive behavior:
  - bounded embedded source payload
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`, `src/impression/io/impress.py`

Routes:

- displacement patch to `.impress` writer to reader

Reuse/extraction decision:

- reuse patch codec recursively

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- embedded source patch payload first

Data ownership:

- displacement patch owns source relationship; `.impress` owns payload identity

Open questions and resolved assumptions:

- Cross-body source references are explicitly refused by this candidate.
  Embedded source payloads and stable in-body source identities are the
  canonical implemented payload.

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

- embedded source round trip, missing source refusal, identity preservation

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
- Data structures/models: 3 x 1 = 3
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
- Total: 22.5

Split decision:

- Review for split. Cohesion reason: source identity and serialization are one
  persistence boundary.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Displacement Source Payload And Impress Identity Policy` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers,
  diagnostics, operation rows, or evidence gates
- unsupported, unsafe, unavailable, or non-applicable cases fail with
  deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired
  test specification created from this leaf
