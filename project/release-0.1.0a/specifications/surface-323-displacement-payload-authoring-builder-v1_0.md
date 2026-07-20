# Surface Spec 323: Displacement Payload Authoring Builder (v1.0)

## Overview

Implement `Displacement Payload Authoring Builder` as a final surface-body availability or patch-family completion leaf.

## Backlink

- [Architecture: Advanced Family Availability Producer Architecture](../architecture/advanced-family-availability-producer-architecture.md)

## Scope

This specification promotes the manifest candidate `Displacement Payload Authoring Builder` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, operation rows, and evidence named by the
  owning manifest candidate
- the exact implementation boundary required by the owning architecture
- deterministic refusal behavior for unsupported, unsafe, unavailable, or
  under-evidenced states

## Responsibilities

- Functions/methods:
  - displacement patch builder
  - displacement payload validator wrapper
  - displacement payload diagnostic builder
- Data structures/models:
  - displacement authoring request
  - displacement producer diagnostic
  - lossiness metadata record
- Dependencies/services:
  - displacement payload validation
  - displacement source identity resolver
- Returns/outputs/signals:
  - native displacement patch
  - source identity and lossiness metadata
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: displacement patch record/evaluator
  - Additions to existing reusable library/module: displacement builder
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference fixture writes in tests
- Security/privacy-sensitive behavior:
  - callable displacement functions are refused unless safe policy supports them
- Performance-sensitive behavior:
  - bounded displacement sampling/evaluation defaults
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`

Routes:

- builder to source resolver to displacement patch to codec/tessellation/seams

Reuse/extraction decision:

- reuse source patch evaluation and displacement validator

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- sampled displacement payload first; persisted callable displacement refused
  unless safety policy supports it

Data ownership:

- builder owns construction metadata; patch owns displacement payload

Open questions and resolved assumptions:

- General callable displacement functions need the same safety posture as
  implicit fields if they are ever persisted.

Implementation prerequisites:

- displacement source identity resolver
- displacement evaluation/tessellation specs

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

- sampled displacement, invalid displacement data, callable refusal,
  `.impress` round-trip, and no-hidden-mesh-fallback tests

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
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 2 x 2 = 4
- Total: 24.5

Split decision:

- Review for split. Cohesion reason: this candidate owns authoring and payload
  validation after source identity exists.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Displacement Payload Authoring Builder` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers,
  diagnostics, operation rows, or evidence gates
- unsupported, unsafe, unavailable, or non-applicable cases fail with
  deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired
  test specification created from this leaf
