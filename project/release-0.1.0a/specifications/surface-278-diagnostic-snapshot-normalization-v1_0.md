# Surface Spec 278: Diagnostic Snapshot Normalization (v1.0)

## Overview

Normalize refusal diagnostics into stable snapshot payloads that ignore
incidental stack traces, temporary paths, and machine-specific details.

## Backlink

- [Architecture: Reference Artifact Promotion Architecture](../architecture/reference-artifact-promotion-architecture.md)

## Scope

This specification promotes the manifest candidate `Diagnostic Snapshot Normalization` into a final implementation leaf.

This specification covers:

- Normalize refusal diagnostics into stable snapshot payloads that ignore
  incidental stack traces, temporary paths, and machine-specific details.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - diagnostic snapshot normalizer
  - snapshot comparator
- Data structures/models:
  - diagnostic snapshot record
  - diagnostic key policy record
- Dependencies/services:
  - current refusal diagnostics
  - reference artifact lifecycle
- Returns/outputs/signals:
  - stable diagnostic snapshot
  - diagnostic drift failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current refusal diagnostics
  - Additions to existing reusable library/module: snapshot normalization
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty diagnostic snapshots during bootstrap
- Security/privacy-sensitive behavior:
  - snapshots must not include sensitive local paths beyond normalized fixture
    identifiers
- Performance-sensitive behavior:
  - bounded fixture count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- reference test helpers under `tests/`

Routes:

- refusal exception/result to normalized snapshot to comparator

Reuse/extraction decision:

- Existing code reused as-is: current refusal diagnostics
- Additions to existing reusable library/module: snapshot normalization
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- snapshots compare stable diagnostic keys, not incidental stack traces

Data ownership:

- snapshot normalizer owns portable diagnostic representation

Open questions and resolved assumptions:

- diagnostic snapshots should avoid machine-specific path fragments

Implementation prerequisites:

- none

## Behavior

The implementation must:

- Normalize refusal diagnostics into stable snapshot payloads that ignore
  incidental stack traces, temporary paths, and machine-specific details.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- normalization tests for path stripping, ordering, and stable keys

Additional verification requirements:

- add focused unit coverage for the records and helpers introduced by this specification
- add regression coverage for deterministic diagnostic text or diagnostic keys
- ensure reference artifact behavior is updated when this leaf changes model-output evidence

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Split decision:

- Review for split. Cohesion reason: normalization is one reusable snapshot
  layer; domain-specific negative fixtures are split separately.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Diagnostic Snapshot Normalization` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
