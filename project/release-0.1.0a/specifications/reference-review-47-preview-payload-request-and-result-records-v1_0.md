# Reference Review Spec 47: Preview Payload Request And Result Records (v1.0)

## Overview

Define the immutable records that describe a preview payload request, successful payload result, and payload failure diagnostic.

## Backlink

- [Architecture: Reference Review Preview Payload Boundary Architecture](../architecture/reference-review-preview-payload-boundary-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Payload Request And Result Records`.
- Manifest score: 23.5

## Scope

This specification covers:

- Define the immutable records that describe a preview payload request, successful payload result, and payload failure diagnostic.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - payload request factory
  - payload result factory
- Data structures/models:
  - `PreviewPayloadRequest`
  - `PreviewPayload`
  - payload diagnostic record
- Dependencies/services:
  - fixture source contract
- Returns/outputs/signals:
  - preview payload result
  - preview payload failure
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is:
    - fixture source records
  - Additions to existing reusable library/module:
    - preview payload record module
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - records carry owner, request id, fixture id, and generation metadata
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics redact unsafe environment details
- Performance-sensitive behavior:
  - records may reference file-backed payloads instead of embedding large data
- Cross-screen reusable behavior:
  - payload state feeds preview pane, artifact comparison, and promotion
    readiness

## Implementation Boundary

Owner/module:

- future `src/impression/devtools/reference_review/preview_payload.py`

Routes:

- selected fixture to request record to worker/controller result

Reuse/extraction decision:

- Existing code reused as-is:
  - fixture source records
- Additions to existing reusable library/module:
  - preview payload record module
- New reusable library/module to create:
  - none

UI field/control inventory:

- none

## Data And Defaults

Chosen defaults / parameters:

- records are immutable and include request identity

Data ownership:

- payload records own request/result data only

Open questions and resolved assumptions:

- exact file-backed payload format should be selected during specification
  refinement

Implementation prerequisites:

- shared preview controller must define the payload shape it can consume

## Behavior

The implementation must:

- Define the immutable records that describe a preview payload request, successful payload result, and payload failure diagnostic.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- record construction, serialization, and diagnostic redaction tests

Additional verification requirements:

- add focused unit coverage for the records, helpers, or routes introduced by this specification
- add regression coverage for stale-result, cancellation, diagnostic, or renderer-lifetime behavior when this leaf touches those paths
- ensure real `.impress` fixtures remain the preferred manual smoke path for interactive preview behavior

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:

- No split needed. Cohesion reason: request, result, and diagnostic records are
  one immutable payload contract; visible diagnostics are owned by pane state.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Payload Request And Result Records boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
