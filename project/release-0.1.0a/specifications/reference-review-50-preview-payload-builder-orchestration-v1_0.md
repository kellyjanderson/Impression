# Reference Review Spec 50: Preview Payload Builder Orchestration (v1.0)

## Overview

Orchestrate non-UI payload creation without importing Qt widgets or mutating live renderer state.

## Backlink

- [Architecture: Reference Review Preview Payload Boundary Architecture](../architecture/reference-review-preview-payload-boundary-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Payload Builder Orchestration`.
- Manifest score: 18.5

## Scope

This specification covers:

- Orchestrate non-UI payload creation without importing Qt widgets or mutating live renderer state.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - payload builder
- Data structures/models:
  - `PreviewPayloadRequest`
  - `PreviewPayload`
- Dependencies/services:
  - source-load/tessellation adapter
  - payload serialization writer
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
    - none
  - New reusable library/module to create:
    - preview payload builder module
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - runs outside UI thread and returns immutable/file-backed payloads
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - coordinates payload work without touching UI thread
- Cross-screen reusable behavior:
  - feeds workbench preview and later artifact comparison readiness

## Implementation Boundary

Owner/module:

- future preview payload builder module

Routes:

- payload request to builder to payload result

Reuse/extraction decision:

- Existing code reused as-is:
  - fixture source records
- Additions to existing reusable library/module:
  - none
- New reusable library/module to create:
  - preview payload builder module

UI field/control inventory:

- none

## Data And Defaults

Chosen defaults / parameters:

- builder imports no workbench UI shell or PySide widget modules

Data ownership:

- builder owns payload creation only

Open questions and resolved assumptions:

- none

Implementation prerequisites:

- payload request/result records

## Behavior

The implementation must:

- Orchestrate non-UI payload creation without importing Qt widgets or mutating live renderer state.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- import-boundary tests and mocked adapter/writer orchestration tests

Additional verification requirements:

- add focused unit coverage for the records, helpers, or routes introduced by this specification
- add regression coverage for stale-result, cancellation, diagnostic, or renderer-lifetime behavior when this leaf touches those paths
- ensure real `.impress` fixtures remain the preferred manual smoke path for interactive preview behavior

## Manifest Assessment

Score:

- Functions/methods: 1 x 2 = 2
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 18.5

Split decision:

- No split needed. Cohesive orchestration leaf; source loading and payload
  serialization are separate candidates.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Payload Builder Orchestration boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
