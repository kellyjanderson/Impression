# Reference Review Spec 54: Preview Payload Failure Diagnostic Handoff (v1.0)

## Overview

Route current-request payload failures to preview-pane diagnostics without clearing newer successful previews.

## Backlink

- [Architecture: Reference Review Preview Payload Boundary Architecture](../architecture/reference-review-preview-payload-boundary-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Payload Failure Diagnostic Handoff`.
- Manifest score: 23.5

## Scope

This specification covers:

- Route current-request payload failures to preview-pane diagnostics without clearing newer successful previews.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - failure result decoder
  - diagnostic handoff
- Data structures/models:
  - payload diagnostic record
- Dependencies/services:
  - preview payload records
  - preview pane
- Returns/outputs/signals:
  - payload failed event
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - loading diagnostic
  - failure diagnostic
- Reusable code plan:
  - Existing code reused as-is:
    - async envelope pattern
  - Additions to existing reusable library/module:
    - async controller preview
      failure route
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - stale failures are ignored; current failures update UI on Qt thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics redact unsafe environment details
- Performance-sensitive behavior:
  - failure handoff does not touch renderer state
- Cross-screen reusable behavior:
  - failure state feeds review readiness

## Implementation Boundary

Owner/module:

- future preview route in async controller

Routes:

- controller failure event to handoff route to preview pane

Reuse/extraction decision:

- Existing code reused as-is:
  - async envelope pattern
- Additions to existing reusable library/module:
  - async controller preview
    failure route
- New reusable library/module to create:
  - none

UI field/control inventory:

- loading diagnostic
- failure diagnostic

## Data And Defaults

Chosen defaults / parameters:

- stale failures never clear current previews

Data ownership:

- preview pane owns visible diagnostic state

Open questions and resolved assumptions:

- none

Implementation prerequisites:

- preview payload records

## Behavior

The implementation must:

- Route current-request payload failures to preview-pane diagnostics without clearing newer successful previews.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- current failure, stale failure, and diagnostic redaction tests

Additional verification requirements:

- add focused unit coverage for the records, helpers, or routes introduced by this specification
- add regression coverage for stale-result, cancellation, diagnostic, or renderer-lifetime behavior when this leaf touches those paths
- ensure real `.impress` fixtures remain the preferred manual smoke path for interactive preview behavior

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:

- No split needed. Cohesive current-failure diagnostic handoff leaf.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Payload Failure Diagnostic Handoff boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
