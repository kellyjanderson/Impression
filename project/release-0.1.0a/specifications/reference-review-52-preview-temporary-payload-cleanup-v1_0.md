# Reference Review Spec 52: Preview Temporary Payload Cleanup (v1.0)

## Overview

Clean file-backed payload artifacts for completed, cancelled, or stale preview requests.

## Backlink

- [Architecture: Reference Review Preview Payload Boundary Architecture](../architecture/reference-review-preview-payload-boundary-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Temporary Payload Cleanup`.
- Manifest score: 24

## Scope

This specification covers:

- Clean file-backed payload artifacts for completed, cancelled, or stale preview requests.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - temporary payload cleanup
  - cancelled request cleanup
- Data structures/models:
  - payload file metadata
- Dependencies/services:
  - preview payload controller
- Returns/outputs/signals:
  - cleanup diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is:
    - none
  - Additions to existing reusable library/module:
    - preview payload controller
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - cleanup must not race current payload ownership
- Destructive/write behavior:
  - deletes only controller-owned temporary payload files
- Security/privacy-sensitive behavior:
  - cleanup diagnostics avoid unsafe full environment details
- Performance-sensitive behavior:
  - cleanup is bounded and never blocks UI rendering
- Cross-screen reusable behavior:
  - protects repeated preview selection

## Implementation Boundary

Owner/module:

- future preview payload controller cleanup path

Routes:

- controller lifecycle to cleanup path

Reuse/extraction decision:

- Existing code reused as-is:
  - none
- Additions to existing reusable library/module:
  - preview payload controller
- New reusable library/module to create:
  - none

UI field/control inventory:

- none

## Data And Defaults

Chosen defaults / parameters:

- cleanup ignores files not owned by the active preview controller

Data ownership:

- controller owns cleanup for its temporary payload files

Open questions and resolved assumptions:

- none

Implementation prerequisites:

- payload file metadata

## Behavior

The implementation must:

- Clean file-backed payload artifacts for completed, cancelled, or stale preview requests.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- completed, cancelled, stale, and missing-file cleanup tests

Additional verification requirements:

- add focused unit coverage for the records, helpers, or routes introduced by this specification
- add regression coverage for stale-result, cancellation, diagnostic, or renderer-lifetime behavior when this leaf touches those paths
- ensure real `.impress` fixtures remain the preferred manual smoke path for interactive preview behavior

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 0 x 0.5 = 0
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 24

Split decision:

- No split needed. Cohesive cleanup leaf with one destructive boundary.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Temporary Payload Cleanup boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
