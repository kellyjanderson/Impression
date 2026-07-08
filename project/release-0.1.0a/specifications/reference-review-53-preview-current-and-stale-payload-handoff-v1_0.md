# Reference Review Spec 53: Preview Current And Stale Payload Handoff (v1.0)

## Overview

Decode current payload results and reject stale payload results before handing current payloads to the preview pane on the Qt UI thread.

## Backlink

- [Architecture: Reference Review Preview Payload Boundary Architecture](../architecture/reference-review-preview-payload-boundary-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Current And Stale Payload Handoff`.
- Manifest score: 20.5

## Scope

This specification covers:

- Decode current payload results and reject stale payload results before handing current payloads to the preview pane on the Qt UI thread.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - payload result decoder
  - owner/request match check
- Data structures/models:
  - active preview request state
- Dependencies/services:
  - preview payload records
  - preview pane
- Returns/outputs/signals:
  - payload ready event
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is:
    - async envelope pattern
  - Additions to existing reusable library/module:
    - async controller preview route
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - UI-thread handoff and request id matching
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - no tessellation or source loading during handoff
- Cross-screen reusable behavior:
  - handoff state feeds review readiness

## Implementation Boundary

Owner/module:

- future preview route in async controller

Routes:

- controller event to handoff route to preview pane

Reuse/extraction decision:

- Existing code reused as-is:
  - async envelope pattern
- Additions to existing reusable library/module:
  - async controller preview route
- New reusable library/module to create:
  - none

UI field/control inventory:

- none

## Data And Defaults

Chosen defaults / parameters:

- fixture B selection wins over fixture A completion

Data ownership:

- preview pane owns current request identity after controller events

Open questions and resolved assumptions:

- none

Implementation prerequisites:

- preview process controller

## Behavior

The implementation must:

- Decode current payload results and reject stale payload results before handing current payloads to the preview pane on the Qt UI thread.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- current result and stale result handoff tests

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
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 20.5

Split decision:

- No split needed. Cohesive current/stale payload handoff leaf.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Current And Stale Payload Handoff boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
