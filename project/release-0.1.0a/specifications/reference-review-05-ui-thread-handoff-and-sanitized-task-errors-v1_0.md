# Reference Review Spec 05: UI Thread Handoff And Sanitized Task Errors (v1.0)

## Overview

Ensure worker results reach QML-visible state only through UI-thread handoff
with sanitized diagnostics.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-async-concurrency.md)

## Source Manifest

- This leaf is derived from the manifest candidate `UI Thread Handoff And Sanitized Task Errors`.
- Manifest score: 21.5

## Scope

This specification covers:

- UI-thread completion bridge
- task error sanitizer
- UI completion event
- sanitized diagnostic
- UI completion signal
- sanitized failure diagnostic

## Behavior

This leaf must define:

- workers never mutate QML view models directly
- worker completion to Qt signal to owner view model

## Constraints

- Async/concurrency behavior: no QML-visible state mutation outside UI
  thread
- Security/privacy-sensitive behavior: errors sanitize paths before UI and
  Codex exposure
- Performance-sensitive behavior: minimal bridge overhead
- Cross-screen reusable behavior: shared by all workbench panels

## Dependencies And Reuse

Dependencies/services:

- Qt signal bridge
- message envelope

Reusable code plan:

- Existing code reused as-is: Qt signal/slot pattern
- Additions to existing reusable library/module: async core UI adapter
- New reusable library/module to create: none

Implementation owner/module:

- future `async_core.qt_handoff`

## Data Ownership And Routes

Data ownership:

- UI bridge owns thread handoff only

Routes:

- worker completion to Qt signal to owner view model

## UI Contract

- Surface/component: none; panels render diagnostics separately

## Test Strategy

- UI-thread handoff and sanitized exception tests

## Open Questions And Prerequisites

Open questions / nuance discovered:

- none

Prerequisites before implementation:

- none

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the UI Thread Handoff And Sanitized Task Errors boundary is implemented as
  described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
