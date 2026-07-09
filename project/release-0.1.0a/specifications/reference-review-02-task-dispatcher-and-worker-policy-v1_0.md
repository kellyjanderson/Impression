# Reference Review Spec 02: Task Dispatcher And Worker Policy (v1.0)

## Overview

Route workbench tasks to bounded workers without blocking the Qt event loop.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-async-concurrency.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Task Dispatcher And Worker Policy`.
- Manifest score: 23

## Scope

This specification covers:

- task dispatcher
- worker pool selector
- queue throttle
- dispatch request
- worker policy record
- dispatch accepted
- dispatch rejected diagnostic

## Behavior

This leaf must define:

- UI thread submits work; worker completions return through typed envelopes
- QML bridge to dispatcher to worker

## Constraints

- Async/concurrency behavior: bounded queues, coalesced preview rebuilds,
  throttled scans
- Security/privacy-sensitive behavior: task payloads do not include
  unrestricted environment dumps
- Performance-sensitive behavior: per-task queue limits and cancellation of
  superseded preview work
- Cross-screen reusable behavior: dispatcher serves all workbench panels

## Dependencies And Reuse

Dependencies/services:

- Qt signal bridge
- worker pool

Reusable code plan:

- Existing code reused as-is: none
- Additions to existing reusable library/module: async core
- New reusable library/module to create: none

Implementation owner/module:

- future `async_core.dispatcher`

## Data Ownership And Routes

Data ownership:

- dispatcher owns task scheduling only

Routes:

- QML bridge to dispatcher to worker

## UI Contract

- none

## Test Strategy

- bounded queue, coalescing, and failed dispatch tests

## Open Questions And Prerequisites

Open questions / nuance discovered:

- exact worker technology remains behind policy interface

Prerequisites before implementation:

- none

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Task Dispatcher And Worker Policy boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
