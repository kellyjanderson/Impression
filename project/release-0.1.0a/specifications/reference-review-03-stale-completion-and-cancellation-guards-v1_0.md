# Reference Review Spec 03: Stale Completion And Cancellation Guards (v1.0)

## Overview

Prevent old background results from mutating newer UI or review state.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-async-concurrency.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Stale Completion And Cancellation Guards`.
- Manifest score: 23.5

## Scope

This specification covers:

- latest request tracker
- stale completion guard
- cancellation marker
- request tracker
- cancellation token
- accepted completion
- stale completion diagnostic

## Behavior

This leaf must define:

- newer owner request supersedes older non-durable tasks
- completion envelope to owner guard to UI state

## Constraints

- Async/concurrency behavior: per-owner latest request guards and
  cancellation markers
- Destructive/write behavior: stale destructive completions are rejected
- Performance-sensitive behavior: constant-time request lookup
- Cross-screen reusable behavior: shared by preview, notes, promotion,
  artifact generation, and Codex

## Dependencies And Reuse

Dependencies/services:

- message envelope
- dispatcher

Reusable code plan:

- Existing code reused as-is: ViewDown stale-result precedent
- Additions to existing reusable library/module: async core
- New reusable library/module to create: none

Implementation owner/module:

- future `async_core.staleness`

## Data Ownership And Routes

Data ownership:

- UI owners decide whether completions apply

Routes:

- completion envelope to owner guard to UI state

## UI Contract

- none

## Test Strategy

- old preview completion after newer selection; stale note completion
  refusal

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

- the Stale Completion And Cancellation Guards boundary is implemented as
  described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
