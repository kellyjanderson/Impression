# Reference Review Spec 01: Review Workbench Message Envelope (v1.0)

## Overview

Define the typed message/result envelope used by all background workbench
tasks.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-async-concurrency.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Review Workbench Message Envelope`.
- Manifest score: 23.5

## Scope

This specification covers:

- request id allocator
- envelope factory
- `ReviewWorkbenchMessage`
- task kind enum
- worker result envelope
- task request envelope
- task result envelope

## Behavior

This leaf must define:

- request ids are per-owner monotonic values
- caller to dispatcher to worker to UI owner

## Constraints

- Async/concurrency behavior: all worker routes use the same envelope shape
- Security/privacy-sensitive behavior: envelope error text supports redacted
  display fields
- Performance-sensitive behavior: lightweight immutable records
- Cross-screen reusable behavior: shared by queue, preview, artifacts,
  notes, promotion, and Codex panes

## Dependencies And Reuse

Dependencies/services:

- PySide signal bridge

Reusable code plan:

- Existing code reused as-is: ViewDown pattern as design precedent only
- Additions to existing reusable library/module: none
- New reusable library/module to create: review workbench async core

Implementation owner/module:

- future `src/impression/devtools/reference_review/async_core/messages`

## Data Ownership And Routes

Data ownership:

- async core owns task envelopes

Routes:

- caller to dispatcher to worker to UI owner

## UI Contract

- none

## Test Strategy

- envelope construction, serialization, and owner/request matching tests

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

- the Review Workbench Message Envelope boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
