# Reference Review Spec 06: Structured Task Audit Events (v1.0)

## Overview

Emit fixture-scoped structured events for task submission, refusal, failure,
completion, cancellation, and stale-result rejection.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-async-concurrency.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Structured Task Audit Events`.
- Manifest score: 24

## Scope

This specification covers:

- audit event builder
- task audit emitter
- audit event record
- fixture-scoped log context
- structured audit event
- audit emission diagnostic

## Behavior

This leaf must define:

- JSON-compatible structured events with `fixture_id`, `task_kind`, and
  `request_id`
- task lifecycle to audit emitter to local log sink

## Constraints

- Async/concurrency behavior: audit emission never blocks UI completion
- Destructive/write behavior: writes non-authoritative local log events
- Security/privacy-sensitive behavior: audit events omit secrets and full
  local environment
- Performance-sensitive behavior: bounded event size
- Cross-screen reusable behavior: audit events support task history, Codex
  refusals, and promotion reports

## Dependencies And Reuse

Dependencies/services:

- structlog or equivalent
- message envelope

Reusable code plan:

- Existing code reused as-is: none
- Additions to existing reusable library/module: async core audit hook
- New reusable library/module to create: none

Implementation owner/module:

- future `async_core.audit`

## Data Ownership And Routes

Data ownership:

- async audit owns task event records

Routes:

- task lifecycle to audit emitter to local log sink

## UI Contract

- none

## Test Strategy

- event shape, redaction, and non-blocking emission tests

## Open Questions And Prerequisites

Open questions / nuance discovered:

- log sink path belongs to dependency/local-state policy

Prerequisites before implementation:

- none

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Structured Task Audit Events boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
