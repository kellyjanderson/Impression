# Reference Review Spec 12: Sidecar Process Boundary And Audit (v1.0)

## Overview

Keep Codex execution/tooling out of trusted UI process authority and emit
structured audit events for every sidecar action and refusal.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-codex-sandbox.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Sidecar Process Boundary And Audit`.
- Manifest score: 24.5

## Scope

This specification covers:

- sidecar process launcher
- sidecar audit emitter
- sidecar session record
- sidecar audit event
- sidecar session started
- sidecar audit event

## Behavior

This leaf must define:

- sidecar may request tools but cannot directly mutate project state
- UI sidecar request to sidecar process to tool broker to audit stream

## Constraints

- Async/concurrency behavior: sidecar session is cancellable and fixture-
  scoped
- Destructive/write behavior: writes audit events only
- Security/privacy-sensitive behavior: in-process Python restrictions are
  not treated as a security boundary
- Performance-sensitive behavior: audit event size is bounded
- Cross-screen reusable behavior: audit events support Codex UI, release
  reports, and troubleshooting

## Dependencies And Reuse

Dependencies/services:

- async dispatcher
- structured logging

Reusable code plan:

- Existing code reused as-is: structured task audit event shape
- Additions to existing reusable library/module: Codex sidecar broker
- New reusable library/module to create: none

Implementation owner/module:

- future `codex_sidecar/process_boundary`

## Data Ownership And Routes

Data ownership:

- broker owns sidecar authority and audit records

Routes:

- UI sidecar request to sidecar process to tool broker to audit stream

## UI Contract

- none

## Test Strategy

- sidecar cancellation, audit event, refused direct write, and process
  failure tests

## Open Questions And Prerequisites

Open questions / nuance discovered:

- exact Codex runtime embedding remains an implementation choice

Prerequisites before implementation:

- none

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Sidecar Process Boundary And Audit boundary is implemented as
  described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
