# Reference Review Spec 07: Codex Fixture Context Builder (v1.0)

## Overview

Build the smallest useful fixture context payload for Codex without exposing
unrelated local files, environment state, or chat history.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-codex-sandbox.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Codex Fixture Context Builder`.
- Manifest score: 24.5

## Scope

This specification covers:

- context payload builder
- context redactor
- Codex context payload
- context omission diagnostic
- sanitized Codex context
- omission diagnostic

## Behavior

This leaf must define:

- deny extra context unless explicitly supplied by fixture metadata
- selected fixture context to Codex context payload

## Constraints

- Async/concurrency behavior: context build is fixture-scoped and stale-
  guarded
- Security/privacy-sensitive behavior: no full local environment, full chat
  log, or unrelated repo files
- Performance-sensitive behavior: bounded payload size
- Cross-screen reusable behavior: context feeds sidecar stream, candidate
  generation, and refusal diagnostics

## Dependencies And Reuse

Dependencies/services:

- source context payload
- review note store

Reusable code plan:

- Existing code reused as-is: deterministic review context payload
- Additions to existing reusable library/module: Codex context adapter
- New reusable library/module to create: Codex sidecar broker

Implementation owner/module:

- future `codex_sidecar/context`

## Data Ownership And Routes

Data ownership:

- broker owns Codex-facing context; source registry owns source context

Routes:

- selected fixture context to Codex context payload

## UI Contract

- none

## Test Strategy

- context minimization, redaction, and stale fixture context tests

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

- the Codex Fixture Context Builder boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
