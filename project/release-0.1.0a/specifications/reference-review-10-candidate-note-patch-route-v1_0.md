# Reference Review Spec 10: Candidate Note Patch Route (v1.0)

## Overview

Let Codex propose note patches without writing durable notes directly.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-codex-sandbox.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Candidate Note Patch Route`.
- Manifest score: 24.5

## Scope

This specification covers:

- candidate note patch writer
- note patch validator
- candidate note patch
- patch validation result
- candidate note patch
- refused patch diagnostic

## Behavior

This leaf must define:

- human adoption required for note changes
- sidecar patch request to candidate patch to human adoption

## Constraints

- Async/concurrency behavior: patch proposals are stale-guarded per fixture
- Destructive/write behavior: no durable note write until human adoption
- Security/privacy-sensitive behavior: patch cannot include full chat logs
  by default
- Performance-sensitive behavior: bounded patch size
- Cross-screen reusable behavior: patch feeds Codex panel and notes panel
  adoption route

## Dependencies And Reuse

Dependencies/services:

- note store protocol
- tool broker

Reusable code plan:

- Existing code reused as-is: note record shape
- Additions to existing reusable library/module: Codex sidecar broker
- New reusable library/module to create: none

Implementation owner/module:

- future `codex_sidecar/note_patches`

## Data Ownership And Routes

Data ownership:

- broker owns proposed patch; note store owns durable notes

Routes:

- sidecar patch request to candidate patch to human adoption

## UI Contract

- none

## Test Strategy

- patch creation, oversized patch refusal, stale patch refusal, adoption
  handoff

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

- the Candidate Note Patch Route boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
