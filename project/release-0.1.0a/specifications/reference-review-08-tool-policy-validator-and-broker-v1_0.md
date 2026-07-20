# Reference Review Spec 08: Tool Policy Validator And Broker (v1.0)

## Overview

Validate every sidecar tool request against an explicit deny-by-default
allowlist.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-codex-sandbox.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Tool Policy Validator And Broker`.
- Manifest score: 24

## Scope

This specification covers:

- tool policy validator
- broker request router
- tool policy record
- tool request record
- accepted tool request
- refused tool call diagnostic

## Behavior

This leaf must define:

- deny by default; every call requires explicit policy match
- sidecar tool request to validator to allowed service

## Constraints

- Async/concurrency behavior: tool calls are cancellable and stale-guarded
  per fixture
- Destructive/write behavior: only routes to explicitly allowed write APIs
- Security/privacy-sensitive behavior: no shell, no git, no gold writes, no
  promotion, strict root validation
- Performance-sensitive behavior: per-fixture tool-call queue is bounded
- Cross-screen reusable behavior: broker authority applies to chat,
  candidate generation, notes, and regen

## Dependencies And Reuse

Dependencies/services:

- candidate store protocol
- regeneration protocol

Reusable code plan:

- Existing code reused as-is: none
- Additions to existing reusable library/module: Codex sidecar broker
- New reusable library/module to create: none

Implementation owner/module:

- future `codex_sidecar/tool_broker`

## Data Ownership And Routes

Data ownership:

- broker owns sidecar authority

Routes:

- sidecar tool request to validator to allowed service

## UI Contract

- none

## Test Strategy

- allowed candidate write, refused shell, refused git, refused promote, and
  outside-root write refusal

## Open Questions And Prerequisites

Open questions / nuance discovered:

- exact Codex runtime API remains an integration detail

Prerequisites before implementation:

- none

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Tool Policy Validator And Broker boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
