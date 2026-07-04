# Reference Review Spec 11: Regeneration Request Route (v1.0)

## Overview

Allow the sidecar to request artifact regeneration for a selected source or
candidate without executing promotion.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-codex-sandbox.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Regeneration Request Route`.
- Manifest score: 24.5

## Scope

This specification covers:

- regeneration request router
- regeneration eligibility validator
- regeneration request
- regeneration request result
- regeneration request id
- refused regeneration diagnostic

## Behavior

This leaf must define:

- regeneration is allowed only for current selected fixture/candidate
- sidecar request to dispatcher to regeneration service

## Constraints

- Async/concurrency behavior: routed through dispatcher and stale-guarded by
  fixture
- Destructive/write behavior: writes dirty artifacts only through
  regeneration service
- Security/privacy-sensitive behavior: cannot write gold artifacts or call
  promotion APIs
- Performance-sensitive behavior: per-fixture regeneration queue is bounded
- Cross-screen reusable behavior: route feeds artifacts panel and candidate
  preview

## Dependencies And Reuse

Dependencies/services:

- artifact regeneration service
- async dispatcher

Reusable code plan:

- Existing code reused as-is: artifact regeneration command shape
- Additions to existing reusable library/module: Codex sidecar broker
- New reusable library/module to create: none

Implementation owner/module:

- future `codex_sidecar/regeneration`

## Data Ownership And Routes

Data ownership:

- regeneration service owns dirty artifacts

Routes:

- sidecar request to dispatcher to regeneration service

## UI Contract

- none

## Test Strategy

- allowed regeneration, stale fixture refusal, candidate regeneration,
  promote refusal

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

- the Regeneration Request Route boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
