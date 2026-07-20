# Reference Review Spec 09: Candidate Model Store (v1.0)

## Overview

Write generated candidate model files only under approved candidate roots and
expose them as non-promoted review sources.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-codex-sandbox.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Candidate Model Store`.
- Manifest score: 24.5

## Scope

This specification covers:

- candidate model writer
- candidate root verifier
- candidate model record
- candidate write result
- candidate file path
- refused candidate write diagnostic

## Behavior

This leaf must define:

- generated candidates are never promoted directly
- tool request to candidate writer to generated source resolver

## Constraints

- Async/concurrency behavior: candidate writes use serialized write lane
- Destructive/write behavior: writes candidate files only
- Security/privacy-sensitive behavior: strict root validation; candidates
  cannot overwrite gold references
- Performance-sensitive behavior: bounded candidate file size
- Cross-screen reusable behavior: candidates feed preview, candidate list,
  and adoption workflow

## Dependencies And Reuse

Dependencies/services:

- generated source resolver
- async durable write lane

Reusable code plan:

- Existing code reused as-is: generated source contract
- Additions to existing reusable library/module: Codex sidecar broker
- New reusable library/module to create: none

Implementation owner/module:

- future `codex_sidecar/candidates`

## Data Ownership And Routes

Data ownership:

- candidate store owns candidate files; human reviewer owns adoption

Routes:

- tool request to candidate writer to generated source resolver

## UI Contract

- none

## Test Strategy

- allowed candidate write, outside-root refusal, overwrite policy, and
  source resolver handoff tests

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

- the Candidate Model Store boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
