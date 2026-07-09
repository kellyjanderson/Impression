# Reference Review Spec 17: Generated Review Module Contract (v1.0)

## Overview

Define how candidate or generated review modules can be loaded by the
workbench without confusing them with committed source fixtures.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-fixture-source-contract.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Generated Review Module Contract`.
- Manifest score: 21.5

## Scope

This specification covers:

- generated module resolver
- candidate root verifier
- generated source reference
- candidate source lifecycle state
- generated source record
- refused generated source diagnostic

## Behavior

This leaf must define:

- generated modules are never promoted directly as gold evidence
- candidate file to generated source record to preview

## Constraints

- Async/concurrency behavior: candidate resolution runs through source
  discovery worker
- Destructive/write behavior: none in resolver; writes belong to Codex
  candidate store
- Security/privacy-sensitive behavior: candidate modules must live under
  allowed generated roots
- Performance-sensitive behavior: no broad import scanning
- Cross-screen reusable behavior: candidate source records feed preview,
  Codex, and adoption UI

## Dependencies And Reuse

Dependencies/services:

- source registry
- Codex candidate store

Reusable code plan:

- Existing code reused as-is: source record schema
- Additions to existing reusable library/module: generated-source adapter
- New reusable library/module to create: none

Implementation owner/module:

- future `source_registry.generated`

## Data Ownership And Routes

Data ownership:

- source registry identifies candidates; Codex store owns candidate files

Routes:

- candidate file to generated source record to preview

## UI Contract

- Surface/component: none; candidate list presentation belongs to Codex/UI
  specs

## Test Strategy

- generated-root acceptance, outside-root refusal, stale candidate rejection

## Open Questions And Prerequisites

Open questions / nuance discovered:

- none

Prerequisites before implementation:

- none; integrate against a candidate-store protocol stub in tests

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Generated Review Module Contract boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
