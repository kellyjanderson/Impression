# Reference Review Spec 16: Deterministic Review Context Payload (v1.0)

## Overview

Define the fixture context payload used by preview, notes, promotion, and
Codex without exposing unrelated local environment state.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-fixture-source-contract.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Deterministic Review Context Payload`.
- Manifest score: 22.5

## Scope

This specification covers:

- context payload builder
- context sanitizer
- review context payload
- expected output summary
- determinism input record
- sanitized context payload
- context omission diagnostic

## Behavior

This leaf must define:

- context is fixture-scoped and source-derived
- source record plus fixture metadata to context payload

## Constraints

- Async/concurrency behavior: context build can run in discovery worker
- Security/privacy-sensitive behavior: omits secrets, unrelated paths, and
  full environment dumps
- Performance-sensitive behavior: no model execution during context build
- Cross-screen reusable behavior: payload feeds context UI, Codex, promotion
  provenance, and notes

## Dependencies And Reuse

Dependencies/services:

- source record schema
- fixture metadata

Reusable code plan:

- Existing code reused as-is: fixture metadata conventions
- Additions to existing reusable library/module: source registry context API
- New reusable library/module to create: none

Implementation owner/module:

- future `source_registry.context`

## Data Ownership And Routes

Data ownership:

- source registry owns context payload construction

Routes:

- source record plus fixture metadata to context payload

## UI Contract

- Surface/component: none; Markdown/context rendering belongs to UI specs

## Test Strategy

- context payload snapshot tests and secret/path redaction tests

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

- the Deterministic Review Context Payload boundary is implemented as
  described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
