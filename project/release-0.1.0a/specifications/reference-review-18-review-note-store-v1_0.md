# Reference Review Spec 18: Review Note Store (v1.0)

## Overview

Persist fixture-scoped review notes without promoting dirty artifacts.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-promotion-and-notes-lifecycle.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Review Note Store`.
- Manifest score: 24.5

## Scope

This specification covers:

- note loader
- note writer
- review note record
- note write result
- loaded note
- saved note

## Behavior

This leaf must define:

- notes without promotion fail review completion
- fixture id to note path to note record

## Constraints

- Async/concurrency behavior: note writes are serialized through durable
  write lane
- Destructive/write behavior: writes note files
- Security/privacy-sensitive behavior: note files do not persist full chat
  logs or local secrets by default
- Performance-sensitive behavior: bounded note size
- Cross-screen reusable behavior: notes feed queue status, notes panel, and
  release reports

## Dependencies And Reuse

Dependencies/services:

- source record resolver
- async durable write queue

Reusable code plan:

- Existing code reused as-is: current reference artifact state concepts
- Additions to existing reusable library/module: reference lifecycle helpers
- New reusable library/module to create: review note store

Implementation owner/module:

- future `reference_review/notes`

## Data Ownership And Routes

Data ownership:

- note store owns durable review notes

Routes:

- fixture id to note path to note record

## UI Contract

- Surface/component: none; notes UI belongs to the UI manifest

## Test Strategy

- note read/write, redaction/default content, stale write completion tests

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

- the Review Note Store boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
