# Reference Review Spec 21: Atomic Promotion Executor (v1.0)

## Overview

Promote dirty artifacts to gold references atomically with cross-process
locking and rollback diagnostics.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-promotion-and-notes-lifecycle.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Atomic Promotion Executor`.
- Manifest score: 24.5

## Scope

This specification covers:

- promotion executor
- rollback diagnostic builder
- promotion request
- promotion result
- promotion success
- promotion failure diagnostic

## Behavior

This leaf must define:

- bounded lock wait; failure leaves dirty artifacts untouched
- validated promotion request to durable write lane to result

## Constraints

- Async/concurrency behavior: promotion writes run through serialized
  durable write lane
- Destructive/write behavior: writes gold artifacts and replaces previous
  promoted evidence
- Security/privacy-sensitive behavior: writes only inside configured
  reference roots
- Performance-sensitive behavior: copies bounded fixture artifacts and
  verifies checksums
- Cross-screen reusable behavior: promotion result feeds queue, notes panel,
  and release reports

## Dependencies And Reuse

Dependencies/services:

- async durable write queue
- file lock wrapper

Reusable code plan:

- Existing code reused as-is: reference path helpers
- Additions to existing reusable library/module: reference lifecycle helpers
- New reusable library/module to create: none

Implementation owner/module:

- future `reference_review/promotion_executor`

## Data Ownership And Routes

Data ownership:

- promotion executor owns gold artifact mutation

Routes:

- validated promotion request to durable write lane to result

## UI Contract

- none

## Test Strategy

- atomic write, lock conflict, checksum failure, rollback, and stale
  completion tests

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

- the Atomic Promotion Executor boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
