# Reference Review Spec 04: Durable Write Serialization And File Locking (v1.0)

## Overview

Serialize note and promotion writes across threads and processes.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-async-concurrency.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Durable Write Serialization And File Locking`.
- Manifest score: 24.5

## Scope

This specification covers:

- durable write scheduler
- file lock wrapper
- durable write request
- lock acquisition result
- write accepted
- lock conflict diagnostic

## Behavior

This leaf must define:

- bounded lock wait; timeout returns a diagnostic instead of blocking UI
- service write request to durable lane to completion envelope

## Constraints

- Async/concurrency behavior: serialized write lane with cross-process locks
- Destructive/write behavior: controls note and gold artifact writes
- Security/privacy-sensitive behavior: lock paths remain inside configured
  review roots
- Performance-sensitive behavior: bounded lock wait and timeout
- Cross-screen reusable behavior: shared by notes, promotion, candidate
  adoption, and release reports

## Dependencies And Reuse

Dependencies/services:

- filelock or equivalent
- promotion/note services

Reusable code plan:

- Existing code reused as-is: reference artifact path helpers
- Additions to existing reusable library/module: async durable write helpers
- New reusable library/module to create: none

Implementation owner/module:

- future `async_core.durable_writes`

## Data Ownership And Routes

Data ownership:

- durable write lane owns write ordering, not write semantics

Routes:

- service write request to durable lane to completion envelope

## UI Contract

- none

## Test Strategy

- concurrent note write, concurrent promotion, lock timeout, stale write
  tests

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

- the Durable Write Serialization And File Locking boundary is implemented
  as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
