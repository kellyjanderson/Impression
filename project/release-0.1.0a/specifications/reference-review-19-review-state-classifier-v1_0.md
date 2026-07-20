# Reference Review Spec 19: Review State Classifier (v1.0)

## Overview

Classify each fixture as unreviewed, needs-work, blocked, approved-source,
promoted, or release-gate failing.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-promotion-and-notes-lifecycle.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Review State Classifier`.
- Manifest score: 21.5

## Scope

This specification covers:

- review state classifier
- state reason builder
- review state enum
- state reason record
- review state
- state reason

## Behavior

This leaf must define:

- notes without promotion classify as needs-work
- notes/provenance/artifact presence to review state

## Constraints

- Async/concurrency behavior: classifier can run in discovery/report worker
- Security/privacy-sensitive behavior: state reasons avoid note body leakage
  in summary reports
- Performance-sensitive behavior: bounded per-fixture reads
- Cross-screen reusable behavior: review state feeds queue, action bar, and
  release reports

## Dependencies And Reuse

Dependencies/services:

- note store
- promotion provenance store

Reusable code plan:

- Existing code reused as-is: reference artifact state concepts
- Additions to existing reusable library/module: reference lifecycle helpers
- New reusable library/module to create: none

Implementation owner/module:

- future `reference_review/review_state`

## Data Ownership And Routes

Data ownership:

- classifier owns derived state, not durable records

Routes:

- notes/provenance/artifact presence to review state

## UI Contract

- none

## Test Strategy

- state matrix tests for every review status

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

- the Review State Classifier boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
