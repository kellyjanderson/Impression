# Reference Review Spec 22: Promotion Provenance And Release Gate Report (v1.0)

## Overview

Record source provenance for promotions and report fixtures that remain
unreviewed, noted-only, blocked, or unpromoted.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-promotion-and-notes-lifecycle.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Promotion Provenance And Release Gate Report`.
- Manifest score: 24.5

## Scope

This specification covers:

- provenance writer
- release gate reporter
- promotion provenance record
- release gate report
- provenance record
- release gate failure report

## Behavior

This leaf must define:

- promotion provenance includes source identity and artifact checksums
- promotion result plus source context to provenance and release report

## Constraints

- Async/concurrency behavior: provenance writes use durable write lane;
  reports can run in worker
- Destructive/write behavior: writes provenance files
- Security/privacy-sensitive behavior: provenance excludes local secrets and
  full chat logs
- Performance-sensitive behavior: release report scans bounded reference
  roots
- Cross-screen reusable behavior: provenance/report data feeds queue,
  release summaries, and audit trails

## Dependencies And Reuse

Dependencies/services:

- source context payload
- review state classifier

Reusable code plan:

- Existing code reused as-is: source context payload
- Additions to existing reusable library/module: reference lifecycle helpers
- New reusable library/module to create: none

Implementation owner/module:

- future `reference_review/provenance`

## Data Ownership And Routes

Data ownership:

- provenance owns promotion evidence metadata

Routes:

- promotion result plus source context to provenance and release report

## UI Contract

- none

## Test Strategy

- provenance shape, redaction, release failure matrix, and report ordering

## Open Questions And Prerequisites

Open questions / nuance discovered:

- UI may say `gold` while legacy code paths may say `clean`

Prerequisites before implementation:

- none

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Promotion Provenance And Release Gate Report boundary is implemented
  as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
