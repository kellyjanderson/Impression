# Reference Review Spec 20: Promotion Validator (v1.0)

## Overview

Validate that a fixture can be promoted only after source model and derived
artifact evidence are reviewable.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-promotion-and-notes-lifecycle.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Promotion Validator`.
- Manifest score: 21.5

## Scope

This specification covers:

- promotion validator
- blocked promotion diagnostic builder
- promotion validation result
- blocked promotion diagnostic
- promotion allowed
- blocked promotion diagnostic

## Behavior

This leaf must define:

- no source record means promotion refused
- fixture evidence to validation result

## Constraints

- Async/concurrency behavior: validator can run in worker before durable
  write
- Security/privacy-sensitive behavior: diagnostics omit unrelated local
  paths
- Performance-sensitive behavior: checksum validation is bounded to fixture
  artifacts
- Cross-screen reusable behavior: validation result feeds confirmation,
  queue, and release report

## Dependencies And Reuse

Dependencies/services:

- source record resolver
- dirty/clean reference path helpers

Reusable code plan:

- Existing code reused as-is: reference path helpers
- Additions to existing reusable library/module: reference lifecycle helpers
- New reusable library/module to create: none

Implementation owner/module:

- future `reference_review/promotion_validation`

## Data Ownership And Routes

Data ownership:

- validator owns promotion eligibility

Routes:

- fixture evidence to validation result

## UI Contract

- none

## Test Strategy

- missing source, missing dirty artifact, checksum mismatch, and allowed
  cases

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

- the Promotion Validator boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
