# Reference Review Spec 14: Source Record Validation And Diagnostics (v1.0)

## Overview

Validate source records without executing the model and report every blocking
fixture problem in one pass.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-fixture-source-contract.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Source Record Validation And Diagnostics`.
- Manifest score: 23.5

## Scope

This specification covers:

- source record validator
- aggregate diagnostic reporter
- validation diagnostic
- validation result envelope
- valid result
- source validation diagnostics

## Behavior

This leaf must define:

- invalid source records block review execution but do not stop full scan
- source record to validation result to queue/context UI

## Constraints

- Async/concurrency behavior: runs in discovery worker and returns typed
  diagnostics
- Security/privacy-sensitive behavior: diagnostics redact unrelated absolute
  paths
- Performance-sensitive behavior: validation never imports or tessellates
  the model
- Cross-screen reusable behavior: diagnostic objects are reusable by queue,
  context, and Codex context summary

## Dependencies And Reuse

Dependencies/services:

- source record schema
- filesystem path policy

Reusable code plan:

- Existing code reused as-is: path helper conventions
- Additions to existing reusable library/module: reference fixture helpers
- New reusable library/module to create: none

Implementation owner/module:

- future `source_registry.validation`

## Data Ownership And Routes

Data ownership:

- validator owns source-record diagnostics

Routes:

- source record to validation result to queue/context UI

## UI Contract

- Surface/component: none; presentation belongs to UI panel specs

## Test Strategy

- missing path, missing callable, bad parameters, and multi-error
  aggregation

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

- the Source Record Validation And Diagnostics boundary is implemented as
  described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
