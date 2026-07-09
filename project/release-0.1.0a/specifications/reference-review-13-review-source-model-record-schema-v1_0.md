# Reference Review Spec 13: Review Source Model Record Schema (v1.0)

## Overview

Define the typed record that maps a reference fixture to the loadable model
source used by the review workbench.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-fixture-source-contract.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Review Source Model Record Schema`.
- Manifest score: 21.5

## Scope

This specification covers:

- source record parser
- source record normalizer
- `ReviewSourceModelRecord`
- source identity value object
- entrypoint parameter record
- normalized source record
- schema diagnostic

## Behavior

This leaf must define:

- every reviewable fixture must define a loadable source record
- fixture id to source record

## Constraints

- Async/concurrency behavior: none; callable from discovery workers
- Security/privacy-sensitive behavior: source paths are normalized before
  display or Codex context use
- Performance-sensitive behavior: bounded parsing with no model import
- Cross-screen reusable behavior: same record feeds queue, preview, notes,
  promotion, and Codex panes

## Dependencies And Reuse

Dependencies/services:

- reference fixture metadata

Reusable code plan:

- Existing code reused as-is: preview-compatible model entrypoint convention
- Additions to existing reusable library/module: reference fixture helpers
- New reusable library/module to create: review source registry types

Implementation owner/module:

- future `src/impression/devtools/reference_review/source_registry`

## Data Ownership And Routes

Data ownership:

- fixture source contract owns source identity

Routes:

- fixture id to source record

## UI Contract

- Surface/component: none; UI display belongs to the context-panel spec

## Test Strategy

- schema acceptance and rejection tests for path, module, callable, and
  parameter forms

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

- the Review Source Model Record Schema boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
