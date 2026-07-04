# Reference Review Spec 27: Selected Fixture Context Panel (v1.0)

## Overview

Implement the selected fixture context panel that displays safe source and
expected-output details.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the split manifest candidate `Queue And Context Panels`.
- Manifest score: 15

## Scope

This specification covers:

- selected fixture context view model
- source path display
- expected output display
- redacted source/context fields

## Behavior

This leaf must define:

- display source-registry fields without owning fixture data
- update dependent panels when selected fixture changes
- preserve redaction boundaries from the source registry

## Constraints

- Async/concurrency behavior: selection changes submit source-load tasks
  through dispatcher
- Security/privacy-sensitive behavior: redacted paths use source-registry
  display fields
- Performance-sensitive behavior: large queues virtualize or reuse delegates
- Cross-screen reusable behavior: selection drives preview, notes,
  artifacts, and Codex

## Dependencies And Reuse

Dependencies/services:

- source registry
- async dispatcher

Reusable code plan:

- Existing code reused as-is: shared workbench components
- Additions to existing reusable library/module: none
- New reusable library/module to create: none

Implementation owner/module:

- future `ui/panels/queue_context`

## Data Ownership And Routes

Data ownership:

- UI owns selected fixture; source registry owns fixture data

Routes:

- queue selection to selected fixture model to dependent panels

## UI Contract

- Surface/component: queue panel
- Surface/component: context panel
- Field/element: previous, next, fixture id, status badge, source path,
  expected output

## Test Strategy

- navigation, selection, redacted display, and empty-state tests

## Open Questions And Prerequisites

Open questions / nuance discovered:

- none

Prerequisites before implementation:

- source registry protocol must exist

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Selected Fixture Context Panel boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
