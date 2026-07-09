# Reference Review Spec 26: Queue Navigation Panel (v1.0)

## Overview

Implement the fixture queue navigation panel for selecting review fixtures.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the split manifest candidate `Queue And Context Panels`.
- Manifest score: 17.5

## Scope

This specification covers:

- fixture queue view model
- previous and next navigation actions
- status badge display
- empty queue state

## Behavior

This leaf must define:

- select the first dirty fixture by default
- route navigation changes through selected fixture state
- virtualize or reuse delegates for large queues

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

- the Queue Navigation Panel boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
