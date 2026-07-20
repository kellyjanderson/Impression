# Reference Review Spec 34: Notes Panel (v1.0)

## Overview

Implement the review notes panel and note-save action surface.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the split manifest candidate `Artifact And Notes Review Panels`.
- Manifest score: 16

## Scope

This specification covers:

- note editor state
- note save action
- failed review flag/status
- save failure display

## Behavior

This leaf must define:

- route note saves through the promotion/notes service
- keep notes without promotion as failed review state
- avoid storing full chat logs by default

## Constraints

- Async/concurrency behavior: note saves route through dispatcher and
  durable write lane
- Destructive/write behavior: none directly from UI; service owns writes
- Security/privacy-sensitive behavior: notes editor does not include full
  chat logs by default
- Performance-sensitive behavior: thumbnails load asynchronously
- Cross-screen reusable behavior: review state updates queue and action bar

## Dependencies And Reuse

Dependencies/services:

- promotion/notes service
- async dispatcher

Reusable code plan:

- Existing code reused as-is: shared components
- Additions to existing reusable library/module: none
- New reusable library/module to create: none

Implementation owner/module:

- future `ui/panels/artifacts_notes`

## Data Ownership And Routes

Data ownership:

- UI owns edit state; service owns durable notes

Routes:

- selected fixture to artifact/note view models to service action

## UI Contract

- Surface/component: artifact panel
- Surface/component: notes panel
- Field/element: artifact tiles, diff badge, status, notes editor, flag

## Test Strategy

- artifact list, thumbnail loading, note edit, save failure, and flag tests

## Open Questions And Prerequisites

Open questions / nuance discovered:

- none

Prerequisites before implementation:

- promotion/notes service protocol must exist

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Notes Panel boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
