# Reference Review Spec 33: Artifact Panel (v1.0)

## Overview

Implement the derived artifact review panel.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the split manifest candidate `Artifact And Notes Review Panels`.
- Manifest score: 15.5

## Scope

This specification covers:

- artifact tile view model
- derived artifact list
- thumbnail loading
- artifact diff/status badges

## Behavior

This leaf must define:

- load thumbnails asynchronously
- present artifacts as derived evidence
- request artifact actions without owning promotion writes

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

- the Artifact Panel boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
