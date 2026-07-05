# Reference Review Spec 28: Preview Adapter Decision Spike (v1.0)

## Overview

Record the preview bridge adapter decision for the first review workbench
implementation.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the split manifest candidate `Interactive Preview Bridge Panel`.
- Manifest score: 9

## Scope

This specification covers:

- embedded pyvistaqt feasibility check
- supervised external preview rejection for normal review workflow
- adapter mode decision record

## Behavior

This leaf must define:

- choose embedded PyVistaQt as the review-workbench adapter mode
- record that supervised external preview is not acceptable for normal STL
  review because interaction must remain in the review app
- keep the decision behind the preview bridge boundary

## Constraints

- Async/concurrency behavior: preview loads run off UI thread; camera
  controls stay responsive
- Security/privacy-sensitive behavior: preview only loads selected source
  records
- Performance-sensitive behavior: model load and tessellation never block
  the UI event loop
- Cross-screen reusable behavior: preview state feeds artifact comparison
  and promotion readiness

## Dependencies And Reuse

Dependencies/services:

- preview bridge
- async dispatcher

Reusable code plan:

- Existing code reused as-is: Impression preview model-loading semantics
- Additions to existing reusable library/module: preview bridge adapter policy
  and widget-hosted preview shell
- New reusable library/module to create: none

Implementation owner/module:

- `ui/preview_bridge`
- `ui/shell`

## Data Ownership And Routes

Data ownership:

- preview bridge owns render state; UI owns controls

Routes:

- selected fixture to embedded preview panel to bridge state

## UI Contract

- Surface/component: preview panel
- Field/element: orbit, pan, zoom, reset

## Test Strategy

- preview load, source change, camera action, failure-state, and embedded shell
  route tests

## Open Questions And Prerequisites

Open questions / nuance discovered:

- embedded versus supervised bridge requires a small spike

Prerequisites before implementation:

- none

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Adapter Decision Spike boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
