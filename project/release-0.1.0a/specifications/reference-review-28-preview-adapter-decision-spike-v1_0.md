# Reference Review Spec 28: Preview Adapter Decision Spike (v1.0)

## Overview

Decide the first preview bridge adapter mode before implementing preview
binding.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the split manifest candidate `Interactive Preview Bridge Panel`.
- Manifest score: 9

## Scope

This specification covers:

- embedded pyvistaqt feasibility check
- supervised external preview fallback check
- adapter mode decision record

## Behavior

This leaf must define:

- choose one adapter mode before preview load binding implementation
- record failure reasons for rejected adapter modes
- keep the decision behind the preview bridge boundary

## Constraints

- Async/concurrency behavior: preview loads run off UI thread; camera
  controls stay responsive
- Security/privacy-sensitive behavior: preview only loads selected source
  records
- Performance-sensitive behavior: model load and tessellation never block
  QML event loop
- Cross-screen reusable behavior: preview state feeds artifact comparison
  and promotion readiness

## Dependencies And Reuse

Dependencies/services:

- preview bridge
- async dispatcher

Reusable code plan:

- Existing code reused as-is: Impression preview model-loading semantics
- Additions to existing reusable library/module: preview bridge adapter
- New reusable library/module to create: none

Implementation owner/module:

- future `ui/preview_bridge`

## Data Ownership And Routes

Data ownership:

- preview bridge owns render state; UI owns controls

Routes:

- selected fixture to preview task to bridge state

## UI Contract

- Surface/component: preview panel
- Field/element: orbit, pan, zoom, reset

## Test Strategy

- preview load, source change, camera action, and failure-state tests

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
