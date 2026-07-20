# Reference Review Spec 38: Component Gallery (v1.0)

## Overview

Implement the reusable component gallery for workbench UI review.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the split manifest candidate `Component Gallery And Screenshot State Suite`.
- Manifest score: 14.5

## Scope

This specification covers:

- reusable component gallery
- component state contract records
- synthetic fixture data

## Behavior

This leaf must define:

- enumerate reusable component states
- render components in realistic shell context
- avoid secret or local path data in gallery fixtures

## Constraints

- Async/concurrency behavior: screenshot generation runs outside interactive
  UI session
- Destructive/write behavior: writes test screenshots only
- Security/privacy-sensitive behavior: fixtures use synthetic non-secret
  data
- Performance-sensitive behavior: screenshot suite is bounded for CI/dev
  runs
- Cross-screen reusable behavior: gallery validates all panel components

## Dependencies And Reuse

Dependencies/services:

- QML test harness
- component library

Reusable code plan:

- Existing code reused as-is: shared components
- Additions to existing reusable library/module: screenshot test harness
- New reusable library/module to create: none

Implementation owner/module:

- future `ui/tests/component_gallery`

## Data Ownership And Routes

Data ownership:

- UI test suite owns visual state evidence

Routes:

- component scenarios to rendered screenshots to review artifacts

## UI Contract

- Surface/component: reusable component gallery
- Field/element: hover, focus, disabled, loading, error, empty, overflow
  states

## Test Strategy

- automated screenshot/state matrix plus manual UI review checklist

## Open Questions And Prerequisites

Open questions / nuance discovered:

- none

Prerequisites before implementation:

- component library must exist

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Component Gallery boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
