# Reference Review Spec 40: Accessibility/Overflow State Matrix (v1.0)

## Overview

Define and verify accessibility and overflow states for workbench UI
components.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the split manifest candidate `Component Gallery And Screenshot State Suite`.
- Manifest score: 8

## Scope

This specification covers:

- focus state matrix
- disabled/loading/error/empty states
- overflow and long-text scenarios

## Behavior

This leaf must define:

- cover keyboard focus and overflow states
- exercise compact and large text cases
- feed manual UI review checklist with concrete screenshots

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

- the Accessibility/Overflow State Matrix boundary is implemented as
  described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
