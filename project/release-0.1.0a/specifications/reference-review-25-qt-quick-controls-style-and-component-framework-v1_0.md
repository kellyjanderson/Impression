# Reference Review Spec 25: Qt Quick Controls Style And Component Framework (v1.0)

## Overview

Establish Qt Quick Controls 2 styling, tokens, and Impression-owned wrapper
component rules.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Qt Quick Controls Style And Component Framework`.
- Manifest score: 21.5

## Scope

This specification covers:

- style configuration loader
- component registration
- style token record
- component contract record
- style loaded diagnostic

## Behavior

This leaf must define:

- Qt Quick Controls 2 base with Impression wrappers; no Kirigami or
  FluentPySide requirement
- QML import to component use

## Constraints

- Performance-sensitive behavior: stable component sizing and no layout
  thrash
- Cross-screen reusable behavior: all panels consume shared controls

## Dependencies And Reuse

Dependencies/services:

- Qt Quick Controls 2

Reusable code plan:

- Existing code reused as-is: Qt Quick Controls 2
- Additions to existing reusable library/module: none
- New reusable library/module to create: workbench component library

Implementation owner/module:

- future `ui/components`

## Data Ownership And Routes

Data ownership:

- component library owns shared visual primitives

Routes:

- QML import to component use

## UI Contract

- Surface/component: shared component foundation
- Field/element: icon button, text field, status badge, split pane

## Test Strategy

- component instantiation and visual state smoke tests

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

- the Qt Quick Controls Style And Component Framework boundary is
  implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
