# Reference Review Spec 73: Preview Pane Display Control Slot (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one preview-pane chrome change: remove title and provide a stable control-bar slot.

## Overview

Replace the preview pane's large title area with a stable display-control slot.

## Backlink

- [Architecture: Reference Review Preview Display Controls Product Definition](../architecture/reference-review-preview-display-controls-product-definition.md)

## Scope

This specification covers removing the visible `Selected Fixture` title and
creating a one-row area above the preview surface for display controls.

It does not cover which controls appear in the row.

## Implementation Boundary

Owner/module:

- reference-review shell/preview pane UI

Reuse/extraction decision:

- Add the slot to the preview pane rather than nesting it into the render widget.

## Behavior

The implementation must:

- remove the large `Selected Fixture` title above the preview surface
- add a stable single-row slot above the preview surface
- keep the slot visible during empty, loading, ready, and failed preview states
- prevent slot state changes from resizing the preview surface

## Verification

Test strategy:

- shell test asserts title absence
- shell test asserts display-control slot exists
- layout test asserts preview surface remains present and stable

## Manifest Assessment

Score:

- Functions/methods: 1 x 2 = 2
- Data structures/models: 0 x 1 = 0
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 13.5

Split decision:

- No split needed. Cohesive preview-pane slot leaf.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when the preview pane has a stable display-control slot and no large title.
