# Reference Review Spec 74: Preview Display Control Row Composition (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one composed control row that places prerequisite reusable controls in product-defined order.

## Overview

Compose the preview display-control row inside the preview pane control slot.

## Backlink

- [Reference Review Spec 68: Exclusive Icon Group Component Composition](reference-review-68-exclusive-icon-group-component-composition-v1_0.md)
- [Reference Review Spec 70: Preview Display Command Routing](reference-review-70-preview-display-command-routing-v1_0.md)
- [Reference Review Spec 73: Preview Pane Display Control Slot](reference-review-73-preview-pane-display-control-slot-v1_0.md)

## Scope

This specification covers group ordering, independent toggle ordering,
separator placement, and disabled/readiness binding for the composed row.

It does not cover reusable button internals, renderer option application, or
preview-pane slot creation.

## Implementation Boundary

Owner/module:

- reference-review shell/preview pane UI

Reuse/extraction decision:

- Compose reusable icon toggles and exclusive icon groups.

## Behavior

The implementation must:

- render color mode group, separator, lighting mode group, separator, then independent toggles
- bind color group to authored/inspection modes
- bind lighting group to flat/face-normal/camera-light modes
- bind independent toggles for fill, object edges, triangle wireframe, grid, axes, background, and polylines
- disable all controls when no preview payload is ready
- preserve control order and geometry across state changes

## Verification

Test strategy:

- component/shell tests assert group and toggle order
- disabled state test asserts controls are disabled without preview readiness
- ready state test asserts controls are enabled when a real preview payload is ready
- command smoke tests assert controls route through display command routing

## Manifest Assessment

Score:

- Functions/methods: 1 x 2 = 2
- Data structures/models: 0 x 1 = 0
- Dependencies/services: 4 x 1 = 4
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 4 x 1 = 4
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 2 x 2 = 4
- Total: 24.5

Split decision:

- No split needed. Score is in the split-review band because the row composes
  prerequisite components, but the implementation boundary is one ordered row.

## Refinement Status

Final leaf, blocked until Specs 63-73 are available.

## Acceptance

This specification is complete when the display-control row composes the reusable controls in the product-defined order.
