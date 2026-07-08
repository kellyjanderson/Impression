# Reference Review Spec 65: Icon Toggle Button Visual State Component (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one reusable visual component for icon toggle states.

## Overview

Implement the reusable icon-only toggle button's visual state behavior.

## Backlink

- [Reference Review Spec 64: Preview Display Icon Metadata Registry](reference-review-64-preview-display-icon-metadata-registry-v1_0.md)
- [Architecture: Reference Review Preview Display Controls Product Definition](../architecture/reference-review-preview-display-controls-product-definition.md)

## Scope

This specification covers:

- fixed icon-only geometry
- off, hover, pressed, checked, focus, and disabled visual states
- selected state visibility without relying only on color

This specification does not cover command routing or preview-specific state.

## Implementation Boundary

Owner/module:

- reusable workbench UI component layer

Reuse/extraction decision:

- Add a generic reusable visual component rather than preview-only buttons.

## Behavior

The implementation must:

- render an icon-only control with stable size
- show checked and unchecked state clearly
- show disabled state clearly
- provide a visible focus state
- never show text inside the control row

## Verification

Test strategy:

- component state tests for checked, unchecked, disabled, hover/focus where supported
- component-gallery scenario for visual review
- geometry tests assert state changes do not resize the control

## Manifest Assessment

Score:

- Functions/methods: 1 x 2 = 2
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 4 x 1 = 4
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Split decision:

- No split needed. Cohesive visual-state component leaf.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when the reusable toggle visual state component exists and is state-covered.
