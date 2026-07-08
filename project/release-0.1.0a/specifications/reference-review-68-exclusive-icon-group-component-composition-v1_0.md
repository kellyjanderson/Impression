# Reference Review Spec 68: Exclusive Icon Group Component Composition (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one reusable component that composes icon toggles according to an exclusive selection model.

## Overview

Compose reusable icon toggles into an exclusive icon option group component.

## Backlink

- [Reference Review Spec 65: Icon Toggle Button Visual State Component](reference-review-65-icon-toggle-button-visual-state-component-v1_0.md)
- [Reference Review Spec 66: Icon Toggle Button Interaction Contract](reference-review-66-icon-toggle-button-interaction-contract-v1_0.md)
- [Reference Review Spec 67: Exclusive Icon Group Selection Model](reference-review-67-exclusive-icon-group-selection-model-v1_0.md)

## Scope

This specification covers rendering child icon toggle controls and synchronizing
their checked/enabled states with the exclusive selection model.

It does not cover preview-specific color or lighting semantics.

## Implementation Boundary

Owner/module:

- reusable workbench UI component layer

Reuse/extraction decision:

- Compose existing icon toggle component; do not duplicate toggle state rendering.

## Behavior

The implementation must:

- render adjacent icon option buttons in supplied order
- set exactly one child as checked
- emit selected option id when a child is activated
- disable all children when the group is disabled
- leave separators to the containing toolbar

## Verification

Test strategy:

- child checked-state synchronization tests
- click emits selected option id
- disabled group prevents child activation
- component-gallery state coverage

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 3 x 1 = 3
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 24

Split decision:

- No split needed. Score is in the split-review band because the component composes several reusable parts, but the leaf has one responsibility: render an exclusive icon group.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when reusable exclusive icon groups can be composed from icon toggles.
