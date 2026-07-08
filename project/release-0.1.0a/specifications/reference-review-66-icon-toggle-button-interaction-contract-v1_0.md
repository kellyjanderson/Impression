# Reference Review Spec 66: Icon Toggle Button Interaction Contract (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one reusable interaction contract for icon toggle activation and accessibility.

## Overview

Implement reusable icon toggle interaction behavior.

## Backlink

- [Reference Review Spec 65: Icon Toggle Button Visual State Component](reference-review-65-icon-toggle-button-visual-state-component-v1_0.md)

## Scope

This specification covers:

- click/activation behavior
- command emission
- checked/enabled property synchronization
- tooltip and accessible name assignment

This specification does not cover the toggle's visual state implementation.

## Implementation Boundary

Owner/module:

- reusable workbench UI component layer

Reuse/extraction decision:

- Extend the reusable icon toggle component with interaction and accessibility behavior.

## Behavior

The implementation must:

- emit one deterministic command when activated while enabled
- reject activation while disabled
- expose tooltip and accessible name
- let the owner set checked and enabled state
- avoid preview renderer imports or mutations

## Verification

Test strategy:

- enabled click emits one command
- disabled click emits no command
- tooltip and accessible-name properties are set
- checked state can be externally controlled

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 3 x 1 = 3
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Split decision:

- No split needed. Cohesive activation/accessibility leaf.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when icon toggles can emit commands accessibly without preview-specific behavior.
