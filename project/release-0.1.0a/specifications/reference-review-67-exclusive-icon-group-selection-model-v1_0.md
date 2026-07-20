# Reference Review Spec 67: Exclusive Icon Group Selection Model (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one reusable state model for exactly-one icon option selection.

## Overview

Define reusable state records and selection rules for an exclusive icon option group.

## Backlink

- [Reference Review Spec 66: Icon Toggle Button Interaction Contract](reference-review-66-icon-toggle-button-interaction-contract-v1_0.md)

## Scope

This specification covers option records, selected option state, disabled state,
and exactly-one selection rules.

It does not cover visual layout or child button composition.

## Implementation Boundary

Owner/module:

- reusable workbench UI component/state layer

Reuse/extraction decision:

- Add a generic exclusive icon group state helper for reuse by color and lighting groups.

## Behavior

The implementation must:

- preserve stable option order
- require exactly one selected option when enabled and options exist
- replace the selected option when a different option is chosen
- reject unknown option ids with deterministic diagnostics or exceptions

## Verification

Test strategy:

- three-option selection tests
- unknown id rejection tests
- disabled state tests
- stable-order tests

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 0 x 0.5 = 0
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 19

Split decision:

- No split needed. Cohesive selection model leaf.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when exactly-one selection can be tested without rendering controls.
