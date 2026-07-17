# Reference Review Spec 78b: Non-Renderable Preview State And Last-Good Guard (v1.0)

Status: Final leaf after review pass 1

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one non-renderable/stale preview state guard.

## Overview

Route diagnostic or non-renderable fixtures to contextual preview state and
preserve last-good render state when stale or failed preview completions arrive.

## Backlink

- [Parent Spec 78](reference-review-78-preview-lifecycle-and-non-renderable-fixture-handling-v1_0.md)
- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Scope

This specification covers non-renderable fixture state and stale/failure
last-good guards. Renderable preview lifecycle is owned by Spec 78a.

## Responsibilities

- Functions/methods:
  - non-renderable fixture state handler
  - preview stale/failure guard
- Data structures/models:
  - preview state fields for current, stale, failure, and non-renderable state
- Dependencies/services:
  - shell selected-fixture state
  - preview payload completion route
- Returns/outputs/signals:
  - non-renderable context state
  - preview diagnostic
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - contextual diagnostic text
  - stale/last-good state indicator where present
- Reusable code plan:
  - Existing code reused as-is: shell and preview visible-state plumbing
  - Additions to existing reusable library/module: none expected
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - stale failures rejected before preview mutation
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics are sanitized before display
- Performance-sensitive behavior:
  - non-renderable route must not call renderer scene replacement
- Cross-screen reusable behavior:
  - none

## Behavior

The implementation must:

- detect diagnostic/non-renderable fixtures before renderer scene replacement;
- show contextual non-renderable state;
- reject stale successes and stale failures before preview mutation;
- preserve last-good render when same-fixture failure occurs after success.

## Verification

Test strategy:

- non-renderable routing test proving renderer replacement is not called;
- stale success/failure tests;
- same-fixture failure after success test;
- manual renderable-to-non-renderable-to-renderable smoke.

## Review Score

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Split decision:

- No split. Cohesion reason: non-renderable state and last-good preservation
  are one failure-state route for the preview pane.

## Readiness Fields

App type: GUI route inside mixed app.
User/caller surface: preview pane.
Invocation route: fixture selection and preview completion.
Wiring owner/module: shell and preview widget visible-state route.
Observable result: non-renderable fixtures do not crash or blank the app.
Integration validation: preview routing tests and manual GUI smoke.
Prerequisites: Spec 77b.
Readiness blockers: none.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when diagnostic/non-renderable fixtures show
context and stale/failure preview work cannot clear last-good render state.
