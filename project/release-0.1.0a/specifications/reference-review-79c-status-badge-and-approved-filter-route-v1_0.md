# Reference Review Spec 79c: Status Badge And Approved Filter Route (v1.0)

Status: Final leaf after review pass 1

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one visible fixture-status projection route.

## Overview

Ensure fixture review status drives the selected-fixture badge and the
show-approved list filter.

## Backlink

- [Parent Spec 79](reference-review-79-review-workflow-persistence-smoke-v1_0.md)
- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Scope

This specification covers status badge and show-approved filtering only.
Status writes are owned by Spec 79b.

## Responsibilities

- Functions/methods:
  - show-approved filter route
  - status badge state update route
- Data structures/models:
  - fixture review status
  - fixture list filter state
- Dependencies/services:
  - fixture list model
  - selected fixture state
- Returns/outputs/signals:
  - filtered fixture list update
  - visible badge state
- UI surfaces/components:
  - fixture list
  - status badge
  - show-approved checkbox
- UI fields/elements:
  - checkbox checked state
  - fixture row visibility
  - badge label/color state
- Reusable code plan:
  - Existing code reused as-is: shell fixture list/status model
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - selected fixture status updates route through UI state
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - filtering remains cheap for current fixture file scale
- Cross-screen reusable behavior:
  - fixture status projects to list and selected context

## Behavior

The implementation must:

- hide approved fixtures by default;
- show approved fixtures when show-approved is checked;
- keep declined and unreviewed fixtures visible;
- update the selected fixture badge to approved, declined, or unreviewed.

## Verification

Test strategy:

- fixture list filter tests;
- selected fixture badge view-model tests;
- manual checkbox and badge smoke.

## Review Score

- Functions/methods: 1 x 2 = 2
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 2 x 2 = 4
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Split decision:

- No split. Cohesion reason: badge and filter are the same visible projection
  of fixture status; splitting would create two incomplete projections of one
  state contract.

## Readiness Fields

App type: GUI route.
User/caller surface: fixture list, show-approved checkbox, status badge.
Invocation route: fixture selection, status update, checkbox toggle.
Wiring owner/module: Reference Review shell/list model.
Observable result: approved filtering and selected status badge match fixture
status.
Integration validation: shell model tests and manual GUI smoke.
Prerequisites: Spec 79b.
Readiness blockers: none.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when fixture status projects correctly to both
the selected badge and approved-filter behavior.
