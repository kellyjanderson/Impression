# Reference Review Spec 77b: UI Handoff And Stale Completion Guard (v1.0)

Status: Final leaf after review pass 1

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one UI-thread handoff and stale-completion guard route.

## Overview

Ensure background worker completions cross into the Reference Review UI through
typed handoff and are rejected when stale before mutating selected fixture,
preview, notes, status, or fixture-list state.

## Backlink

- [Parent Spec 77](reference-review-77-non-blocking-shell-bootstrap-and-task-handoff-v1_0.md)
- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Scope

This specification covers worker completion handoff and stale-result rejection.
Startup deferral is owned by Spec 77a.

## Responsibilities

- Functions/methods:
  - worker completion handoff handler
  - owner/request currentness check
  - sanitized failure handoff
- Data structures/models:
  - existing message/result envelope or compatible kit record
- Dependencies/services:
  - task dispatcher or worker lane
  - latest-request/staleness helper
  - Qt handoff helper
- Returns/outputs/signals:
  - UI-safe completion
  - rejected stale completion
- UI surfaces/components:
  - fixture list, preview, notes, and status state routes
- UI fields/elements:
  - selected fixture state
  - visible diagnostic state
- Reusable code plan:
  - Existing code reused as-is: current async core where import-safe
  - Additions to existing reusable library/module: optional kit staleness and
    Qt handoff helpers when Spec 76b allows
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - UI mutation only after currentness check
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - worker failures are sanitized before display
- Performance-sensitive behavior:
  - stale rejection is cheap and synchronous in the UI route
- Cross-screen reusable behavior:
  - protects all selected-fixture panels

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/shell.py`
- Reference Review async core or accepted kit handoff/staleness helpers

Routes:

- worker completion to UI shell

## Behavior

The implementation must:

- route worker completions through UI-safe handoff;
- reject stale success, failure, and cancellation before UI mutation;
- sanitize worker error text before display;
- preserve current selected fixture, preview, notes, and status when stale
  completions arrive.

## Verification

Test strategy:

- focused handoff test;
- stale success/failure/cancellation tests for visible state routes.

Additional verification requirements:

- run `git diff --check`

## Review Score

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 24.5

Split decision:

- No split. Cohesion reason: this is one currentness-checked UI handoff route;
  the review challenged panel-by-panel splitting and rejected it as duplicate
  implementation of the same invariant.

## Readiness Fields

App type: GUI route inside mixed app.
User/caller surface: all selected-fixture panels receiving worker completions.
Invocation route: worker completion handoff.
Wiring owner/module: shell and async handoff helpers.
Observable result: stale completions cannot corrupt current visible state.
Integration validation: focused stale-result and handoff tests.
Prerequisites: Spec 76b.
Readiness blockers: none.

## Refinement Status

Final leaf.

## Child Specifications

None.

## Acceptance

This specification is complete when stale completions are rejected before UI
mutation across the selected-fixture state routes.
