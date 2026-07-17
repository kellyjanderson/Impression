# Reference Review Spec 79b: Approve/Decline Persistence And Promotion Route (v1.0)

Status: Final leaf after review pass 1

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one paired review-action persistence route.

## Overview

Ensure approve and decline persist fixture status correctly: approve moves dirty
artifacts to matching gold paths and decline leaves dirty artifacts in place.

## Backlink

- [Parent Spec 79](reference-review-79-review-workflow-persistence-smoke-v1_0.md)
- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Scope

This specification covers approve and decline actions. Notes are owned by Spec
79a; list filter and badge state are owned by Spec 79c.

## Responsibilities

- Functions/methods:
  - approve action route
  - decline action route
  - promotion validation/completion handoff
- Data structures/models:
  - fixture review status
  - artifact dirty/gold path mapping
- Dependencies/services:
  - fixture store
  - artifact promotion service
  - durable write lane
- Returns/outputs/signals:
  - approve completion
  - decline completion
  - promotion diagnostic
- UI surfaces/components:
  - approve/decline controls
- UI fields/elements:
  - action enabled/status state where present
- Reusable code plan:
  - Existing code reused as-is: promotion and status lifecycle modules
  - Additions to existing reusable library/module: optional durable write helper
    if Spec 76b allows
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none expected
- Async/concurrency behavior:
  - durable writes serialized; stale durable results remain visible diagnostics
- Destructive/write behavior:
  - approve moves artifacts; decline writes status only
- Security/privacy-sensitive behavior:
  - diagnostics avoid unrelated paths
- Performance-sensitive behavior:
  - promotion work does not block UI thread
- Cross-screen reusable behavior:
  - status feeds list and badge after completion

## Behavior

The implementation must:

- persist approved status and move dirty artifacts to matching gold paths;
- persist declined status without moving artifacts;
- report promotion failures near the review action route;
- avoid silently discarding durable completion results.

## Verification

Test strategy:

- approve test using temporary fixture/artifact copies;
- decline test proving artifacts are not moved;
- status persistence test for both actions.

## Review Score

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 24

Split decision:

- No split. Cohesion reason: approve and decline are one review-action
  persistence route with opposite artifact movement policies.

## Readiness Fields

App type: GUI route with durable file side effects.
User/caller surface: approve and decline controls.
Invocation route: button click.
Wiring owner/module: shell, promotion service, fixture store.
Observable result: approved/declined status persists and artifacts move only
for approve.
Integration validation: temporary-artifact promotion tests and manual smoke.
Prerequisites: Spec 77b.
Readiness blockers: none.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when approve and decline persist correct status
and artifact side effects through the real review action route.
