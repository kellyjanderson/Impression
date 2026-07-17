# Reference Review Spec 79a: Notes Persistence Route (v1.0)

Status: Final leaf after review pass 1

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one selected-fixture notes load/save route.

## Overview

Ensure notes load for the selected fixture, save in real time to the fixture
record or database, and stale notes completions cannot update a newer selected
fixture.

## Backlink

- [Parent Spec 79](reference-review-79-review-workflow-persistence-smoke-v1_0.md)
- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Scope

This specification covers notes only. Approve/decline is owned by Spec 79b;
filter and badge state are owned by Spec 79c.

## Responsibilities

- Functions/methods:
  - selected fixture notes load route
  - notes save route
  - stale notes completion guard
- Data structures/models:
  - selected fixture notes
- Dependencies/services:
  - fixture record file or database store
  - durable write lane
- Returns/outputs/signals:
  - notes load value
  - notes save completion
- UI surfaces/components:
  - notes tab/panel
- UI fields/elements:
  - notes text
- Reusable code plan:
  - Existing code reused as-is: current notes lifecycle
  - Additions to existing reusable library/module: optional durable write helper
    if Spec 76b allows
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none expected
- Async/concurrency behavior:
  - writes serialized; stale completion rejected
- Destructive/write behavior:
  - notes write to fixture persistence
- Security/privacy-sensitive behavior:
  - notes remain in fixture persistence path
- Performance-sensitive behavior:
  - notes writes do not block UI thread
- Cross-screen reusable behavior:
  - selected fixture notes route

## Behavior

The implementation must:

- load notes on selected fixture change;
- save notes in real time;
- serialize notes writes;
- reject stale notes completions before visible state mutation.

## Verification

Test strategy:

- fixture-store notes load/save test;
- stale selected-fixture notes completion test;
- manual note edit/select-away/select-back smoke.

## Review Score

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Split decision:

- No split. Cohesion reason: notes load, save, and stale guard form one
  selected-fixture notes route.

## Readiness Fields

App type: GUI route with durable file side effect.
User/caller surface: notes panel.
Invocation route: fixture selection and notes edit.
Wiring owner/module: shell and notes lifecycle modules.
Observable result: selected fixture notes persist and reload.
Integration validation: notes persistence tests and manual smoke.
Prerequisites: Spec 77b.
Readiness blockers: none.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when notes persist in real time for the selected
fixture and stale completions cannot corrupt newer fixture notes.
