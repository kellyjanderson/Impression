# Reference Review Spec 78a: Renderable Preview Lifecycle (v1.0)

Status: Final leaf after review pass 1

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one renderable fixture preview lifecycle route.

## Overview

Stabilize the renderable fixture preview path: one renderer per preview widget
lifetime, off-thread payload preparation, UI/render-thread scene mutation, and
display-control routing to the current preview surface.

## Backlink

- [Parent Spec 78](reference-review-78-preview-lifecycle-and-non-renderable-fixture-handling-v1_0.md)
- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Scope

This specification covers renderable artifacts and display-control routing.
Non-renderable fixtures and stale/failure last-good behavior are owned by Spec
78b.

## Responsibilities

- Functions/methods:
  - selected fixture to preview request route
  - renderable payload completion handler
  - display-control command handler
- Data structures/models:
  - preview payload result records
  - current preview identity/state
- Dependencies/services:
  - preview payload builder/controller
  - preview widget
  - Impression preview semantics
- Returns/outputs/signals:
  - payload application command
  - display-control command
- UI surfaces/components:
  - preview pane
  - display-control button row
- UI fields/elements:
  - preview visible state
  - display-control toggles
- Reusable code plan:
  - Existing code reused as-is: preview widget, payload builder, display controls
  - Additions to existing reusable library/module: optional kit display records
    if Spec 76b allows
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - payload work off-thread; renderer mutation UI/render-thread only
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - sanitized diagnostics for payload failure
- Performance-sensitive behavior:
  - renderer is not recreated per fixture selection
- Cross-screen reusable behavior:
  - display controls apply to every renderable preview

## Behavior

The implementation must:

- preserve one live renderer per preview widget lifetime;
- prepare renderable payloads off the UI thread;
- apply payloads only through the UI/render thread;
- keep display-control commands functional for the current preview surface.

## Verification

Test strategy:

- renderable artifact payload routing test;
- renderer lifetime test using a preview widget double where needed;
- display-control command routing test;
- manual real-render smoke.

## Review Score

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Split decision:

- No split. Cohesion reason: renderable payload application and display-control
  routing are one current-preview mutation route.

## Readiness Fields

App type: GUI route inside mixed app.
User/caller surface: preview pane and display-control button row.
Invocation route: fixture selection and display-control toggle.
Wiring owner/module: shell, preview widget, payload builder.
Observable result: renderable fixtures display and controls affect the current
preview.
Integration validation: payload tests, display-control route tests, manual
real-render smoke.
Prerequisites: Spec 76b and Spec 77b.
Readiness blockers: none.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when renderable fixtures preview through a
stable widget-owned renderer and display controls still route to that preview.
