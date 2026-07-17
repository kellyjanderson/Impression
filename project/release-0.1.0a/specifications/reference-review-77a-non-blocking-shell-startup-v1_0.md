# Reference Review Spec 77a: Non-Blocking Shell Startup (v1.0)

Status: Final leaf after review pass 1

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one shell-startup deferral boundary.

## Overview

Ensure app startup creates the shell, UI hierarchy, lightweight models, and
deferred task triggers without synchronously importing fixture sources,
building models, tessellating, constructing preview scene content, scanning
large fixture roots, or writing durable state.

## Backlink

- [Parent Spec 77](reference-review-77-non-blocking-shell-bootstrap-and-task-handoff-v1_0.md)
- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Scope

This specification covers startup deferral only. Worker completion handoff and
stale-result rejection are owned by Spec 77b.

## Responsibilities

- Functions/methods:
  - shell startup orchestration
  - deferred fixture refresh trigger
- Data structures/models:
  - initial shell/selected-fixture state
- Dependencies/services:
  - Qt event loop
  - fixture refresh lane trigger
- Returns/outputs/signals:
  - shell-ready state
  - deferred refresh signal
- UI surfaces/components:
  - shell
  - fixture list placeholder
  - preview placeholder
- UI fields/elements:
  - initial selection state
  - loading/empty state where present
- Reusable code plan:
  - Existing code reused as-is: shell scaffolding
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - expensive work starts after event loop startup
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - startup remains responsive
- Cross-screen reusable behavior:
  - shell startup protects all panels

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/shell.py`

Routes:

- app startup
- deferred fixture refresh trigger

## Behavior

The implementation must:

- start with no selected fixture until fixture data is safely available;
- avoid fixture source import and preview scene construction during bootstrap;
- defer fixture refresh until the event loop is alive;
- expose shell-ready/placeholder state suitable for tests.

## Verification

Test strategy:

- focused shell bootstrap test with doubles proving expensive calls were not
  made;
- manual launch smoke confirming responsive startup.

Additional verification requirements:

- run `git diff --check`

## Review Score

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 3 x 2 = 6
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 24.5

Split decision:

- No split. Cohesion reason: although the score is high, every item is part of
  one startup deferral boundary and has one shell-bootstrap verification route.

## Readiness Fields

App type: mixed GUI and console entrypoint.
User/caller surface: Reference Review app window.
Invocation route: shell startup.
Wiring owner/module: `src/impression/devtools/reference_review/ui/shell.py`.
Observable result: shell opens responsive before expensive fixture work.
Integration validation: shell bootstrap test and manual launch smoke.
Prerequisites: Spec 76a and Spec 76b.
Readiness blockers: none.

## Refinement Status

Final leaf.

## Child Specifications

None.

## Acceptance

This specification is complete when shell startup is non-blocking and fixture
work begins only after the event loop is alive.
