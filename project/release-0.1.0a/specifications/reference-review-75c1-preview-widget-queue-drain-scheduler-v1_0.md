# Reference Review Spec 75c1: Preview Widget Queue Drain Scheduler (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one widget-owned event-loop drain scheduler for queued preview commands.

## Overview

Add the preview widget/controller drain scheduler that ensures queued preview
commands are applied on the Qt UI thread and not recursively.

## Backlink

- [Reference Review Spec 75c: Preview Widget Command Drain And Renderer Mutation Boundary](reference-review-75c-preview-widget-command-drain-and-renderer-mutation-boundary-v1_0.md)

## Source Manifest

- This leaf is split from ad hoc remediation spec 75c.
- Manifest score: 22.5

## Scope

This specification covers:

- schedule queue drain
- drain command queue
- prevent nested drain scheduling
- clear pending commands on close

## Responsibilities

- Functions/methods:
  - schedule queue drain
  - drain command queue
  - guard nested drain
  - clear pending commands
- Data structures/models:
  - queue drain state
- Dependencies/services:
  - Spec 75a coalescing queue
  - `PreviewRendererLifecycleWidget`
- Returns/outputs/signals:
  - command application result
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - preview viewport
- Reusable code plan:
  - Existing code reused as-is:
    - render command queue
  - Additions to existing reusable library/module:
    - preview widget drain methods
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - drain is Qt UI-thread-only
  - drain scheduling is non-reentrant
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - one scheduled drain handles current coalesced state
- Cross-screen reusable behavior:
  - none

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/preview_widget.py`

Routes:

- shell enqueues commands
- widget schedules a drain

Reuse/extraction decision:

- do not add renderer mutation to shell

## Data And Defaults

Chosen defaults / parameters:

- drain is scheduled with Qt event-loop delivery
- pending commands are cleared before renderer disposal

Data ownership:

- preview widget owns drain state

Open questions and resolved assumptions:

- native VTK rendering remains inside the existing renderer surface

Implementation prerequisites:

- Spec 75a1
- Spec 75a2

## Behavior

The implementation must:

- schedule exactly one pending drain while work is queued
- avoid nested drain calls from command application
- clear pending commands on close

## Verification

Test strategy:

- widget tests for one scheduled drain, non-reentrant behavior, and close
  cleanup

Additional verification requirements:

- run Reference Review UI shell tests

## Manifest Assessment

Score:

- Functions/methods: 4 x 2 = 8
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Split decision:

- No split needed. Cohesion reason: this leaf only schedules and drains
  commands; actual renderer application details are in Spec 75c2.

## Refinement Status

Final leaf.

## Child Specifications

None.

## Acceptance

This specification is complete when preview command drains are scheduled and
owned by the preview widget.
