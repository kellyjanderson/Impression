# Reference Review Spec 75b2: Preview Completion To Command Routing (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one routing change from current payload completion events into render queue commands.

## Overview

Convert current preview payload successes and failures into render queue
commands without directly mutating preview widgets.

## Backlink

- [Reference Review Spec 75b: Qt Queued Completion Handoff And Shell Routing](reference-review-75b-qt-queued-completion-handoff-and-shell-routing-v1_0.md)

## Source Manifest

- This leaf is split from ad hoc remediation spec 75b.
- Manifest score: 24

## Scope

This specification covers:

- current successful payload completion to payload command
- current failed payload completion to failure command
- stale successful/failed completion discard before enqueue
- display-control readiness remains disabled until payload command applies

## Responsibilities

- Functions/methods:
  - payload-ready command enqueue
  - payload-failed command enqueue
  - stale completion discard
- Data structures/models:
  - completion-to-command mapping
- Dependencies/services:
  - `PreviewPayloadProcessController`
  - Spec 75a command queue
- Returns/outputs/signals:
  - queued payload/failure command result
- UI surfaces/components:
  - Reference Review shell
- UI fields/elements:
  - display-control enabled state
- Reusable code plan:
  - Existing code reused as-is:
    - payload process controller handoff decisions
    - render command queue
  - Additions to existing reusable library/module:
    - shell command enqueue helpers
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - worker completions enqueue commands only
- Destructive/write behavior:
  - payload cleanup remains owned by payload controller
- Security/privacy-sensitive behavior:
  - visible diagnostics remain sanitized by existing payload controller
- Performance-sensitive behavior:
  - completion handling does not apply renderer work directly
- Cross-screen reusable behavior:
  - none

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/shell.py`

Routes:

- `PreviewPayloadProcessController.handle_completion` returns event
- shell converts current event to queued command

Reuse/extraction decision:

- use existing payload controller staleness decisions

## Data And Defaults

Chosen defaults / parameters:

- successful command does not enable controls until applied by widget drain
- failed command disables controls through widget drain

Data ownership:

- shell owns completion-to-command routing only

Open questions and resolved assumptions:

- renderer mutation belongs to Spec 75c leaves

Implementation prerequisites:

- Spec 75a1
- Spec 75a2
- Spec 75b1

## Behavior

The implementation must:

- stop direct `set_preview_payload` and `clear_preview` calls from completion
  handlers
- enqueue payload/failure commands instead

## Verification

Test strategy:

- shell tests for current success, current failure, stale success, and stale
  failure routing

Additional verification requirements:

- run Reference Review shell and payload-controller tests

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 24

Split decision:

- No split needed. Cohesion reason: this leaf only routes current/stale
  completion events to command queue outcomes.

## Refinement Status

Final leaf.

## Child Specifications

None.

## Acceptance

This specification is complete when payload completions enqueue commands and
do not mutate renderer widgets directly.
