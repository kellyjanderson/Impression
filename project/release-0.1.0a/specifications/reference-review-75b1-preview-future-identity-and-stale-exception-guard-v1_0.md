# Reference Review Spec 75b1: Preview Future Identity And Stale Exception Guard (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one shell-owned future identity guard that prevents stale exceptions from mutating preview state.

## Overview

Track preview futures by request identity so stale future exceptions cannot
clear or overwrite the current preview.

## Backlink

- [Reference Review Spec 75b: Qt Queued Completion Handoff And Shell Routing](reference-review-75b-qt-queued-completion-handoff-and-shell-routing-v1_0.md)

## Source Manifest

- This leaf is split from ad hoc remediation spec 75b.
- Manifest score: 22.5

## Scope

This specification covers:

- future-to-request identity tracking
- stale future exception rejection
- current future exception conversion into a queued failure command

## Responsibilities

- Functions/methods:
  - track accepted future identity
  - classify future exception freshness
  - enqueue current failure command
- Data structures/models:
  - future identity map
- Dependencies/services:
  - `PreviewPayloadProcessController`
  - Spec 75a command queue
- Returns/outputs/signals:
  - stale/current exception decision
- UI surfaces/components:
  - Reference Review shell
- UI fields/elements:
  - preview unavailable text
- Reusable code plan:
  - Existing code reused as-is:
    - payload process controller active identity
  - Additions to existing reusable library/module:
    - shell future tracking helpers
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - stale exceptions cannot mutate preview state
  - future handling only enqueues commands
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - visible exception text remains sanitized
- Performance-sensitive behavior:
  - exception handling returns quickly after enqueue/discard
- Cross-screen reusable behavior:
  - none

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/shell.py`

Routes:

- payload launch stores accepted future identity
- polling/callback exception handling consults identity before enqueue

Reuse/extraction decision:

- keep `ProcessPoolExecutor`; do not add external queue dependencies

## Data And Defaults

Chosen defaults / parameters:

- untracked future exceptions are ignored for visible preview state
- stale future exceptions are ignored for visible preview state

Data ownership:

- shell owns future identity map

Open questions and resolved assumptions:

- diagnostics may be retained later, but visible mutation is forbidden for
  stale exceptions

Implementation prerequisites:

- Spec 75a1
- Spec 75a2

## Behavior

The implementation must:

- prevent stale future exceptions from calling `clear_preview`
- enqueue a current failure command for current future exceptions

## Verification

Test strategy:

- shell tests for stale exception and current exception paths

Additional verification requirements:

- run Reference Review shell tests

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
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
- Async/concurrency behavior: 2 x 3 = 6
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Split decision:

- No split needed. Cohesion reason: this leaf only tracks future identity and
  prevents stale exceptions from mutating visible preview state.

## Refinement Status

Final leaf.

## Child Specifications

None.

## Acceptance

This specification is complete when stale future exceptions cannot clear the
current preview.
