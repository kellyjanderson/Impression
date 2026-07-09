# Reference Review Spec 75a2: Preview Render Coalescing Queue (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one bounded coalescing queue using the command records from Spec 75a1.

## Overview

Add the bounded/coalescing queue that stores only the latest preview render
command per lane.

## Backlink

- [Reference Review Spec 75a1: Preview Render Command Record Contract](reference-review-75a1-preview-render-command-record-contract-v1_0.md)

## Source Manifest

- This leaf is split from ad hoc remediation spec 75a.
- Manifest score: 20.5

## Scope

This specification covers:

- `PreviewRenderCommandQueue`
- `PreviewRenderQueueState`
- lane mapping
- command enqueue/coalescing
- deterministic drain order
- clearing pending commands

## Responsibilities

- Functions/methods:
  - enqueue command
  - coalesce command by lane
  - drain pending commands
  - clear pending commands
- Data structures/models:
  - queue state
- Dependencies/services:
  - Spec 75a1 command records
- Returns/outputs/signals:
  - accepted/replaced/rejected command result
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is:
    - Spec 75a1 command records
  - Additions to existing reusable library/module:
    - add queue behavior to `preview_render_queue.py`
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - queue stores bounded latest commands only
  - queue does not execute renderer work
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - repeated display commands collapse to one latest command
- Cross-screen reusable behavior:
  - none

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/preview_render_queue.py`

Routes:

- shell and preview widget/controller will use this queue in later leaves

Reuse/extraction decision:

- implement a small typed coalescing queue rather than using a general
  third-party message queue

## Data And Defaults

Chosen defaults / parameters:

- lanes: `payload`, `display`, `lifecycle`, `camera`
- latest command wins within each lane
- drain order: lifecycle, payload, display, camera

Data ownership:

- queue owns pending commands only

Open questions and resolved assumptions:

- stale identity rejection is supported by command fields but enforced in
  shell/widget leaves

Implementation prerequisites:

- Spec 75a1

## Behavior

The implementation must:

- coalesce repeated commands by lane
- return a result indicating accepted, replaced, or rejected state
- expose queue state for tests

## Verification

Test strategy:

- unit tests for lane mapping, coalescing, drain order, and clearing pending
  commands

Additional verification requirements:

- run queue-related tests and `git diff --check`

## Manifest Assessment

Score:

- Functions/methods: 4 x 2 = 8
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 2 x 3 = 6
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Split decision:

- No split needed. Cohesion reason: this leaf implements one bounded
  coalescing queue and does not mutate UI or renderer state.

## Refinement Status

Final leaf.

## Child Specifications

None.

## Acceptance

This specification is complete when the queue behavior is implemented and
covered by unit tests.
