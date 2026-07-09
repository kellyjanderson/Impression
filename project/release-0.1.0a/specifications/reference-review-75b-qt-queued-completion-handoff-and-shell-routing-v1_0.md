# Reference Review Spec 75b: Qt Queued Completion Handoff And Shell Routing (v1.0)

## Overview

Route preview payload completions and failures into the preview command queue
without directly mutating preview widgets from future polling or worker
callbacks.

## Backlink

- [Reference Review Spec 75: Preview Render Command Queue](reference-review-75-preview-render-command-queue-v1_0.md)

## Source Manifest

- This leaf is derived from ad hoc remediation spec 75.
- Manifest score: 40 before split

## Scope

This specification covers:

- converting successful payload completions into payload render commands
- converting current failures into failure render commands
- discarding stale successes and stale failures before visible preview mutation
- ensuring future exceptions are identity-aware and cannot clear the current
  preview
- using Qt queued delivery or UI-thread polling only as an enqueue mechanism

## Responsibilities

- Functions/methods:
  - future completion handling
  - payload-ready command enqueue
  - payload-failed command enqueue
  - stale exception rejection
  - preview display-control enable/disable routing
- Data structures/models:
  - future-to-request identity map
  - completion command
- Dependencies/services:
  - `PreviewPayloadProcessController`
  - `PreviewRenderCommandQueue`
  - Qt signal/slot queued delivery
- Returns/outputs/signals:
  - queued payload/failure command
  - rejected stale completion diagnostic
- UI surfaces/components:
  - Reference Review shell
- UI fields/elements:
  - preview unavailable/loading text
  - display-control enabled state
- Reusable code plan:
  - Existing code reused as-is:
    - payload process controller
    - render command queue
  - Additions to existing reusable library/module:
    - shell routing helpers
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - no worker completion directly mutates Qt widgets
  - stale completions and exceptions are rejected before enqueue
- Destructive/write behavior:
  - payload cleanup remains owned by payload controller
- Security/privacy-sensitive behavior:
  - visible diagnostics remain sanitized
- Performance-sensitive behavior:
  - polling or callbacks enqueue a command and return quickly
- Cross-screen reusable behavior:
  - none

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/shell.py`
- `src/impression/devtools/reference_review/async_core/qt_handoff.py`

Routes:

- `PreviewPayloadProcessController.handle_completion` returns events that shell
  converts to render commands
- future exception path also routes through identity-aware command handling

Reuse/extraction decision:

- keep `ProcessPoolExecutor` payload workers
- use Qt queued signals when crossing threads; polling may remain only if it
  enqueues commands and does not mutate widgets

## Data And Defaults

Chosen defaults / parameters:

- future identity is stored when launch returns an accepted future
- exceptions from untracked or stale futures are ignored for visible preview
  state

Data ownership:

- shell owns future tracking
- payload controller owns active identity
- queue owns pending render commands

Open questions and resolved assumptions:

- no external queue dependency is required

Implementation prerequisites:

- Spec 75a

## Behavior

The implementation must:

- stop direct calls to preview widget `set_preview_payload` and `clear_preview`
  from future polling
- enqueue payload/failure commands instead
- prevent stale exceptions from clearing current preview state
- keep preview display controls disabled until a current payload command is
  applied successfully

## Verification

Test strategy:

- shell tests for stale success, stale failure, stale exception, and current
  failure behavior

Additional verification requirements:

- run Reference Review shell and payload-controller tests

## Manifest Assessment

Score:

- Functions/methods: 5 x 2 = 10
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 4 x 3 = 12
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 40

Split decision:

- Split required. Split into Spec 75b1 for future identity/stale exception
  guarding and Spec 75b2 for current completion-to-command routing.

## Refinement Status

Split parent. Do not implement this document directly.

## Child Specifications

- [Reference Review Spec 75b1: Preview Future Identity And Stale Exception Guard](reference-review-75b1-preview-future-identity-and-stale-exception-guard-v1_0.md)
- [Reference Review Spec 75b2: Preview Completion To Command Routing](reference-review-75b2-preview-completion-to-command-routing-v1_0.md)

## Acceptance

This specification is complete when:

- Specs 75b1 and 75b2 are implemented and verified
