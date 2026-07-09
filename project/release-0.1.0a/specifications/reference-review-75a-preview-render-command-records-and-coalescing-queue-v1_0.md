# Reference Review Spec 75a: Preview Render Command Records And Coalescing Queue (v1.0)

## Overview

Add the typed command records and bounded coalescing queue used by the
Reference Review preview path.

## Backlink

- [Reference Review Spec 75: Preview Render Command Queue](reference-review-75-preview-render-command-queue-v1_0.md)

## Source Manifest

- This leaf is derived from ad hoc remediation spec 75.
- Manifest score: 26.5 before split

## Scope

This specification covers:

- `PreviewRenderCommandKind`
- `PreviewRenderCommand`
- `PreviewRenderQueueState`
- `PreviewRenderCommandResult`
- `PreviewRenderCommandQueue`
- deterministic replacement/coalescing rules for payload, display, lifecycle,
  reset, and failure commands

## Responsibilities

- Functions/methods:
  - enqueue command
  - coalesce command by lane
  - drain pending commands
  - clear pending commands
- Data structures/models:
  - render command kind
  - render command
  - queue state
  - command result
- Dependencies/services:
  - existing payload identity records
  - existing `PreviewDisplayOptions`
- Returns/outputs/signals:
  - accepted/replaced/rejected command result
- UI surfaces/components:
  - preview pane command source
- UI fields/elements:
  - none directly
- Reusable code plan:
  - New reusable library/module to create:
    - `src/impression/devtools/reference_review/ui/preview_render_queue.py`
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - queue records are immutable and safe to pass between producer paths
  - queue stores bounded latest commands only
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - no raw paths beyond already-sanitized diagnostics
- Performance-sensitive behavior:
  - repeated display commands collapse to one latest command
- Cross-screen reusable behavior:
  - none

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/preview_render_queue.py`

Routes:

- imported by preview shell and preview widget/controller only

Reuse/extraction decision:

- create a small project-local typed queue rather than adding an external
  message-queue dependency

## Data And Defaults

Chosen defaults / parameters:

- lanes: `payload`, `display`, `lifecycle`, `camera`
- latest command wins within each lane
- stale identity rejection is supported by command fields but applied by later
  leaves

Data ownership:

- queue owns pending commands only

Open questions and resolved assumptions:

- no external queue library is required for this leaf

Implementation prerequisites:

- existing `PreviewDisplayOptions`
- existing payload identity fields

## Behavior

The implementation must:

- create frozen command records
- reject unknown command kinds
- coalesce repeated commands by lane
- drain commands in deterministic order:
  - lifecycle/failure
  - payload
  - display
  - camera
- expose queue state for tests

## Verification

Test strategy:

- unit tests for record validation, lane mapping, coalescing, drain order, and
  clearing pending commands

Additional verification requirements:

- run focused Reference Review async/UI tests and `git diff --check`

## Manifest Assessment

Score:

- Functions/methods: 4 x 2 = 8
- Data structures/models: 4 x 1 = 4
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 2 x 3 = 6
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 26.5

Split decision:

- Split required. Split into Spec 75a1 for command records and Spec 75a2 for
  queue coalescing behavior.

## Refinement Status

Split parent. Do not implement this document directly.

## Child Specifications

- [Reference Review Spec 75a1: Preview Render Command Record Contract](reference-review-75a1-preview-render-command-record-contract-v1_0.md)
- [Reference Review Spec 75a2: Preview Render Coalescing Queue](reference-review-75a2-preview-render-coalescing-queue-v1_0.md)

## Acceptance

This specification is complete when:

- Specs 75a1 and 75a2 are implemented and verified
