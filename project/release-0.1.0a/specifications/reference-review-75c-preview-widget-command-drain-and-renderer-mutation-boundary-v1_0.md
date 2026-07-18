# Reference Review Spec 75c: Preview Widget Command Drain And Renderer Mutation Boundary (v1.0)

## Overview

Make the preview widget/controller the only component that drains render
commands and mutates the renderer surface.

## Backlink

- [Reference Review Spec 75: Preview Render Command Queue](reference-review-75-preview-render-command-queue-v1_0.md)

## Source Manifest

- This leaf is derived from ad hoc remediation spec 75.
- Manifest score: 43 before split

## Scope

This specification covers:

- preview widget command-drain method
- Qt event-loop scheduling for queue drains
- payload command application
- display-options command application without payload JSON reload
- lifecycle/failure command application
- camera reset command application

## Responsibilities

- Functions/methods:
  - schedule queue drain
  - drain command queue
  - apply payload command
  - apply display command
  - apply failure/lifecycle command
  - apply camera command
- Data structures/models:
  - preview queue state consumed by widget
- Dependencies/services:
  - `PreviewRenderCommandQueue`
  - `PreviewRendererLifecycleWidget`
  - `QtPreviewSurface`
- Returns/outputs/signals:
  - command application result
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - preview viewport
  - preview status text
  - reset view command
- Reusable code plan:
  - Existing code reused as-is:
    - shared Qt preview surface
  - Additions to existing reusable library/module:
    - preview widget command-drain methods
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - renderer mutation is UI-thread-only
  - queue drain is scheduled, not nested recursively
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - sanitized diagnostics only
- Performance-sensitive behavior:
  - display commands must reuse decoded current datasets
  - repeated toggles must result in one coherent render application per drain
- Cross-screen reusable behavior:
  - none

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/preview_widget.py`
- `src/impression/preview_qt.py` only if a small non-reset display update hook
  is required

Routes:

- shell enqueues commands
- widget schedules and drains
- renderer surface receives final coherent state

Reuse/extraction decision:

- do not add rendering logic to shell
- do not add a second renderer path

## Data And Defaults

Chosen defaults / parameters:

- drain is scheduled with Qt event-loop delivery
- display command does not reset camera
- payload command aligns camera only for new current payload

Data ownership:

- widget owns decoded current datasets
- renderer owns render surface lifetime

Open questions and resolved assumptions:

- software fallback may continue to project in `paintEvent`; native Qt path
  should remain shared-preview based

Implementation prerequisites:

- Spec 75a
- Spec 75b

## Behavior

The implementation must:

- make renderer mutation reachable only through queue drain
- keep renderer long-lived
- apply display options without re-reading payload files
- avoid duplicate render/apply passes for one display toggle
- preserve current ready/failure visible state semantics

## Verification

Test strategy:

- widget tests with fake renderer proving one scene update after coalesced
  toggles
- widget tests proving payload JSON is not reread for display-only commands
- lifecycle tests proving renderer is not recreated

Additional verification requirements:

- run Reference Review UI shell tests and preview controller tests

## Manifest Assessment

Score:

- Functions/methods: 6 x 2 = 12
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 4 x 1 = 4
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 3 x 2 = 6
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 3 x 1 = 3
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 3 x 3 = 9
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 43

Split decision:

- Split required. Split into Spec 75c1 for queue drain scheduling and Spec
  75c2 for efficient queued command application.

## Refinement Status

Split parent. Do not implement this document directly.

## Child Specifications

- [Reference Review Spec 75c1: Preview Widget Queue Drain Scheduler](reference-review-75c1-preview-widget-queue-drain-scheduler-v1_0.md)
- [Reference Review Spec 75c2: Preview Command Application Efficiency](reference-review-75c2-preview-command-application-efficiency-v1_0.md)

## Acceptance

This specification is complete when:

- Specs 75c1 and 75c2 are implemented and verified
