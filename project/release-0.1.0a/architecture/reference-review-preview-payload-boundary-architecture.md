# Reference Review Preview Payload Boundary Architecture

## Overview

This supplemental architecture document defines how selected fixtures become
render-ready preview payloads without blocking the Reference Review Workbench
UI thread or letting worker code own renderer state.

The preview payload boundary sits between fixture/source loading and the
embedded `ImpressionPreviewWidget`.

## Parent And Related Architecture

- [Reference Review Preview Remediation Plan](reference-review-preview-remediation-plan.md)
- [Reference Review Preview Engine Sharing Architecture](reference-review-preview-engine-sharing-architecture.md)
- [Reference Review Preview Qt Wrapper Architecture](reference-review-preview-qt-wrapper-architecture.md)
- [Reference Review Async Concurrency](reference-review-async-concurrency.md)
- [Reference Review Fixture Source Contract](reference-review-fixture-source-contract.md)

## Boundary Model

```text
fixture source record
-> preview payload request
-> non-UI payload builder
-> immutable/file-backed PreviewPayload
-> typed Qt handoff
-> stale-result guard
-> ImpressionPreviewWidget.set_preview_payload(...)
```

The payload boundary prevents three bad outcomes:

- blocking the Qt event loop while loading or tessellating source models
- returning live VTK/PyVista/Qt objects from workers
- applying old preview results after the selected fixture changes

## Target Components

- `PreviewPayloadRequest`: owner, request id, fixture id, source path,
  artifact path or `.impress` path, and requested preview style.
- `PreviewPayload`: immutable or file-backed payload that can be applied by the
  shared preview controller on the UI thread.
- `PreviewPayloadBuilder`: non-UI worker module that loads source records and
  prepares payloads.
- `PreviewPayloadController`: Qt-side process or worker controller that owns
  request ids, cancellation, timeouts, stdout/stderr, and stale-result checks.

## Payload Rules

- Payloads must be safe to pass between process or thread boundaries.
- Payloads must not contain live Qt widgets, PyVistaQt interactors, VTK
  renderers, or open UI-owned handles.
- Payloads must include fixture id and request id metadata.
- Payloads must preserve enough information for the shared preview controller
  to apply the scene without reloading the source model on the UI thread.
- Payloads may be temporary files when direct immutable records would be too
  large.

## Worker Rules

- Worker modules may import Impression source-loading and tessellation code.
- Worker modules must not import workbench UI shell modules.
- Worker modules must not import PySide widget modules.
- Worker modules must not create or destroy renderers.
- Worker failures must return structured diagnostics instead of hanging the UI.

## UI Handoff Rules

- The Qt side allocates request ids before starting payload work.
- Selection change cancels or stale-marks older payload requests.
- Completion applies only if owner, kind, fixture id, and request id still
  match the current preview pane state.
- Errors from stale requests do not clear a newer successful preview.
- Current-request errors surface as preview-pane diagnostics.

## Required Code Changes

- Move preview build functions out of UI shell modules.
- Define `PreviewPayloadRequest` and `PreviewPayload` records.
- Add a payload builder module with no workbench UI imports.
- Replace `ProcessPoolExecutor` polling with a Qt-owned completion path or
  process controller.
- Add stale-result tests for rapid fixture switching.
- Add import-boundary tests for payload builder modules.

## Specification Manifest For Discovery

### Candidate Spec: Preview Payload Boundary And Builder

Discovery purpose:
- Define and implement the non-UI preview payload boundary between fixture
  source loading and the embedded preview widget.

Responsibilities:
- Functions/methods:
  - payload request factory
  - payload builder
  - payload result decoder
  - stale-result guard
- Data structures/models:
  - `PreviewPayloadRequest`
  - `PreviewPayload`
  - payload diagnostic record
- Dependencies/services:
  - fixture source contract
  - Impression source loading and tessellation
  - Qt-side async controller
- Returns/outputs/signals:
  - preview payload result
  - preview payload failure
  - cancellation diagnostic
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - loading and failure diagnostics
- Reusable code plan:
  - Existing code reused as-is: fixture source records and Impression loading
    code
  - Additions to existing reusable library/module: async controller preview
    route
  - New reusable library/module to create: preview payload builder module
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - worker-owned payload build, UI-thread handoff, cancellation, stale-result
    rejection
- Destructive/write behavior:
  - temporary payload files only, cleaned by controller lifecycle
- Security/privacy-sensitive behavior:
  - diagnostics redact unsafe environment details
- Performance-sensitive behavior:
  - no source loading or tessellation on the UI thread
- Cross-screen reusable behavior:
  - payload state feeds preview pane, artifact comparison, and promotion
    readiness

Project readiness fields:
- Implementation owner/module:
  - future `src/impression/devtools/reference_review/preview_payload.py`
  - future preview route in async controller
- Chosen defaults / parameters:
  - one active selected-fixture preview payload build at a time
- Test strategy:
  - import-boundary tests, rapid fixture-switch stale-result tests, payload
    failure tests, and manual fixture smoke
- Data ownership:
  - builder owns payload creation; preview pane owns current request identity;
    widget owns renderer mutation
- Routes:
  - selected fixture to payload controller to builder to preview pane to widget
- Open questions / nuance discovered:
  - exact file-backed payload format should be selected during specification
    refinement
- Readiness blockers:
  - shared preview controller must define the payload shape it can consume

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 40.5

Split decision:
- Split required before final specification. Split into payload records,
  payload builder, Qt-side controller/handoff, and stale-result verification
  leaves.

## Change History

- 2026-07-07: Created supplemental architecture for the preview payload
  boundary between async fixture loading and the embedded preview widget.
