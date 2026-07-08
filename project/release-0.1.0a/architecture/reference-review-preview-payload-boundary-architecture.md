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

## Manifest Review History

- 2026-07-07 loop 1: Initial review found the payload boundary candidate scored
  above the split threshold because it bundled records, builder, Qt handoff,
  cancellation, stale results, temporary files, and diagnostics.
- 2026-07-07 loop 2: Split immutable records from builder behavior so payload
  shape can stabilize before worker implementation.
- 2026-07-07 loop 3: Split Qt-side controller/handoff from worker payload
  building because the controller owns request ids, cancellation, and
  stdout/stderr.
- 2026-07-07 loop 4: Final review confirmed no remaining candidate scores at
  or above `25`; `16-24` candidates include cohesion explanations.

### Candidate Spec: Preview Payload Request And Result Records

Discovery purpose:
- Define the immutable records that describe a preview payload request,
  successful payload result, and payload failure diagnostic.

Responsibilities:
- Functions/methods:
  - payload request factory
  - payload result factory
- Data structures/models:
  - `PreviewPayloadRequest`
  - `PreviewPayload`
  - payload diagnostic record
- Dependencies/services:
  - fixture source contract
- Returns/outputs/signals:
  - preview payload result
  - preview payload failure
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: fixture source records
  - Additions to existing reusable library/module: preview payload record module
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - records carry owner, request id, fixture id, and generation metadata
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics redact unsafe environment details
- Performance-sensitive behavior:
  - records may reference file-backed payloads instead of embedding large data
- Cross-screen reusable behavior:
  - payload state feeds preview pane, artifact comparison, and promotion
    readiness

Project readiness fields:
- Implementation owner/module:
  - future `src/impression/devtools/reference_review/preview_payload.py`
- Chosen defaults / parameters:
  - records are immutable and include request identity
- Test strategy:
  - record construction, serialization, and diagnostic redaction tests
- Data ownership:
  - payload records own request/result data only
- Routes:
  - selected fixture to request record to worker/controller result
- Open questions / nuance discovered:
  - exact file-backed payload format should be selected during specification
    refinement
- Readiness blockers:
  - shared preview controller must define the payload shape it can consume

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:
- No split needed. Cohesion reason: request, result, and diagnostic records are
  one immutable payload contract; visible diagnostics are owned by pane state.

### Candidate Spec: Preview Payload Builder Orchestration

Discovery purpose:
- Orchestrate non-UI payload creation without importing Qt widgets or mutating
  live renderer state.

Responsibilities:
- Functions/methods:
  - payload builder
- Data structures/models:
  - `PreviewPayloadRequest`
  - `PreviewPayload`
- Dependencies/services:
  - source-load/tessellation adapter
  - payload serialization writer
- Returns/outputs/signals:
  - preview payload result
  - preview payload failure
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: fixture source records
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: preview payload builder module
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - runs outside UI thread and returns immutable/file-backed payloads
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - coordinates payload work without touching UI thread
- Cross-screen reusable behavior:
  - feeds workbench preview and later artifact comparison readiness

Project readiness fields:
- Implementation owner/module:
  - future preview payload builder module
- Chosen defaults / parameters:
  - builder imports no workbench UI shell or PySide widget modules
- Test strategy:
  - import-boundary tests and mocked adapter/writer orchestration tests
- Data ownership:
  - builder owns payload creation only
- Routes:
  - payload request to builder to payload result
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - payload request/result records

Score:
- Functions/methods: 1 x 2 = 2
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 18.5

Split decision:
- No split needed. Cohesive orchestration leaf; source loading and payload
  serialization are separate candidates.

### Candidate Spec: Preview Source Load And Tessellation Adapter

Discovery purpose:
- Load the selected fixture source and tessellate it for preview payload
  creation outside the UI thread.

Responsibilities:
- Functions/methods:
  - source load adapter
  - tessellation adapter
- Data structures/models:
  - loaded preview dataset
- Dependencies/services:
  - fixture source contract
  - Impression source loading and tessellation
- Returns/outputs/signals:
  - loaded preview datasets
  - source/tessellation diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: fixture source records and Impression loading
    code
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - runs outside the Qt UI thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics redact unsafe environment details
- Performance-sensitive behavior:
  - no source loading or tessellation on the UI thread
- Cross-screen reusable behavior:
  - loaded preview datasets feed payload serialization

Project readiness fields:
- Implementation owner/module:
  - future preview payload builder module
- Chosen defaults / parameters:
  - adapter imports no workbench UI shell or PySide widget modules
- Test strategy:
  - import-boundary and fixture load/tessellation tests
- Data ownership:
  - adapter owns loaded preview datasets until serialization
- Routes:
  - payload request to source/tessellation adapter
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - payload request records

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 21.5

Split decision:
- No split needed. Cohesive non-UI source-load/tessellation leaf.

### Candidate Spec: Preview Payload Serialization Writer

Discovery purpose:
- Convert loaded preview datasets into immutable or file-backed payloads that
  the UI-owned preview widget can consume.

Responsibilities:
- Functions/methods:
  - payload serializer
  - temporary payload file writer
- Data structures/models:
  - `PreviewPayload`
  - payload file metadata
- Dependencies/services:
  - preview payload records
- Returns/outputs/signals:
  - serialized payload
  - serialization diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: none
  - Additions to existing reusable library/module: preview payload module
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - writer runs outside UI thread
- Destructive/write behavior:
  - writes temporary payload files only
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - avoids passing live VTK or Qt objects across boundaries
- Cross-screen reusable behavior:
  - serialized payload feeds the embedded preview widget

Project readiness fields:
- Implementation owner/module:
  - future preview payload module
- Chosen defaults / parameters:
  - file-backed payloads are used when immutable in-memory payloads would be too
    large
- Test strategy:
  - serialization, temp file, and invalid payload tests
- Data ownership:
  - writer owns payload files until controller cleanup takes over
- Routes:
  - loaded preview datasets to serialized payload result
- Open questions / nuance discovered:
  - exact payload file format remains a specification default
- Readiness blockers:
  - preview payload records

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 0 x 0.5 = 0
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 22

Split decision:
- No split needed. Cohesive serialization/file-backed payload leaf.

### Candidate Spec: Preview Payload Process Controller

Discovery purpose:
- Add the Qt-side controller that starts payload work, tracks active request
  identity, and captures process diagnostics.

Responsibilities:
- Functions/methods:
  - preview payload request launcher
  - process diagnostic collector
- Data structures/models:
  - controller diagnostic record
- Dependencies/services:
  - Qt process or worker controller
  - preview payload records
- Returns/outputs/signals:
  - controller diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: async envelope pattern
  - Additions to existing reusable library/module: async controller preview route
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - owner/request id tracking and stdout/stderr capture
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics redact unsafe environment details
- Performance-sensitive behavior:
  - controller does not decode or tessellate payloads on UI thread
- Cross-screen reusable behavior:
  - controller events feed preview handoff route

Project readiness fields:
- Implementation owner/module:
  - future preview route in async controller
- Chosen defaults / parameters:
  - one active selected-fixture preview payload build at a time
- Test strategy:
  - success, failure, stdout/stderr, and process-state tests
- Data ownership:
  - controller owns active request identity and diagnostics
- Routes:
  - selection to controller to builder to pane handoff
- Open questions / nuance discovered:
  - exact worker technology remains behind controller policy
- Readiness blockers:
  - payload records

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- No split needed. Cohesive controller/process-supervision leaf; handoff and
  cleanup are separate candidates.

### Candidate Spec: Preview Current And Stale Payload Handoff

Discovery purpose:
- Decode current payload results and reject stale payload results before
  handing current payloads to the preview pane on the Qt UI thread.

Responsibilities:
- Functions/methods:
  - payload result decoder
  - owner/request match check
- Data structures/models:
  - active preview request state
- Dependencies/services:
  - preview payload records
  - preview pane
- Returns/outputs/signals:
  - payload ready event
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: async envelope pattern
  - Additions to existing reusable library/module: async controller preview route
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - UI-thread handoff and request id matching
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - no tessellation or source loading during handoff
- Cross-screen reusable behavior:
  - handoff state feeds review readiness

Project readiness fields:
- Implementation owner/module:
  - future preview route in async controller
- Chosen defaults / parameters:
  - fixture B selection wins over fixture A completion
- Test strategy:
  - current result and stale result handoff tests
- Data ownership:
  - preview pane owns current request identity after controller events
- Routes:
  - controller event to handoff route to preview pane
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - preview process controller

Score:
- Functions/methods: 2 x 2 = 4
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
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 20.5

Split decision:
- No split needed. Cohesive current/stale payload handoff leaf.

### Candidate Spec: Preview Payload Failure Diagnostic Handoff

Discovery purpose:
- Route current-request payload failures to preview-pane diagnostics without
  clearing newer successful previews.

Responsibilities:
- Functions/methods:
  - failure result decoder
  - diagnostic handoff
- Data structures/models:
  - payload diagnostic record
- Dependencies/services:
  - preview payload records
  - preview pane
- Returns/outputs/signals:
  - payload failed event
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - loading diagnostic
  - failure diagnostic
- Reusable code plan:
  - Existing code reused as-is: async envelope pattern
  - Additions to existing reusable library/module: async controller preview
    failure route
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - stale failures are ignored; current failures update UI on Qt thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics redact unsafe environment details
- Performance-sensitive behavior:
  - failure handoff does not touch renderer state
- Cross-screen reusable behavior:
  - failure state feeds review readiness

Project readiness fields:
- Implementation owner/module:
  - future preview route in async controller
- Chosen defaults / parameters:
  - stale failures never clear current previews
- Test strategy:
  - current failure, stale failure, and diagnostic redaction tests
- Data ownership:
  - preview pane owns visible diagnostic state
- Routes:
  - controller failure event to handoff route to preview pane
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - preview payload records

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:
- No split needed. Cohesive current-failure diagnostic handoff leaf.

### Candidate Spec: Preview Temporary Payload Cleanup

Discovery purpose:
- Clean file-backed payload artifacts for completed, cancelled, or stale
  preview requests.

Responsibilities:
- Functions/methods:
  - temporary payload cleanup
  - cancelled request cleanup
- Data structures/models:
  - payload file metadata
- Dependencies/services:
  - preview payload controller
- Returns/outputs/signals:
  - cleanup diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: none
  - Additions to existing reusable library/module: preview payload controller
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - cleanup must not race current payload ownership
- Destructive/write behavior:
  - deletes only controller-owned temporary payload files
- Security/privacy-sensitive behavior:
  - cleanup diagnostics avoid unsafe full environment details
- Performance-sensitive behavior:
  - cleanup is bounded and never blocks UI rendering
- Cross-screen reusable behavior:
  - protects repeated preview selection

Project readiness fields:
- Implementation owner/module:
  - future preview payload controller cleanup path
- Chosen defaults / parameters:
  - cleanup ignores files not owned by the active preview controller
- Test strategy:
  - completed, cancelled, stale, and missing-file cleanup tests
- Data ownership:
  - controller owns cleanup for its temporary payload files
- Routes:
  - controller lifecycle to cleanup path
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - payload file metadata

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 0 x 0.5 = 0
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 24

Split decision:
- No split needed. Cohesive cleanup leaf with one destructive boundary.

### Candidate Spec: Preview Stale Success And Failure Rejection Tests

Discovery purpose:
- Verify that stale payload successes and stale payload failures cannot mutate
  a newer selected fixture's preview state.

Responsibilities:
- Functions/methods:
  - stale-result test harness
- Data structures/models:
  - fixture switch scenario record
- Dependencies/services:
  - preview payload controller
  - preview pane
- Returns/outputs/signals:
  - stale-result test pass/fail
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - loading and failure diagnostics
- Reusable code plan:
  - Existing code reused as-is: Qt test harness
  - Additions to existing reusable library/module: preview async tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - validates request id rejection and stale error handling
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics redaction remains intact in failure paths
- Performance-sensitive behavior:
  - verifies stale results do not trigger extra scene application
- Cross-screen reusable behavior:
  - protects preview, artifact comparison, and promotion readiness

Project readiness fields:
- Implementation owner/module:
  - future preview async tests
- Chosen defaults / parameters:
  - fixture B selection wins over fixture A completion
- Test strategy:
  - controlled rapid fixture switch, stale success, and stale failure tests
- Data ownership:
  - tests own verification scenarios only
- Routes:
  - test selection changes to payload controller to pane state
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - preview payload UI handoff route

Score:
- Functions/methods: 1 x 2 = 2
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 21.5

Split decision:
- No split needed. Cohesive stale-result verification leaf.

### Candidate Spec: Preview Cancellation Ordering Tests

Discovery purpose:
- Verify that cancelled preview payload requests do not mutate preview state.

Responsibilities:
- Functions/methods:
  - cancellation scenario tests
- Data structures/models:
  - cancellation scenario record
- Dependencies/services:
  - preview payload controller
- Returns/outputs/signals:
  - cancellation test pass/fail
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - loading and failure diagnostics
- Reusable code plan:
  - Existing code reused as-is: Qt test harness
  - Additions to existing reusable library/module: preview async tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - validates cancellation ordering
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics redaction remains intact in cancellation paths
- Performance-sensitive behavior:
  - cancellation does not block UI rendering
- Cross-screen reusable behavior:
  - protects repeated preview selection

Project readiness fields:
- Implementation owner/module:
  - future preview async tests
- Chosen defaults / parameters:
  - cancellation is best-effort and stale-result guard remains authoritative
- Test strategy:
  - cancellation and stale completion after cancellation tests
- Data ownership:
  - tests own verification scenarios only
- Routes:
  - test cancellation to controller to preview pane state
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - preview payload controller

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:
- No split needed. Cohesive cancellation-ordering verification leaf.

### Candidate Spec: Preview Cleanup Deletion Tests

Discovery purpose:
- Verify temporary payload cleanup deletes only controller-owned files for
  completed, cancelled, and stale preview requests.

Responsibilities:
- Functions/methods:
  - cleanup verification tests
  - owned-file fixture builder
- Data structures/models:
  - cleanup scenario record
- Dependencies/services:
  - temporary payload cleanup
- Returns/outputs/signals:
  - cleanup test pass/fail
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: test harness
  - Additions to existing reusable library/module: preview async tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - validates cleanup ownership after cancellation/stale completion
- Destructive/write behavior:
  - verifies deletion of controller-owned temporary payload files
- Security/privacy-sensitive behavior:
  - cleanup diagnostics avoid unsafe environment details
- Performance-sensitive behavior:
  - cleanup tests verify bounded deletion scope
- Cross-screen reusable behavior:
  - protects repeated preview selection

Project readiness fields:
- Implementation owner/module:
  - future preview async tests
- Chosen defaults / parameters:
  - cleanup never deletes files outside controller-owned temp payload roots
- Test strategy:
  - completed, cancelled, stale, missing-file, and non-owned-file tests
- Data ownership:
  - tests own verification scenarios only
- Routes:
  - cleanup scenario to cleanup path
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - temporary payload cleanup

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:
- No split needed. Cohesive cleanup-deletion verification leaf.

## Change History

- 2026-07-07: Ran four manifest review loops, split the high-scoring payload
  boundary candidate, and recorded final split targets for oversized leaves.
- 2026-07-07: Created supplemental architecture for the preview payload
  boundary between async fixture loading and the embedded preview widget.
