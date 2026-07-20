# Reference Review Spec 75: Preview Render Command Queue (v1.0)

## Overview

Add a typed, bounded, Qt-aware preview render command queue so fixture loads,
payload completions, display-control toggles, and renderer lifecycle changes
are decoupled from direct renderer mutation.

This is an ad hoc remediation spec derived from the 2026-07-08 concurrency
review of the live Reference Review preview path.

## Backlink

- [Architecture: Reference Review Async Concurrency](../architecture/reference-review-async-concurrency.md)
- [Architecture: Reference Review Preview Qt Wrapper Architecture](../architecture/reference-review-preview-qt-wrapper-architecture.md)
- [Architecture: Reference Review Preview Payload Boundary Architecture](../architecture/reference-review-preview-payload-boundary-architecture.md)

## Source Manifest

- Ad hoc source: preview concurrency/threading review on 2026-07-08.
- Related implemented leaves:
  - [Reference Review Spec 05: UI Thread Handoff And Sanitized Task Errors](reference-review-05-ui-thread-handoff-and-sanitized-task-errors-v1_0.md)
  - [Reference Review Spec 51: Preview Payload Process Controller](reference-review-51-preview-payload-process-controller-v1_0.md)
  - [Reference Review Spec 55: Preview Widget Renderer Lifecycle](reference-review-55-preview-widget-renderer-lifecycle-v1_0.md)
  - [Reference Review Spec 56: Preview Widget Payload Application](reference-review-56-preview-widget-payload-application-v1_0.md)
  - [Reference Review Spec 70: Preview Display Command Routing](reference-review-70-preview-display-command-routing-v1_0.md)
- Manifest score: 75.5 before split

## Scope

This specification covers:

- a typed preview render command record for payload, display-options, clear,
  reset-camera, and failure/diagnostic state transitions
- a bounded/coalescing preview render command queue owned by the UI preview
  surface controller
- a Qt queued-signal handoff for worker completions and UI-originated preview
  commands
- replacement of direct renderer mutation from future polling, fixture
  selection, payload handoff, and display-control button handlers
- stale-result rejection for both successful and failed payload completions
  before any preview widget state is cleared or rendered

This specification does not replace the preview payload process worker or the
shared `impression.preview_qt.QtPreviewSurface`; it routes access to them.

## Responsibilities

- Functions/methods:
  - enqueue preview command
  - coalesce display-options and payload-apply commands
  - drain at most one latest render command per Qt event-loop tick
  - reject stale command identities before widget mutation
  - convert payload worker completions and failures into render commands
  - execute clear/reset/render commands only on the Qt UI thread
- Data structures/models:
  - `PreviewRenderCommand`
  - `PreviewRenderCommandKind`
  - `PreviewRenderCommandQueue`
  - `PreviewRenderQueueState`
  - `PreviewRenderCommandResult`
- Dependencies/services:
  - Qt `QObject` signal/slot queued delivery
  - existing `PreviewPayloadProcessController`
  - existing `PreviewDisplayOptions`
  - existing `PreviewRendererLifecycleWidget`
  - existing shared `QtPreviewSurface`
- Returns/outputs/signals:
  - queued command acceptance/rejection diagnostic
  - render command completion state
  - preview ready/failure visible state update
- UI surfaces/components:
  - Reference Review preview pane
  - preview display-control button row
- UI fields/elements:
  - preview viewport
  - preview unavailable/loading status text
  - preview display-control enabled state
- Reusable code plan:
  - Existing code reused as-is:
    - `ReviewWorkbenchMessage`
    - `WorkerResultEnvelope`
    - `PreviewPayloadProcessController`
    - `PreviewDisplayOptions`
    - `QtPreviewSurface`
  - Additions to existing reusable library/module:
    - add preview render command queue records to the Reference Review UI
      preview boundary
  - New reusable library/module to create:
    - `src/impression/devtools/reference_review/ui/preview_render_queue.py`
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - all preview renderer mutation occurs on the Qt UI thread
  - worker results and UI actions enqueue typed commands rather than calling
    renderer methods directly
  - queue is bounded and coalesces repeated commands so interaction cannot
    accumulate unbounded render work
  - stale commands cannot clear or overwrite the current preview
- Destructive/write behavior:
  - none; payload cleanup remains owned by `PreviewPayloadProcessController`
- Security/privacy-sensitive behavior:
  - diagnostics stay sanitized before visible UI handoff
- Performance-sensitive behavior:
  - display toggles produce at most one renderer scene application per event
    loop drain
  - payload deserialization must not be repeated when only display options
    change
  - queue draining must not render more often than necessary for the latest
    state
- Cross-screen reusable behavior:
  - none

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/preview_render_queue.py`
- `src/impression/devtools/reference_review/ui/shell.py`
- `src/impression/devtools/reference_review/ui/preview_widget.py`

Routes:

- fixture selection:
  - launch payload worker
  - enqueue loading/clear state
  - enqueue payload application only after current successful completion
- worker completion:
  - convert future result or exception into a typed command
  - reject stale identities before visible mutation
- display controls:
  - route button command to `PreviewDisplayOptions`
  - enqueue a display-options command
  - queue coalesces repeated display-options commands
- renderer:
  - preview widget/controller drains queue and mutates `QtPreviewSurface`
    only on the Qt UI thread

Reuse/extraction decision:

- Keep process-pool payload building as-is.
- Keep shared `impression.preview_qt` renderer surface as-is.
- Add a small Reference Review UI command queue rather than introducing
  Celery, RQ, asyncio queues, or a third-party message broker.
- Use Qt queued signals for cross-thread UI handoff when worker callbacks are
  introduced.

UI field/control inventory:

- preview viewport
- preview loading/unavailable text
- preview display-control row enabled/disabled state
- reset view command

## Data And Defaults

Chosen defaults / parameters:

- Queue capacity: one command per coalescing lane.
- Coalescing lanes:
  - `payload`: latest payload application for current fixture
  - `display`: latest display-options state
  - `lifecycle`: clear/loading/failure/reset commands
- Drain policy:
  - a `QTimer.singleShot(0, drain)` or equivalent queued Qt signal schedules
    drain after enqueue
  - one drain applies the latest coherent state, not every intermediate state
- Staleness identity:
  - owner
  - request id
  - fixture id
  - generation
- Failure handling:
  - stale failures are discarded
  - current failures enqueue a failure-state command
  - future exceptions without a matching current identity must not clear the
    current preview

Data ownership:

- payload process controller owns worker launch, active identity, payload
  cleanup, and stale payload decisions
- preview render command queue owns pending render commands
- preview widget owns decoded current datasets and renderer lifecycle
- renderer surface owns VTK/QtInteractor lifetime only

Open questions and resolved assumptions:

- Use stdlib `queue.Queue` only if cross-thread producers need a Python
  synchronized queue; otherwise a UI-thread-owned coalescing deque/dict is
  enough.
- Do not introduce external queue libraries for this leaf.
- Existing polling may remain temporarily if it only enqueues commands and
  stops mutating widgets directly.

Implementation prerequisites:

- existing preview payload process controller
- existing preview display options state and command routing
- existing shared Qt preview surface

## Concurrency Contract

Ownership:

- `PreviewPayloadProcessController` owns request identity and payload cleanup.
- `PreviewRenderCommandQueue` owns pending preview commands.
- `PreviewRendererLifecycleWidget` owns decoded datasets and render surface
  lifecycle.

Isolation:

- Commands must carry explicit kind and identity.
- A command without the current identity may update only neutral loading/empty
  state when explicitly requested by current selection logic.

Ordering:

- If request B is selected after request A, request A success or failure cannot
  clear, render, or enable controls for request B.
- Display commands after a payload command apply to the current decoded
  datasets without reloading payload JSON.

Staleness:

- Staleness is checked before status text, display controls, datasets, or
  renderer state are mutated.

Thread affinity:

- Qt widgets, `QtPreviewSurface`, and `QtInteractor` are touched only on the
  Qt UI thread.
- Worker completions cross into the UI thread through Qt queued delivery or
  are polled by the UI thread and then enqueued.

Resource lifetime:

- Queue/controller lifetime is tied to the preview widget/window.
- Pending commands are cleared on close before renderer disposal.

Shared state:

- Commands are frozen dataclasses.
- Dataset arrays are immutable-by-convention once stored on the widget.

Cancellation/timeout:

- Cancellation remains best-effort for running process-pool work.
- Completed stale work is discarded and cleaned up through the existing
  payload controller.

Backpressure:

- Queue capacity is bounded by coalescing lanes.
- Repeated display toggles replace the pending display command.
- Repeated fixture selections replace pending payload/lifecycle commands for
  older identities.

Failure path:

- Current failure commands show sanitized preview-unavailable text and disable
  display controls.
- Stale failure commands are recorded only as diagnostics if needed; they do
  not mutate visible preview state.

Validation:

- Tests must exercise rapid fixture selection, rapid display toggles, stale
  success, stale failure, and future exception paths through the integrated
  shell/widget route.

## Behavior

The implementation must:

- Introduce a typed preview render command queue with deterministic coalescing.
- Stop direct renderer mutation from:
  - `_poll_preview_payloads`
  - `_apply_preview_payload_ready`
  - `_apply_preview_payload_failed`
  - `_route_preview_display_command`
  - fixture selection/loading paths
- Preserve one long-lived render surface per preview widget.
- Preserve current payload cleanup behavior.
- Apply display-option changes without re-reading payload JSON and without
  resetting the camera.
- Apply at most one coherent renderer scene update per queue drain.
- Keep offscreen tests using fake/software render surfaces.

## Verification

Test strategy:

- focused unit tests for command queue coalescing and stale rejection
- shell integration tests for stale success/failure/exception behavior
- display-control integration tests proving multiple toggles coalesce into one
  renderer update
- renderer lifecycle tests proving the surface is not recreated by queued
  commands

Additional verification requirements:

- run:
  - `.venv/bin/python -m pytest tests/test_reference_review_async_core.py tests/test_reference_review_ui_shell.py tests/test_reference_review_preview_payload_controller.py -q`
  - `.venv/bin/python -m pytest tests/test_preview_controller.py -q`
- run `git diff --check`
- manually launch:
  - `.venv/bin/impression-reference-review --fixture-file tests/reference_review_fixtures/dirty-impress-fixtures.json`
- manual smoke:
  - rapidly select fixtures
  - rapidly toggle authored/inspection color, object edges, triangle wireframe,
    bounds grid, and axis triad
  - verify no beachball, stale failure, stale success, or old payload clears
    the current preview

## Manifest Assessment

Score:

- Functions/methods: 6 x 2 = 12
- Data structures/models: 4 x 1 = 4
- Dependencies/services: 5 x 1 = 5
- Returns/outputs/signals: 3 x 1 = 3
- Existing reusable code reused as-is: 5 x 0.5 = 2.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 3 x 2 = 6
- UI surfaces/components: 2 x 2 = 4
- UI fields/elements: 4 x 1 = 4
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 8 x 3 = 24
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total before split: 75.5

Split decision:

- Split required by score. For implementation, this ad hoc spec should be
  treated as a short remediation program with these final implementation
  slices:
  - 75a: Preview render command records and coalescing queue
  - 75b: Qt queued handoff and shell completion routing
  - 75c: Preview widget command drain and renderer mutation boundary
  - 75d: Integrated stale/rapid-interaction regression tests

## Refinement Status

Ad hoc remediation program spec. Split into child final leaves before
implementation unless the implementation pass is explicitly scoped to one
listed child slice.

## Child Specifications

- [Reference Review Spec 75a: Preview Render Command Records And Coalescing Queue](reference-review-75a-preview-render-command-records-and-coalescing-queue-v1_0.md)
- [Reference Review Spec 75a1: Preview Render Command Record Contract](reference-review-75a1-preview-render-command-record-contract-v1_0.md)
- [Reference Review Spec 75a2: Preview Render Coalescing Queue](reference-review-75a2-preview-render-coalescing-queue-v1_0.md)
- [Reference Review Spec 75b: Qt Queued Completion Handoff And Shell Routing](reference-review-75b-qt-queued-completion-handoff-and-shell-routing-v1_0.md)
- [Reference Review Spec 75b1: Preview Future Identity And Stale Exception Guard](reference-review-75b1-preview-future-identity-and-stale-exception-guard-v1_0.md)
- [Reference Review Spec 75b2: Preview Completion To Command Routing](reference-review-75b2-preview-completion-to-command-routing-v1_0.md)
- [Reference Review Spec 75c: Preview Widget Command Drain And Renderer Mutation Boundary](reference-review-75c-preview-widget-command-drain-and-renderer-mutation-boundary-v1_0.md)
- [Reference Review Spec 75c1: Preview Widget Queue Drain Scheduler](reference-review-75c1-preview-widget-queue-drain-scheduler-v1_0.md)
- [Reference Review Spec 75c2: Preview Command Application Efficiency](reference-review-75c2-preview-command-application-efficiency-v1_0.md)
- [Reference Review Spec 75d: Preview Render Queue Regression Tests](reference-review-75d-preview-render-queue-regression-tests-v1_0.md)

## Five-Pass Review Log

- Pass 1: Scope/readiness review. Confirmed Spec 75 is a remediation program,
  not a final implementation leaf.
- Pass 2: Score integrity review. Found Spec 75a undercounted the new reusable
  module and Specs 75b/75c were too broad for final leaves.
- Pass 3: Split review. Split command records from queue behavior, shell future
  exception guarding from completion routing, and widget drain scheduling from
  renderer application efficiency.
- Pass 4: Test-spec review. Added paired test specifications for every final
  leaf and kept broader parent test specs as rollups.
- Pass 5: Progression/readiness review. Updated progression to include only
  final leaves and paired test specs; parent specs remain durable context but
  are not implementation items.

## Acceptance

This specification is complete when:

- child final leaves exist or this spec is explicitly split during
  implementation intake
- the preview path is message-driven from worker completion and UI actions to
  renderer mutation
- stale completions and exceptions cannot clear or overwrite the current
  preview
- rapid fixture selection and rapid display toggles do not generate unbounded
  renderer work
