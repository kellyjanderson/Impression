# Reference Review Async Concurrency

## Overview

This document defines how the Reference Review Workbench keeps the UI
responsive while doing expensive or blocking work.

The workbench loads Python model modules, builds surface bodies, tessellates for
preview, generates artifacts, scans filesystem roots, writes notes, promotes
artifacts, and may run Codex sidecar tasks. None of that work may block the Qt
Quick UI thread.

## Parent Architecture

- [Reference Review Workbench Architecture](reference-review-workbench-architecture.md)
- [Reference Review Qt Workbench UI](reference-review-qt-workbench-ui.md)
- [Reference Review Preview Payload Boundary Architecture](reference-review-preview-payload-boundary-architecture.md)
- [Reference Review Preview Qt Wrapper Architecture](reference-review-preview-qt-wrapper-architecture.md)

## Core Contract

- The UI thread owns QML state and Qt model mutation.
- Background workers own filesystem scans, source loading, model build,
  tessellation, artifact generation, note writes, promotion writes, and Codex
  tool execution.
- The embedded preview widget owns renderer mutation on the Qt UI thread; no
  worker may create, mutate, or destroy Qt widgets, PyVistaQt interactors, or
  live VTK renderer objects.
- Results cross boundaries only through typed immutable envelopes.
- Every request has owner, kind, request id, fixture id, cancellation, timeout,
  and stale-result semantics.
- Errors are routed to the pane/action that requested the work.

## Message Envelope

```python
@dataclass(frozen=True)
class ReviewWorkbenchMessage:
    owner: str
    kind: str
    request_id: int
    fixture_id: str | None
    payload: object | None = None
    error: ReviewWorkbenchError | None = None
```

## Worker Ownership

- Queue scan worker: discovers dirty/promoted/unresolved fixtures.
- Source load worker: imports or calls the source model entrypoint.
- Preview payload worker: prepares preview-ready payloads for the embedded
  actual-preview widget. It does not import Qt widget modules or mutate live
  renderer state.
- Artifact worker: regenerates PNG/STL/slice/diagnostic evidence.
- Notes worker: writes and reads durable review notes.
- Promotion worker: validates and writes gold artifacts atomically.
- Codex worker: executes allowlisted sidecar tool calls.

## Embedded Preview Concurrency Contract

The workbench preview is split into two ownership zones:

- Preview payload zone: source loading, `.impress` parsing, model construction,
  tessellation, and serialization into a render-ready payload. This work runs
  outside the Qt UI thread.
- Preview widget zone: renderer creation, scene replacement, camera mutation,
  interaction handling, and disposal. This work runs on the Qt UI thread inside
  the embedded `ImpressionPreviewWidget`.

Payload workers must return immutable or file-backed payload references. They
must not return live Qt, PyVistaQt, VTK renderer, or interactor objects.

The UI handoff sequence is:

```text
fixture selection
-> owner/request id allocated
-> preview payload worker launched
-> typed preview payload result
-> stale-result check
-> UI-thread handoff
-> ImpressionPreviewWidget.set_preview_payload(...)
```

Renderer lifetime rules:

- The widget creates one renderer for its preview surface.
- Fixture changes replace the scene inside that renderer.
- Cancellation or stale completion never destroys the renderer.
- Renderer disposal happens only when the preview widget closes or the
  workbench shuts down.
- Errors from a stale worker cannot clear a newer interactive preview.

## Ordering And Staleness

- Each visible owner keeps a latest request id.
- A completion can mutate UI state only when its owner and request id still
  match the latest visible request.
- Selection changes cancel or stale-mark source load, preview load, artifact
  load, and Codex context work for the prior fixture.
- Promotion completions are never silently discarded; if stale, they become a
  status diagnostic because they may have written durable files.

## Backpressure

- Only one selected-fixture source load runs at a time.
- Preview rebuild requests are coalesced by fixture id and source revision.
- Only one preview payload build for the selected fixture may be active; newer
  selections cancel or stale-mark older preview payload builds.
- Artifact regeneration is explicit and queued, not automatic on every
  keystroke.
- Codex tool calls are serialized per fixture.
- Queue scan refreshes are throttled while artifact generation is active.

## Failure Paths

Failure results must name:

- fixture id
- owner
- operation
- source path or artifact path when safe
- recovery action when known

Worker exceptions must never clear a newer successful preview or newer notes
state.

## Specification Manifest For Discovery

## Manifest Review History

- 2026-05-31 loop 1: Critical review found the original candidate mixed
  envelopes, dispatch, stale guards, durable writes, and UI handoff.
- 2026-05-31 loop 2: Rescored after moving visible loading/error components to
  the UI manifest.
- 2026-05-31 loop 3: Added cross-process write coordination as a separate leaf
  because promotion safety cannot rely on in-process task ordering.
- 2026-05-31 loop 4: Added structured audit logging as a separate leaf because
  it serves concurrency, sandbox, promotion, and review reports.
- 2026-05-31 loop 5: Final review confirmed no candidate remains at or above
  the split threshold.

### Candidate Spec: Review Workbench Message Envelope

Discovery purpose:
- Define the typed message/result envelope used by all background workbench
  tasks.

Responsibilities:
- Functions/methods:
  - request id allocator
  - envelope factory
- Data structures/models:
  - `ReviewWorkbenchMessage`
  - task kind enum
  - worker result envelope
- Dependencies/services:
  - PySide signal bridge
- Returns/outputs/signals:
  - task request envelope
  - task result envelope
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: ViewDown pattern as design precedent only
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: review workbench async core
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - all worker routes use the same envelope shape
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - envelope error text supports redacted display fields
- Performance-sensitive behavior:
  - lightweight immutable records
- Cross-screen reusable behavior:
  - shared by queue, preview, artifacts, notes, promotion, and Codex panes

Project readiness fields:
- Implementation owner/module:
  - future `src/impression/devtools/reference_review/async_core/messages`
- Chosen defaults / parameters:
  - request ids are per-owner monotonic values
- Test strategy:
  - envelope construction, serialization, and owner/request matching tests
- Data ownership:
  - async core owns task envelopes
- Routes:
  - caller to dispatcher to worker to UI owner
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Split decision:
- No split needed. This is a cohesive reusable envelope leaf.

### Candidate Spec: Task Dispatcher And Worker Policy

Discovery purpose:
- Route workbench tasks to bounded workers without blocking the Qt event loop.

Responsibilities:
- Functions/methods:
  - task dispatcher
  - worker pool selector
  - queue throttle
- Data structures/models:
  - dispatch request
  - worker policy record
- Dependencies/services:
  - Qt signal bridge
  - worker pool
- Returns/outputs/signals:
  - dispatch accepted
  - dispatch rejected diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: none
  - Additions to existing reusable library/module: async core
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - bounded queues, coalesced preview rebuilds, throttled scans
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - task payloads do not include unrestricted environment dumps
- Performance-sensitive behavior:
  - per-task queue limits and cancellation of superseded preview work
- Cross-screen reusable behavior:
  - dispatcher serves all workbench panels

Project readiness fields:
- Implementation owner/module:
  - future `async_core.dispatcher`
- Chosen defaults / parameters:
  - UI thread submits work; worker completions return through typed envelopes
- Test strategy:
  - bounded queue, coalescing, and failed dispatch tests
- Data ownership:
  - dispatcher owns task scheduling only
- Routes:
  - QML bridge to dispatcher to worker
- Open questions / nuance discovered:
  - exact worker technology remains behind policy interface
- Readiness blockers:
  - none

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 0 x 0.5 = 0
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
- Readiness blockers: 0 x 2 = 0
- Total: 23

Split decision:
- No split needed. Worker implementation details remain behind the policy
  interface.

### Candidate Spec: Stale Completion And Cancellation Guards

Discovery purpose:
- Prevent old background results from mutating newer UI or review state.

Responsibilities:
- Functions/methods:
  - latest request tracker
  - stale completion guard
  - cancellation marker
- Data structures/models:
  - request tracker
  - cancellation token
- Dependencies/services:
  - message envelope
  - dispatcher
- Returns/outputs/signals:
  - accepted completion
  - stale completion diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: ViewDown stale-result precedent
  - Additions to existing reusable library/module: async core
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - per-owner latest request guards and cancellation markers
- Destructive/write behavior:
  - stale destructive completions are rejected
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - constant-time request lookup
- Cross-screen reusable behavior:
  - shared by preview, notes, promotion, artifact generation, and Codex

Project readiness fields:
- Implementation owner/module:
  - future `async_core.staleness`
- Chosen defaults / parameters:
  - newer owner request supersedes older non-durable tasks
- Test strategy:
  - old preview completion after newer selection; stale note completion refusal
- Data ownership:
  - UI owners decide whether completions apply
- Routes:
  - completion envelope to owner guard to UI state
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
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
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Split decision:
- No split needed. Staleness and cancellation share the same request ownership
  contract.

### Candidate Spec: Durable Write Serialization And File Locking

Discovery purpose:
- Serialize note and promotion writes across threads and processes.

Responsibilities:
- Functions/methods:
  - durable write scheduler
  - file lock wrapper
- Data structures/models:
  - durable write request
  - lock acquisition result
- Dependencies/services:
  - filelock or equivalent
  - promotion/note services
- Returns/outputs/signals:
  - write accepted
  - lock conflict diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: reference artifact path helpers
  - Additions to existing reusable library/module: async durable write helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - serialized write lane with cross-process locks
- Destructive/write behavior:
  - controls note and gold artifact writes
- Security/privacy-sensitive behavior:
  - lock paths remain inside configured review roots
- Performance-sensitive behavior:
  - bounded lock wait and timeout
- Cross-screen reusable behavior:
  - shared by notes, promotion, candidate adoption, and release reports

Project readiness fields:
- Implementation owner/module:
  - future `async_core.durable_writes`
- Chosen defaults / parameters:
  - bounded lock wait; timeout returns a diagnostic instead of blocking UI
- Test strategy:
  - concurrent note write, concurrent promotion, lock timeout, stale write tests
- Data ownership:
  - durable write lane owns write ordering, not write semantics
- Routes:
  - service write request to durable lane to completion envelope
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
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
- Readiness blockers: 0 x 2 = 0
- Total: 24.5

Split decision:
- No split needed. Score is high but cohesive because file locking is only
  meaningful at the durable write boundary.

### Candidate Spec: UI Thread Handoff And Sanitized Task Errors

Discovery purpose:
- Ensure worker results reach QML-visible state only through UI-thread handoff
  with sanitized diagnostics.

Responsibilities:
- Functions/methods:
  - UI-thread completion bridge
  - task error sanitizer
- Data structures/models:
  - UI completion event
  - sanitized diagnostic
- Dependencies/services:
  - Qt signal bridge
  - message envelope
- Returns/outputs/signals:
  - UI completion signal
  - sanitized failure diagnostic
- UI surfaces/components:
  - none; panels render diagnostics separately
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: Qt signal/slot pattern
  - Additions to existing reusable library/module: async core UI adapter
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - no QML-visible state mutation outside UI thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - errors sanitize paths before UI and Codex exposure
- Performance-sensitive behavior:
  - minimal bridge overhead
- Cross-screen reusable behavior:
  - shared by all workbench panels

Project readiness fields:
- Implementation owner/module:
  - future `async_core.qt_handoff`
- Chosen defaults / parameters:
  - workers never mutate QML view models directly
- Test strategy:
  - UI-thread handoff and sanitized exception tests
- Data ownership:
  - UI bridge owns thread handoff only
- Routes:
  - worker completion to Qt signal to owner view model
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
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
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Split decision:
- No split needed. UI-thread handoff and error sanitization share one boundary:
  converting worker completions into safe owner-routed UI events.

### Candidate Spec: Structured Task Audit Events

Discovery purpose:
- Emit fixture-scoped structured events for task submission, refusal, failure,
  completion, cancellation, and stale-result rejection.

Responsibilities:
- Functions/methods:
  - audit event builder
  - task audit emitter
- Data structures/models:
  - audit event record
  - fixture-scoped log context
- Dependencies/services:
  - structlog or equivalent
  - message envelope
- Returns/outputs/signals:
  - structured audit event
  - audit emission diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: none
  - Additions to existing reusable library/module: async core audit hook
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - audit emission never blocks UI completion
- Destructive/write behavior:
  - writes non-authoritative local log events
- Security/privacy-sensitive behavior:
  - audit events omit secrets and full local environment
- Performance-sensitive behavior:
  - bounded event size
- Cross-screen reusable behavior:
  - audit events support task history, Codex refusals, and promotion reports

Project readiness fields:
- Implementation owner/module:
  - future `async_core.audit`
- Chosen defaults / parameters:
  - JSON-compatible structured events with `fixture_id`, `task_kind`, and
    `request_id`
- Test strategy:
  - event shape, redaction, and non-blocking emission tests
- Data ownership:
  - async audit owns task event records
- Routes:
  - task lifecycle to audit emitter to local log sink
- Open questions / nuance discovered:
  - log sink path belongs to dependency/local-state policy
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
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
- Readiness blockers: 0 x 2 = 0
- Total: 24

Split decision:
- No split needed. This remains below threshold and is cohesive around the
  structured task event boundary.

## Change History

- 2026-07-07: Added embedded actual-preview widget concurrency contract,
  separating preview payload workers from Qt-thread renderer ownership.
- 2026-05-31: Ran five critical review, rescore, and split passes over the
  specification manifest. Split the original async candidate into envelope,
  dispatcher, stale guard, durable write serialization, UI handoff, and audit
  leaves.
- 2026-05-30: Created concurrency architecture split using ViewDown's typed
  envelope, owner routing, stale guard, and UI-thread handoff lessons.
