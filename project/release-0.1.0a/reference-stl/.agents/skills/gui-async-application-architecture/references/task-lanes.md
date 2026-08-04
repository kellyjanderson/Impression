# Task Lanes

Use these lanes as a starting inventory for workbench-scale GUI apps. Add or remove lanes to match the app, but do not leave ownership implicit.

## UI Lane

- Owner: main UI thread or framework event loop.
- Allowed work: widget/model mutation, visible state changes, command dispatch, lightweight validation.
- Forbidden work: blocking IO, long model builds, subprocess waits, unbounded scans.
- Completion route: direct UI event handling or typed handoff from other lanes.
- Validation: UI event smoke or state inspection.

## Render Lane

- Owner: render thread, scene owner, or framework-specific render context.
- Allowed work: scene application, camera commands, render-resource mutation.
- Forbidden work: source loading, heavy geometry build, file scans.
- Completion route: typed render command accepted by render owner.
- Validation: render command smoke and stale command rejection.

## Preview/Build Lane

- Owner: preview controller, build supervisor, or worker process.
- Allowed work: source loading, model build, geometry/tessellation, preview artifact generation.
- Forbidden work: direct UI mutation and unbounded same-process active-code execution.
- Completion route: typed success/failure envelope with request id.
- Validation: stale success and stale failure rejection.

## File/Index Lane

- Owner: file watcher, indexer, or project context service.
- Allowed work: workspace scans, dependency parsing, file event coalescing.
- Forbidden work: UI mutation and unbounded queue growth.
- Completion route: typed index update or invalidation event.
- Validation: adjacent edits while preview/build runs.

## Agent/Subprocess Lane

- Owner: agent task manager or subprocess supervisor.
- Allowed work: LLM/tool subprocesses, background agent tasks, command execution.
- Forbidden work: direct UI mutation and unsupervised process lifetime.
- Completion route: typed lifecycle events and diagnostics.
- Validation: cancel, timeout, crash, and stale callback scenarios.

## Durable Write Lane

- Owner: save manager, patch applier, database writer, or proposal adoption service.
- Allowed work: saves, patches, proposal adoption, database/file writes.
- Forbidden work: hidden writes without conflict policy.
- Completion route: success/failure event with durable target and generation.
- Validation: conflict handling and failure recovery.

## Export/Snapshot Lane

- Owner: export manager or snapshot service.
- Allowed work: screenshots, exports, artifact generation.
- Forbidden work: blocking UI during long exports.
- Completion route: artifact-ready or failure event.
- Validation: export while source changes and stale artifact handling.

## Telemetry/Audit Lane

- Owner: telemetry or audit logger.
- Allowed work: diagnostics, structured events, audit records.
- Forbidden work: blocking user actions on non-critical logging.
- Completion route: best-effort or durable write policy.
- Validation: logging failure does not break primary UI behavior.
