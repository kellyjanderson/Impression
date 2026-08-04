---
name: gui-async-application-architecture
description: Use when designing, reviewing, or implementing GUI applications, desktop workbenches, Qt/PySide apps, Electron apps, web apps with live panels, renderers, file watchers, background tasks, subprocesses, or agent integrations. Requires non-blocking UI architecture, task lanes, message queues, UI-thread handoff, cancellation, stale-result handling, and integration validation before marking GUI work complete.
---

# GUI Async Application Architecture

Use this skill before designing, reviewing, or implementing GUI behavior that can trigger IO, rendering, model execution, subprocesses, file scans, indexing, network/database work, background jobs, or agent integrations.

## Non-Blocking UI Rule

Blocking is never allowed on the UI thread. A GUI app needs an explicit message/event model even when it appears simple, because the event loop, user actions, timers, signals, watchers, renderers, and workers are concurrent producers.

## UI Thread Ownership

UI state, widget models, scene mutation, and visible status updates must be owned by the UI/main/render thread appropriate to the framework. Background work returns results through a typed handoff; it does not mutate UI state directly.

## Task Lane Inventory

Every producer of work needs an owner, lane, queue policy, completion route, stale-result policy, and validation. For workbench apps, start from `references/task-lanes.md`.

## Message Envelope

Prefer typed result envelopes that include owner, kind, request id or generation, payload, diagnostic metadata, and completion state. Consumers must reject messages for the wrong owner, kind, or generation before mutating state.

## Queue And Backpressure Policy

Define what happens when requests arrive faster than they complete. Choose bounded queues, replacement, coalescing, debouncing, cancellation, or refusal deliberately.

## Cancellation And Timeout Policy

Every long-running task needs cancellation or replacement semantics. Define whether cancellation is cooperative, best-effort, or hard process termination. Define timeout behavior and what the user sees.

## Stale-Result Policy

Reject stale successes, stale failures, and stale cancellations before UI mutation. A stale success cannot overwrite newer state, a stale failure cannot clear newer good state, and a stale cancellation cannot destroy the current live view.

## Failure And Diagnostic Routing

Errors, crashes, timeouts, and invalid outputs need a visible diagnostic route. Background failure must not silently become empty UI state unless empty is the correct user-facing state and the diagnostic is still available.

## Shutdown Cleanup

Define who owns worker lifetime and how workers, subprocesses, timers, file watchers, queues, callbacks, and handles stop when the app closes or the owning panel disappears.

## Integration Validation

Validate through the real GUI route whenever practical. Use Qt signal/slot integration, widget event smoke, offscreen launch plus state inspection, subprocess crash/timeout smoke, or manual smoke when automation is not practical.

Read the reference files for lane details, active-work code isolation, and integration validation examples.

## Deferred GUI Code Improvements

When GUI, task-lane, message-routing, or UI-thread cleanup is too broad to fix
inside the current task, document it as a `codeimprovement` issue using the
`coding` skill's Code Improvement Issues process. Include `code-location`
blocks for the visible route, producer/consumer, worker, handoff, stale-result
check, or state mutation that shows the problem.

## SkillsKeeper Directives

<!-- skillskeeper-directive: gui-branch-of-the-integration-contract -->
### GUI Branch Of The Integration Contract

## GUI Branch Of The Integration Contract

This skill covers the GUI branch of the application integration contract. Use it for visible GUI routes, UI event flows, live panels, renderers, file watchers, background jobs, subprocesses, and agent integrations.

Do not impose GUI-only proof on console or API/service routes. Console apps need command/subcommand, args/stdin/config, stdout/stderr/exit-code, side-effect, and CLI validation proof. API/service apps need endpoint/caller contract, auth/error behavior, side-effect, observability, and route-level validation proof.

For mixed apps, pair this GUI guidance with the relevant console or API/service proof for each independently failing route. The GUI part is complete only when the visible route, UI state, async handoff, stale-result behavior, and GUI validation are all accounted for.
<!-- /skillskeeper-directive: gui-branch-of-the-integration-contract -->

<!-- skillskeeper-directive: document-deferred-gui-code-improvements -->
### Document Deferred GUI Code Improvements

If GUI architecture review finds blocking UI work, missing task lanes, weak message routing, unsafe UI-thread mutation, stale-result bugs, or missing integrated validation that cannot be fixed in the current task, create or update a `codeimprovement` issue using the `coding` skill's Code Improvement Issues process.
<!-- /skillskeeper-directive: document-deferred-gui-code-improvements -->
