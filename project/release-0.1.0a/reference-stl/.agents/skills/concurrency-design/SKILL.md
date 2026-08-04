---
name: concurrency-design
description: Use when making, reviewing, or debugging any concurrency decision or code path, including async, await, asyncio, threading, workers, queues, event loops, timers, callbacks, signals/slots, background jobs, parallel execution, locks, cancellation, stale results, shared mutable state, UI-thread handoff, database concurrency, process supervision, or multiple readers/writers. Apply during specs, architecture review, code review, implementation, and bug triage whenever concurrent behavior is present or implied.
---

# Concurrency Design

## Core Rule

Concurrency is ownership plus ordering. Before approving a design or marking code complete, identify who owns each task/result/state mutation, which thread/event loop may touch it, how stale work is discarded, and how failures surface.

Avoid treating “not asyncio” as “not concurrent.” Threads, Qt workers, callbacks, timers, signals, subprocesses, file watchers, database writers, UI event loops, schedulers, and background collectors are all concurrency.

## Trigger Synonyms

Use this skill when you see any of these words or concepts:

- async, await, asyncio, coroutine, future, promise;
- thread, worker, pool, queue, task, job, background, parallel;
- event loop, callback, signal, slot, observer, subscription, timer;
- lock, mutex, semaphore, atomic, race, deadlock, reentrancy;
- cancellation, timeout, retry, debounce, throttle, stale result;
- UI thread, main thread, worker thread, dispatch, handoff;
- subprocess, supervisor, daemon, service, collector;
- SQLite busy timeout, WAL, single writer, multiple readers.

## Required Pass

For every concurrent path, write or verify these answers:

1. **Ownership**: Which component owns the request, result, state, connection, file handle, process, or lock?
2. **Isolation**: Can unrelated tasks/results reach this consumer? If yes, how are they typed, routed, or ignored?
3. **Ordering**: What happens when request B finishes before request A?
4. **Staleness**: How are obsolete results detected and discarded?
5. **Thread affinity**: Which state must only mutate on the UI/main/event-loop thread?
6. **Resource lifetime**: Who keeps workers, signals, callbacks, and handles alive until completion? Who shuts them down?
7. **Shared state**: What mutable state crosses task/thread boundaries? Is it immutable, copied, locked, or single-owner?
8. **Cancellation/timeout**: What does cancellation mean? Is it best-effort, cooperative, or a hard stop?
9. **Backpressure**: What prevents unlimited tasks, refreshes, retries, queue growth, or DB work?
10. **Failure path**: Where do errors go? Can errors from task A clear or overwrite task B’s good state?
11. **Validation**: What integration test proves two concurrent/adjacent paths cannot interfere?

## Common Failure Patterns

- Global worker or event bus emits every result to every consumer.
- Result handlers infer payload type with weak shape checks like `hasattr(payload, "rows")`.
- A single request ID counter exists, but no per-owner/per-kind routing contract exists.
- Old results overwrite newer selections or paused views.
- UI state mutates directly from a worker thread.
- Worker/callback/signal objects are garbage-collected before completion.
- A timer keeps enqueueing refreshes while the prior refresh is still running.
- One SQLite connection is shared across threads, or a UI refresh uses a writer connection.
- A lock prevents corruption but creates deadlock or unbounded UI wait.
- Tests validate service and UI separately but never exercise the integrated async route.

## Preferred Patterns

### Typed Result Envelope

Prefer typed envelopes over shape checks:

```python
@dataclass(frozen=True)
class QueryEnvelope:
    owner: str
    kind: str
    request_id: int
    payload: object
    metadata: QueryResultMetadata
```

A consumer must check `owner` and `kind` before mutating state.

### Per-Owner Request Version

```python
request_id = next_id()
latest_by_owner[owner] = request_id

# On finish:
if latest_by_owner.get(owner) != request_id:
    discard_result()
else:
    apply_on_ui_thread(payload)
```

### UI Worker Rule

- Do blocking work off the UI thread.
- Create DB connections inside the worker/task that uses them.
- Emit typed results back to the UI thread.
- Mutate Qt/QML models only on the UI thread.
- Keep worker/signal objects alive until completion.

### Database Rule

- Prefer one connection per worker/task.
- Prefer read-only connections for UI reads.
- Keep writes short and explicit.
- Use one supported writer/collector manager; duplicate writers are a system-level concurrency bug.
- Surface lock/busy failures as recoverable UI status, not silent empty data.

## Spec and Review Checklist

When reviewing specs, architecture, or code that mentions concurrency:

- Add or verify a **Concurrency Contract** section.
- Include request/result ownership and routing.
- Include stale-result and cancellation semantics.
- Include thread-affinity rules.
- Include resource lifetime and cleanup.
- Include backpressure limits.
- Include at least one integration validation scenario that runs adjacent concurrent paths.
- When code contains a concurrency hazard that is too broad to fix in the
  current task, document it as a `codeimprovement` issue using the `coding`
  skill's Code Improvement Issues process, including line-number blocks for
  the affected producers, consumers, state owners, or resource lifetimes.

For deeper examples, read `references/concurrency-patterns.md`.

## SkillsKeeper Directives

<!-- skillskeeper-directive: gui-async-completion-rules -->
### GUI Async Completion Rules

## GUI Async Completion Rules

GUI apps are concurrent systems by default. The UI event loop, timers, signals, file watchers, renderers, subprocesses, background jobs, and user interactions are concurrent producers even before explicit worker threads are introduced.

A non-trivial GUI design is incomplete without task lanes or message queues, backpressure, cancellation or replacement behavior, stale-result rejection, failure routing, and explicit UI-thread handoff. Blocking the UI thread is suspect by default.

For workbench-scale apps, identify at least the relevant lanes before approving the design: UI, render, preview/build, file/index, agent/subprocess, durable write, export/snapshot, and telemetry/audit. Keep detailed lane inventories in `gui-async-application-architecture` references rather than duplicating them here.

When a GUI app executes user-authored, generated, plugin, model, or active-work code, discuss process isolation. Threads are not sufficient protection from `BaseException`, `SystemExit`, native crashes, memory blowups, infinite loops, import-cache poisoning, or host-process exit.

Stale failure handling must be symmetric with stale success handling: stale successes cannot overwrite current state, stale failures cannot clear current state, and stale cancellations cannot destroy current live state.
<!-- /skillskeeper-directive: gui-async-completion-rules -->

<!-- skillskeeper-directive: deferred-concurrency-code-improvements -->
### Deferred Concurrency Code Improvements

When implementation or review finds a concurrency defect that cannot be fixed within the current task, create or update a `codeimprovement` issue using the `coding` skill's Code Improvement Issues process. The issue must include `code-location` blocks for the relevant file and line ranges, name the ownership, ordering, stale-result, thread-affinity, lifetime, cancellation, backpressure, or failure-routing problem, and describe the validation needed to close it.
<!-- /skillskeeper-directive: deferred-concurrency-code-improvements -->
