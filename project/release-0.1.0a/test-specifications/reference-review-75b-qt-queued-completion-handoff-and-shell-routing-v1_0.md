# Reference Review Test Spec 75b: Qt Queued Completion Handoff And Shell Routing (v1.0)

## Paired Specification

- [Reference Review Spec 75b](../specifications/reference-review-75b-qt-queued-completion-handoff-and-shell-routing-v1_0.md)

## Automated Tests

- current successful completion enqueues payload command
- current failure enqueues failure command and disables display controls
- stale successful completion does not enqueue payload command
- stale failed completion does not clear current preview
- stale future exception does not clear current preview
- completion routing does not directly call renderer mutation methods

## Acceptance

- shell completion tests pass
- payload-controller tests pass
