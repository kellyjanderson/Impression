# Reference Review Test Spec 77b: UI Handoff And Stale Completion Guard (v1.0)

Status: Final test leaf after review pass 1

## Paired Specification

- [Reference Review Spec 77b](../specifications/reference-review-77b-ui-handoff-and-stale-completion-guard-v1_0.md)

## Automated Smoke Tests

- Worker completion reaches visible state only through the UI handoff route.
- Stale success, stale failure, and stale cancellation are rejected before UI
  mutation.

## Automated Acceptance Tests

- Stale completion cannot mutate selected fixture, preview, notes, status, or
  fixture-list state.

## Acceptance

- stale-result handoff tests pass
- `git diff --check` passes
