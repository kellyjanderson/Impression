# Reference Review Test Spec 75b2: Preview Completion To Command Routing (v1.0)

## Paired Specification

- [Reference Review Spec 75b2](../specifications/reference-review-75b2-preview-completion-to-command-routing-v1_0.md)

## Automated Tests

- current successful completion enqueues payload command
- current failure enqueues failure command
- stale successful completion does not enqueue payload command
- stale failed completion does not enqueue failure command
- completion handlers do not directly call renderer mutation methods

## Acceptance

- shell completion-routing tests pass
- payload-controller tests pass
