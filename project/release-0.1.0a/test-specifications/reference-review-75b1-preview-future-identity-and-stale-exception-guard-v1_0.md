# Reference Review Test Spec 75b1: Preview Future Identity And Stale Exception Guard (v1.0)

## Paired Specification

- [Reference Review Spec 75b1](../specifications/reference-review-75b1-preview-future-identity-and-stale-exception-guard-v1_0.md)

## Automated Tests

- accepted preview future is tracked with request identity
- stale future exception does not call `clear_preview`
- untracked future exception does not mutate visible preview state
- current future exception enqueues a failure command

## Acceptance

- shell future-exception tests pass
