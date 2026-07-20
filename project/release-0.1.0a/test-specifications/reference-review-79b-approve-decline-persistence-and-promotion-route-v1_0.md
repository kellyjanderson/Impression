# Reference Review Test Spec 79b: Approve/Decline Persistence And Promotion Route (v1.0)

Status: Final test leaf after review pass 1

## Paired Specification

- [Reference Review Spec 79b](../specifications/reference-review-79b-approve-decline-persistence-and-promotion-route-v1_0.md)

## Manual Smoke Check

- Approve and decline disposable fixture copies and verify status/artifact side
  effects.

## Automated Smoke Tests

- Approve moves dirty artifacts to matching gold paths and persists approved.
- Decline persists declined and does not move artifacts.

## Acceptance

- approve/decline persistence tests pass
- `git diff --check` passes
