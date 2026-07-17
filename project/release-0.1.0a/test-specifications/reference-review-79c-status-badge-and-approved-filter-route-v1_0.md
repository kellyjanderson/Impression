# Reference Review Test Spec 79c: Status Badge And Approved Filter Route (v1.0)

Status: Final test leaf after review pass 1

## Paired Specification

- [Reference Review Spec 79c](../specifications/reference-review-79c-status-badge-and-approved-filter-route-v1_0.md)

## Manual Smoke Check

- Toggle show-approved and confirm approved fixture visibility changes.
- Select approved, declined, and unreviewed fixtures and confirm badge state.

## Automated Smoke Tests

- Filter model hides approved by default and shows approved when checked.
- Badge state matches selected fixture status.

## Acceptance

- filter and badge tests pass
- `git diff --check` passes
