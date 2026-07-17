# Reference Review Test Spec 79: Review Workflow Persistence Smoke (v1.0)

Status: Split parent test spec - superseded by test specs 79a, 79b, and 79c

## Paired Specification

- [Reference Review Spec 79](../specifications/reference-review-79-review-workflow-persistence-smoke-v1_0.md)

## Architecture Ancestor

- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Manual Smoke Check

- Select a fixture and edit notes; confirm notes persist after selecting away
  and returning.
- Approve a fixture using disposable or expected test fixture data; confirm
  status persists and dirty artifact paths move to matching gold paths.
- Decline a fixture; confirm status persists and dirty artifacts remain in
  place.
- Toggle show-approved and confirm approved fixture visibility changes.

## Automated Smoke Tests

- Fixture-store test verifies selected fixture notes load and save.
- Status persistence test verifies approved, declined, and unreviewed states.
- Filter model test verifies approved fixtures are hidden by default and shown
  when show-approved is checked.

## Automated Acceptance Tests

- Notes write completion for an older selected fixture cannot update a newer
  selected fixture's visible notes.
- Approve persists approved status and performs dirty-to-gold artifact movement
  with matching folder structure.
- Decline persists declined status without moving artifacts.
- Status badge view model matches selected fixture status.

## Implementation Notes

- Use temporary fixture copies for tests that move artifacts.
- Avoid mutating canonical fixture data during automated tests.

## Acceptance

- paired automated tests pass
- manual review workflow smoke is recorded where needed
- `git diff --check` passes
