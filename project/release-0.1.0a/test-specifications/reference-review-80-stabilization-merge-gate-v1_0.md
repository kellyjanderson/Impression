# Reference Review Test Spec 80: Stabilization Merge Gate (v1.0)

Status: Final test leaf after review pass 2

## Paired Specification

- [Reference Review Spec 80](../specifications/reference-review-80-stabilization-merge-gate-v1_0.md)

## Architecture Ancestor

- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Manual Smoke Check

- Launch the real app through `.venv/bin/impression-reference-review` with real
  fixture data.
- Confirm fixture list, renderable preview, non-renderable fixture context,
  notes, approve/decline, status badge, and show-approved filtering behave as
  expected.
- Record the command and any notable environment assumptions in the PR.

## Automated Smoke Tests

- Run focused preview payload tests.
- Run focused UI shell launch/routing tests.
- Run focused notes/status/promotion tests.
- Run focused display-control state and command-routing tests.
- Run `git diff --check`.

## Automated Acceptance Tests

- The stabilization branch cannot be treated as merge-ready until Specs 76a,
  76b, 77a, 77b, 78a, 78b, 79a, 79b, and 79c have their focused tests and
  manual route evidence satisfied.
- Validation evidence distinguishes helper-level tests from the real
  console-launched GUI route.
- Full Workbench Kit migration work is not included as a hidden prerequisite
  for this merge gate.

## Implementation Notes

- This test spec is a merge gate, not product behavior. It should be satisfied
  by the validation transcript, PR checklist, and final focused test results.

## Acceptance

- focused test set passes
- real-entrypoint smoke is recorded
- branch is pushed and PR is created only after stabilization evidence exists
- `git diff --check` passes
