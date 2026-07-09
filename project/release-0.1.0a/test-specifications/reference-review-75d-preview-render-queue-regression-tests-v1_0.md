# Reference Review Test Spec 75d: Preview Render Queue Regression Tests (v1.0)

## Paired Specification

- [Reference Review Spec 75d](../specifications/reference-review-75d-preview-render-queue-regression-tests-v1_0.md)

## Automated Tests

- integrated rapid fixture-selection stale success path
- integrated rapid fixture-selection stale failure path
- integrated rapid display-toggle coalescing path
- close-event pending-command cleanup path

## Manual Smoke

Run:

```bash
.venv/bin/impression-reference-review --fixture-file tests/reference_review_fixtures/dirty-impress-fixtures.json
```

Verify rapid fixture selection and rapid display toggles remain responsive.

## Acceptance

- automated integration tests pass
- manual smoke result is recorded in implementation notes
