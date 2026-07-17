# Reference Review Test Spec 78b: Non-Renderable Preview State And Last-Good Guard (v1.0)

Status: Final test leaf after review pass 1

## Paired Specification

- [Reference Review Spec 78b](../specifications/reference-review-78b-non-renderable-preview-state-and-last-good-guard-v1_0.md)

## Manual Smoke Check

- Select renderable, non-renderable, then renderable fixtures and confirm the
  app does not blank, crash, or lose future preview capability.

## Automated Smoke Tests

- Non-renderable fixture does not call renderer scene replacement.
- Stale success and stale failure cannot corrupt current preview.
- Same-fixture failure preserves last-good preview state.

## Acceptance

- non-renderable/stale preview tests pass
- manual selection smoke is recorded where needed
- `git diff --check` passes
