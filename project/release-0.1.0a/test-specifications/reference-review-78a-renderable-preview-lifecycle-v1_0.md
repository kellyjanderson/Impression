# Reference Review Test Spec 78a: Renderable Preview Lifecycle (v1.0)

Status: Final test leaf after review pass 1

## Paired Specification

- [Reference Review Spec 78a](../specifications/reference-review-78a-renderable-preview-lifecycle-v1_0.md)

## Manual Smoke Check

- Select a renderable STL or `.impress` fixture and confirm it renders.
- Toggle display controls and confirm they affect the current preview.

## Automated Smoke Tests

- Renderable artifact fixture enters preview payload path.
- Preview widget renderer is not recreated for every fixture selection.
- Display-control command reaches current preview surface.

## Acceptance

- renderable preview lifecycle tests pass
- manual real-render smoke is recorded where needed
- `git diff --check` passes
