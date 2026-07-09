# Reference Review Test Spec 75a: Preview Render Command Records And Coalescing Queue (v1.0)

## Paired Specification

- [Reference Review Spec 75a](../specifications/reference-review-75a-preview-render-command-records-and-coalescing-queue-v1_0.md)

## Automated Tests

- command kind validation
- immutable command record fields
- lane mapping for payload, display, lifecycle, camera, and failure commands
- latest command wins within each lane
- deterministic drain order
- clear pending commands empties queue state

## Acceptance

- queue unit tests pass
- `git diff --check` passes
