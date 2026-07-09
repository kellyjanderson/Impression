# Reference Review Test Spec 75a2: Preview Render Coalescing Queue (v1.0)

## Paired Specification

- [Reference Review Spec 75a2](../specifications/reference-review-75a2-preview-render-coalescing-queue-v1_0.md)

## Automated Tests

- lane mapping for payload, display, lifecycle, and camera commands
- latest command wins within each lane
- enqueue result reports accepted or replaced
- drain order is lifecycle, payload, display, camera
- clearing pending commands empties queue state

## Acceptance

- coalescing queue unit tests pass
- `git diff --check` passes
