# Reference Review Test Spec 78: Preview Lifecycle And Non-Renderable Fixture Handling (v1.0)

Status: Split parent test spec - superseded by test specs 78a and 78b

## Paired Specification

- [Reference Review Spec 78](../specifications/reference-review-78-preview-lifecycle-and-non-renderable-fixture-handling-v1_0.md)

## Architecture Ancestor

- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Manual Smoke Check

- Select a renderable STL or `.impress` fixture and confirm it renders.
- Select a diagnostic or non-renderable fixture and confirm it shows contextual
  non-renderable state without crashing or blanking the app.
- Return to a renderable fixture and confirm the preview still renders and
  display-control buttons still affect the preview.

## Automated Smoke Tests

- Preview payload routing test confirms renderable artifact fixtures enter the
  render path.
- Preview routing test confirms diagnostic/non-renderable fixtures do not call
  renderer scene replacement.
- Display-control routing test confirms commands reach the preview surface
  through the UI route.

## Automated Acceptance Tests

- Preview widget renderer is not recreated for every fixture selection.
- Stale preview success cannot overwrite a newer selected fixture.
- Stale preview failure cannot clear a newer good preview.
- Same-fixture failure after a good render preserves last-good preview state and
  exposes a diagnostic/stale state.
- Renderer mutation happens only on the UI/render-thread route exposed to the
  test harness.

## Implementation Notes

- Use a lightweight preview widget double when real Qt/VTK integration is too
  heavy for automated acceptance.
- Keep one manual real-render smoke for the actual app route.

## Acceptance

- paired automated tests pass
- manual renderable/non-renderable selection smoke is recorded
- `git diff --check` passes
