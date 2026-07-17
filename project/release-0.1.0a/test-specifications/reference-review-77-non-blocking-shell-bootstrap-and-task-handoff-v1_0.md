# Reference Review Test Spec 77: Non-Blocking Shell Bootstrap And Task Handoff (v1.0)

Status: Split parent test spec - superseded by test specs 77a and 77b

## Paired Specification

- [Reference Review Spec 77](../specifications/reference-review-77-non-blocking-shell-bootstrap-and-task-handoff-v1_0.md)

## Architecture Ancestor

- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Manual Smoke Check

- Launch the app through `.venv/bin/impression-reference-review`.
- Confirm the shell becomes responsive before any selected fixture preview is
  built.
- Select a fixture and confirm fixture work starts after launch rather than
  during shell bootstrap.

## Automated Smoke Tests

- Shell bootstrap test confirms no fixture source import, model build,
  tessellation, preview scene build, or durable write is triggered during
  startup.
- Worker completion handoff test confirms UI-visible state changes happen only
  through the shell/UI route.

## Automated Acceptance Tests

- A stale worker completion cannot mutate the selected fixture, preview state,
  notes state, or status state.
- A shell startup failure produces a diagnostic or controlled failure instead
  of an unclassified hang in the test route.
- Deferred fixture refresh starts only after the shell/event-loop startup route
  is established.

## Implementation Notes

- Prefer doubles for expensive fixture/model work so tests can assert the work
  was not called during bootstrap.

## Acceptance

- paired automated tests pass
- manual shell responsiveness smoke is recorded where automation is impractical
- `git diff --check` passes
