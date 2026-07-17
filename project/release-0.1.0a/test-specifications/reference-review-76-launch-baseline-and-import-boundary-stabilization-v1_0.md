# Reference Review Test Spec 76: Launch Baseline And Import Boundary Stabilization (v1.0)

Status: Split parent test spec - superseded by test specs 76a and 76b

## Paired Specification

- [Reference Review Spec 76](../specifications/reference-review-76-launch-baseline-and-import-boundary-stabilization-v1_0.md)

## Architecture Ancestor

- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Manual Smoke Check

- Launch the app through the `.venv` entrypoint with the real fixture file used
  for stabilization.
- Confirm any failure is classified as import/dependency, QML/bootstrap,
  UI-thread blocking/handoff, preview renderer construction, or fixture/payload
  selection.
- Confirm no `impression_gui` dependency is required for launch.

## Automated Smoke Tests

- Import Reference Review UI modules in a clean process without selecting or
  rendering a fixture.
- Verify `impression_workbench` importability from the same `.venv` when kit
  helpers are used.
- Verify Reference Review imports do not import `impression_gui`.

## Automated Acceptance Tests

- Import-boundary test fails quickly and clearly when a required kit import is
  unavailable.
- Import-boundary test does not build models, tessellate geometry, or construct
  a preview scene.
- Launch baseline route records or exposes enough diagnostic information to
  classify a failure.

## Implementation Notes

- Use isolated subprocess import tests where module-cache leakage would hide an
  accidental `impression_gui` import.

## Acceptance

- paired automated tests pass
- manual or automated launch smoke is recorded
- `git diff --check` passes
