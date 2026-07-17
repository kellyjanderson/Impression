# Reference Review Test Spec 76b: Import Boundary And Kit Availability (v1.0)

Status: Final test leaf after review pass 1

## Paired Specification

- [Reference Review Spec 76b](../specifications/reference-review-76b-import-boundary-and-kit-availability-v1_0.md)

## Automated Smoke Tests

- Clean-process Reference Review UI import succeeds or fails with controlled
  diagnostic.
- Import guard proves Reference Review does not import `impression_gui`.
- `impression_workbench` availability is asserted when kit helpers are used.

## Acceptance

- import-boundary tests pass
- `git diff --check` passes
