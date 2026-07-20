# Reference Review Test Spec 77a: Non-Blocking Shell Startup (v1.0)

Status: Final test leaf after review pass 1

## Paired Specification

- [Reference Review Spec 77a](../specifications/reference-review-77a-non-blocking-shell-startup-v1_0.md)

## Manual Smoke Check

- Launch the app and confirm the shell becomes responsive before fixture
  preview work begins.

## Automated Smoke Tests

- Shell bootstrap test proves fixture import, model build, tessellation,
  preview scene build, large fixture scan, and durable writes are not called
  during startup.

## Acceptance

- startup deferral tests pass
- `git diff --check` passes
