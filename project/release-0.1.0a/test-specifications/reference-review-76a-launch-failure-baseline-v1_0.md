# Reference Review Test Spec 76a: Launch Failure Baseline (v1.0)

Status: Final test leaf after review pass 1

## Paired Specification

- [Reference Review Spec 76a](../specifications/reference-review-76a-launch-failure-baseline-v1_0.md)

## Manual Smoke Check

- Launch `.venv/bin/impression-reference-review` with the stabilization fixture
  file and record whether the shell starts or which failure class is observed.

## Automated Smoke Tests

- Controlled launch or shell probe returns a classified result.
- Probe does not import fixture sources, build models, tessellate, or construct
  preview scene content.

## Acceptance

- launch failure is classified or startup succeeds
- `git diff --check` passes
