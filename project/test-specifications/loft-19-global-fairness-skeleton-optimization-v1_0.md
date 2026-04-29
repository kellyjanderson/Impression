# Loft Spec 19 Test: Global Fairness and Skeleton Optimization

## Overview

This test specification defines verification for deterministic fairness and
skeleton-guided loft planning.

## Backlink

- [Loft Spec 19: Global Fairness and Skeleton Optimization (v1.0)](../specifications/loft-19-global-fairness-skeleton-optimization-v1_0.md)

## Manual Smoke Check

- Run a designated branching loft fixture with fairness off and with global
  fairness enabled.
- Confirm the fairness-enabled result is visibly smoother while remaining valid.
- Exercise `skeleton_mode="required"` on a fixture where skeleton guidance is
  unavailable and confirm explicit failure.

## Automated Smoke Tests

- fairness metadata is emitted in the loft plan
- same input and controls replay deterministically
- `skeleton_mode="required"` fails with the documented error when unavailable

## Automated Acceptance Tests

- designated fixtures show non-worse crossing/continuity metrics versus the
  baseline plan
- `skeleton_mode="auto"` falls back deterministically and remains valid
- fairness optimization never violates hard topology or watertightness gates
- convergence status and objective-term diagnostics are recorded in plan metadata

## Notes

- Use at least one real-world branching fixture, not only synthetic blobs.
