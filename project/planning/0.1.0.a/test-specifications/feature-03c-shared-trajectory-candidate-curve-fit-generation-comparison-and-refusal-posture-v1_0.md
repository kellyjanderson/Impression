# Feature Spec 03C Test: Shared-Trajectory Candidate Curve-Fit Generation, Comparison, and Refusal Posture

## Overview

This test specification defines verification for shared-trajectory candidate
curve-fit generation and comparison.

## Backlink

- [Feature Spec 03C: Shared-Trajectory Candidate Curve-Fit Generation, Comparison, and Refusal Posture (v1.0)](../specifications/feature-03c-shared-trajectory-candidate-curve-fit-generation-comparison-and-refusal-posture-v1_0.md)

## Automated Smoke Tests

- representative dense fixtures produce shared-trajectory candidate fits
- comparison returns either an accepted shared-trajectory candidate or an
  explicit refusal

## Automated Acceptance Tests

- shared-trajectory comparison references explicit residual diagnostics
- refusal remains explicit when no shared-trajectory candidate is trustworthy
- weak trajectory fits are not silently promoted
