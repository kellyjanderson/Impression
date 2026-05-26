# Surface Spec 138 Test: Surface Boolean Reference Fixture Composition and Slice Verification

## Overview

This test specification defines verification for surfaced CSG reference
fixtures that are meant to prove boolean correctness rather than only render and
export plumbing.

## Backlink

- [Surface Spec 138: Surface Boolean Reference Fixture Composition and Slice Verification (v1.0)](../specifications/surface-138-surface-boolean-reference-fixture-composition-and-slice-verification-v1_0.md)

## Manual Smoke Check

- render a representative surfaced CSG fixture and confirm the composition shows
  operand A, result, and operand B clearly enough that a human can tell whether
  the boolean outcome is plausible

## Automated Smoke Tests

- representative surfaced CSG fixtures can emit deterministic operand/result
  presentation artifacts without empty or degenerate output
- fixtures that declare canonical section checks can emit expected, actual, and
  diff slice artifacts in a shared local frame

## Automated Acceptance Tests

- overlap-evidence fixtures use non-trivial overlapping operands instead of
  disjoint, no-op, or exact-containment stand-ins unless they are explicitly
  named as no-cut baselines
- orientation-sensitive fixtures use an asymmetric cue, such as an outward
  notch, so a rotated result is distinguishable from the intended silhouette
- canonical section checks classify results as
  `same_shape_same_orientation`, `same_shape_rotated`, or `different_shape`
- each fixture declares whether `same_shape_rotated` is acceptable or a failure
- operand/result presentation and any section artifacts follow the documented
  dirty/clean reference lifecycle
