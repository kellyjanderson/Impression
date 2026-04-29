# Testing Spec 10 Test: Reference Artifact Baseline Lifecycle and Invalidation Contract

## Overview

This test specification defines verification for shared dirty/clean baseline
lifecycle and invalidation behavior.

## Backlink

- [Testing Spec 10: Reference Artifact Baseline Lifecycle and Invalidation Contract (v1.0)](../specifications/testing-10-reference-artifact-baseline-lifecycle-and-invalidation-contract-v1_0.md)

## Automated Smoke Tests

- representative fixtures select clean references when present and dirty
  references otherwise
- lifecycle state remains inspectable through deterministic fixture paths

## Automated Acceptance Tests

- contract changes invalidate the previous dirty and clean baselines
- invalidated fixtures do not silently reuse stale references
- dirty references remain change detectors rather than truth claims
- clean promotion requires explicit review posture rather than automatic
  acceptance
