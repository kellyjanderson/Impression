# Feature Spec 03B Test: Station-Derived Candidate Curve-Fit Generation, Comparison, and Refusal Posture

## Overview

This test specification defines verification for station-derived candidate
curve-fit generation and comparison.

## Backlink

- [Feature Spec 03B: Station-Derived Candidate Curve-Fit Generation, Comparison, and Refusal Posture (v1.0)](../specifications/feature-03b-station-derived-candidate-curve-fit-generation-comparison-and-refusal-posture-v1_0.md)

## Automated Smoke Tests

- representative dense fixtures produce station-derived candidate fits
- comparison returns either an accepted candidate or an explicit refusal

## Automated Acceptance Tests

- candidate comparison references explicit residual diagnostics
- refusal remains explicit when no station-derived candidate is trustworthy
- weak candidates are not silently selected
