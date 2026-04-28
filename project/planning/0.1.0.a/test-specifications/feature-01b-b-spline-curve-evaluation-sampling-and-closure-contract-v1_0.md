# Feature Spec 01B Test: B-Spline Curve Evaluation, Sampling, and Closure Contract

## Overview

This test specification defines verification for deterministic B-spline curve
evaluation and sampling behavior.

## Backlink

- [Feature Spec 01B: B-Spline Curve Evaluation, Sampling, and Closure Contract (v1.0)](../specifications/feature-01b-b-spline-curve-evaluation-sampling-and-closure-contract-v1_0.md)

## Automated Smoke Tests

- evaluation returns stable points for identical inputs
- tangent or derivative access is available for supported parameter values

## Automated Acceptance Tests

- identical curves produce identical sampled point order
- closure or periodic behavior affects evaluation consistently across runs
- consumer-boundary sampling does not silently rewrite authored curve ownership
