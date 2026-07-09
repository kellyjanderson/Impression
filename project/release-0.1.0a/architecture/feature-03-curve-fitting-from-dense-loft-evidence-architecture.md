# Feature 03 — Curve Fitting From Dense Loft Evidence Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document corresponds directly to Feature `3` in
[0.1.0.a Feature List](../planning/feature-list.md).

Related architecture:

- [Feature 01 — B-Spline Curve Support Architecture](feature-01-b-spline-curve-support-architecture.md)
- [Feature 02 — Explicit Fit Policy And Diagnostics Architecture](feature-02-explicit-fit-policy-and-diagnostics-architecture.md)

## Purpose

Define how dense loft evidence becomes candidate smooth curve explanations.

## Included Scope

- dense-station fit candidate generation
- curve fitting over station-derived descriptors
- fit comparison against dense evidence
- candidate residual reporting
- refusal when the fit is not structurally trustworthy

## Core Rule

This feature is not automatic destructive simplification.

It is a structured analysis lane that asks:

- what smooth curve could explain this dense evidence
- where is that explanation good enough
- where does it drift too far to trust

## Evidence Sources

- station positions
- station spacing over distance
- correspondence-track movement
- loop centroid and scale evolution
- other later descriptor bands if they prove useful

## Output Shape

The smallest honest output is:

- one or more candidate fitted curves
- policy and residual reports
- structural confidence or refusal reasons

The fitted curve is supporting truth for later inference, not the final loft
result by itself.

## Architectural Conclusion

Feature `03` is the first place where the new B-spline and fit-policy work
becomes visible as a modeling improvement rather than just as infrastructure.
