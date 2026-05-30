# Priority 02 — Parameterization, Knot, And Fit Policy Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document covers Priority Read item `2` from
[Low-Level Construct Gap Report](../../../research/2026-04-25-low-level-construct-gap-report.md).

Related architecture:

- [Priority 01 — B-Spline Curve Constructs Architecture](priority-01-b-spline-curve-constructs-architecture.md)
- [B-Spline Control-Station Inference Architecture](b-spline-control-station-inference-architecture.md)
- [Priority 04 — Control-Station Inference Result Architecture](priority-04-control-station-inference-result-architecture.md)

## Purpose

This branch defines the policy objects and diagnostic records required for any
future fitting work that depends on parameter assignment, knot selection, and
residual reporting.

## Core Need

The research repeatedly points out that fitting cannot be trusted if these
choices stay implicit:

- how samples are parameterized
- how many knots are allowed
- where knots are placed
- how residuals are measured
- which tolerance gate determines acceptance

Impression should therefore own these decisions as explicit records rather than
burying them inside opaque helpers.

## Required Policy Families

The branch should define explicit policy objects for:

- parameter assignment
- knot-count selection
- knot-placement selection
- fitting objective selection
- residual measurement
- acceptance tolerance

These can begin as small structured records rather than a deep policy class
hierarchy, but they should still be durable and inspectable.

## Likely Core Records

The smallest useful architecture set is:

- `ParameterizationPolicy`
- `KnotCountPolicy`
- `KnotPlacementPolicy`
- `SplineFitConfiguration`
- `SplineFitResidualReport`
- `SplineFitAcceptanceReport`

Optional later additions:

- parameter-domain normalization policy
- fairness or smoothness weighting policy
- exact-vs-approximate result classification record

## Behavioral Rules

This branch should enforce:

1. fitting policy is explicit input
2. fitting residuals are explicit output
3. acceptance or refusal is explained by diagnostics
4. identical data plus identical policy must produce identical fit results
5. fitting policy stays separate from the authored B-spline primitive

## Consumer Relationship

These policy records are not a user-facing feature by themselves.

They are enabling infrastructure for:

- control-station inference
- curve-intent inference
- trajectory fitting
- future section-derived reconstruction

So the architecture should keep them generic enough to serve more than one
consumer.

## Scope Boundary

This branch should not define:

- final control-station result objects
- path attachment semantics
- surfaced patch families

It only defines the fit-policy and diagnostics layer that those later branches
can depend on.

## Delivery Guidance

Recommended implementation order:

1. parameterization policy
2. knot-count and knot-placement policy
3. fit configuration record
4. residual and acceptance reports
5. first consumer integration in inference tooling

## Architectural Conclusion

Priority `2` is where Impression turns fitting from hidden math into owned
project truth.

Without this branch, future B-spline-backed features would be difficult to
debug, compare, or trust.
