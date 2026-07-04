# Feature 09 — Inference Diagnostics And Explainability Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document corresponds directly to Feature `9` in
[0.1.0.a Feature List](../planning/feature-list.md).

Related architecture:

- [Feature 02 — Explicit Fit Policy And Diagnostics Architecture](feature-02-explicit-fit-policy-and-diagnostics-architecture.md)
- [Feature 05 — Control-Station Inference Architecture](feature-05-control-station-inference-architecture.md)
- [Feature 06 — Curve-Intent Inference Architecture](feature-06-curve-intent-inference-architecture.md)
- [Feature 07 — Shared Trajectory Inference And Guidance Architecture](feature-07-shared-trajectory-inference-and-guidance-architecture.md)

## Purpose

Define the cross-cutting diagnostic layer that keeps the whole inference program
explainable rather than opaque.

## Included Scope

- retained vs dropped station explanation
- fit drift reporting
- structural preservation reporting
- refusal diagnostics
- inferred trajectory evidence bundles
- inferred curve-intent evidence bundles
- exact-vs-inferred provenance display

## Core Rule

Every inference feature in `0.1.0.a` should be able to answer:

- what did the system infer
- what did it keep
- what did it reject
- why did it do that
- how confident is the supporting evidence

## System Role

This branch is cross-cutting infrastructure for all inference features, not a
single narrow algorithm.

It should unify reporting from:

- fit policy
- control-station inference
- curve-intent inference
- shared trajectory inference
- progression upgrade provenance

## Output Contract

The architecture should prefer durable bundles and reports over scattered debug
strings or hidden metrics.

## Architectural Conclusion

Feature `09` is what makes the `0.1.0.a` program trustworthy. Without it, the
other inference features would behave more like opaque heuristics than like
owned product capabilities.
