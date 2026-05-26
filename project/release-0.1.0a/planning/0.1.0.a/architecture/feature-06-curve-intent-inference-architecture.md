# Feature 06 — Curve-Intent Inference Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document corresponds directly to Feature `6` in
[0.1.0.a Feature List](../feature-list.md).

Related architecture:

- [Feature 03 — Curve Fitting From Dense Loft Evidence Architecture](feature-03-curve-fitting-from-dense-loft-evidence-architecture.md)
- [Feature 05 — Control-Station Inference Architecture](feature-05-control-station-inference-architecture.md)

## Purpose

Define the feature branch that interprets dense loft evidence as likely smooth
curve behavior rather than as mere station noise or brute-force tessellation of
author intent.

## Included Scope

- station-density-over-distance analysis
- loop descriptor continuity analysis
- correspondence-track stability analysis
- smooth size, centroid, and anisotropy evolution signals
- curve-intent candidate reports

## Core Rule

The feature should infer intent from descriptor stability and rhythm, not just
from raw point spacing.

This is what distinguishes the branch from plain curve fitting:

- curve fitting explains geometry
- curve-intent inference explains likely author intent

## Output Contract

The feature should produce:

- candidate intent classifications
- candidate fitted curves where appropriate
- confidence or refusal reporting
- evidence records that later control-station or trajectory branches can reuse

## Architectural Conclusion

Feature `06` is the semantic bridge between dense station input and the higher-
order loft meaning that `0.1.0.a` is trying to recover.
