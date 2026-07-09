# Feature 01 — B-Spline Curve Support Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document corresponds directly to Feature `1` in
[0.1.0.a Feature List](../planning/feature-list.md).

Related architecture:

- [B-Spline Implementation Architecture](b-spline-implementation-architecture.md)
- [Priority 01 — B-Spline Curve Constructs Architecture](priority-01-b-spline-curve-constructs-architecture.md)

## Purpose

Define the version-level product branch for first-class B-spline curve support.

This feature is the enabling foundation for the rest of the inference-and-curve-
fitting program.

## Included Scope

- `BSpline2D`
- `BSpline3D`
- explicit degree ownership
- explicit knot-vector ownership
- explicit closure or periodic policy
- deterministic evaluation
- deterministic tangent and derivative access
- deterministic sampling

## Excluded Scope

- fitting policy
- trajectory attachment semantics
- control-station inference results
- surfaced B-spline patch families

Those belong to later feature branches.

## Why It Exists

The current modeling surface has first-class:

- line
- arc
- bezier

But it does not yet have a durable compact smooth primitive suitable for:

- fit-backed curve explanation
- loft trajectory semantics
- richer progression ownership

## System Role

```text
authored or fitted curve truth
-> BSpline2D / BSpline3D
-> deterministic evaluation and sampling
-> downstream loft or inference consumer
```

## Expected Consumers

- fit policy and diagnostics
- dense-evidence curve fitting
- trajectory inference and guidance
- progression model upgrade

## Architectural Conclusion

Feature `01` is the first hard dependency of the `0.1.0.a` program. Without
first-class B-spline curves, the rest of the inference path stays shallow or
polyline-bound.
