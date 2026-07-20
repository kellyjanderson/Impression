# Feature Spec 01A Test: B-Spline Curve Primitive Representation and Ownership

## Overview

This test specification defines verification for first-class B-spline curve
primitive ownership.

## Backlink

- [Feature Spec 01A: B-Spline Curve Primitive Representation and Ownership (v1.0)](../specifications/feature-01a-b-spline-curve-primitive-representation-and-ownership-v1_0.md)

## Automated Smoke Tests

- `BSpline2D` and `BSpline3D` accept explicit control points, degree, and knot
  vectors
- closure or periodic policy remains inspectable from the primitive

## Automated Acceptance Tests

- authored knot vectors remain durable owned data
- primitive representation does not require fitting-policy inputs
- closure semantics are explicit rather than guessed from repeated endpoints
