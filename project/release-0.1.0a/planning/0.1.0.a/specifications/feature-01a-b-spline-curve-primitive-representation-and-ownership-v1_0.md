# Feature Spec 01A: B-Spline Curve Primitive Representation and Ownership (v1.0)

## Overview

This specification defines the first-class primitive representation for
`BSpline2D` and `BSpline3D`.

## Backlink

- [Feature Spec 01: B-Spline Curve Support Program (v1.0)](feature-01-b-spline-curve-support-program-v1_0.md)

## Scope

This specification covers:

- `BSpline2D`
- `BSpline3D`
- control-point ownership
- degree ownership
- knot-vector ownership
- closure or periodic policy ownership

## Behavior

This leaf must define:

- the minimal first-class fields required for authored B-spline curves
- how authored curves preserve explicit user or importer truth
- how closure is represented without hidden guessing

## Constraints

- primitive objects must not hide fitting policy inside their representation
- knot vectors must remain durable owned data
- closure must be explicit rather than inferred from repeated endpoints alone

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- first-class B-spline primitive ownership is explicit
- authored curve truth is durable and inspectable
- primitive representation is separated from fitting policy
