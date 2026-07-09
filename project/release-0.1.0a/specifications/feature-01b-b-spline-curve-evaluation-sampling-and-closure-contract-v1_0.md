# Feature Spec 01B: B-Spline Curve Evaluation, Sampling, and Closure Contract (v1.0)

## Overview

This specification defines the deterministic behavior contract for B-spline
curve evaluation and sampling.

## Backlink

- [Feature Spec 01: B-Spline Curve Support Program (v1.0)](feature-01-b-spline-curve-support-program-v1_0.md)

## Scope

This specification covers:

- point evaluation
- derivative and tangent evaluation
- deterministic sampling
- closure and periodic interpretation during evaluation

## Behavior

This leaf must define:

- how identical inputs produce identical evaluation and sampling results
- how closure affects parameter-domain behavior
- what minimum derivative access the primitive guarantees

## Constraints

- sampling must remain deterministic
- closure handling must not silently alter authored curve ownership
- consumer-boundary tessellation must remain explicit

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- deterministic evaluation behavior is explicit
- sampling and closure behavior are explicit
- derivative and tangent access are defined
