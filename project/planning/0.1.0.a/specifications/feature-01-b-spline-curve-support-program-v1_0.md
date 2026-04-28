# Feature Spec 01: B-Spline Curve Support Program (v1.0)

## Overview

This specification defines the `0.1.0.a` feature branch for first-class
B-spline curve support.

## Backlink

- [Feature 01 — B-Spline Curve Support Architecture](../architecture/feature-01-b-spline-curve-support-architecture.md)

## Scope

This specification covers:

- first-class B-spline curve primitives
- durable representation ownership
- deterministic curve evaluation behavior

## Behavior

This branch must define:

- the leaf that owns B-spline primitive representation
- the leaf that owns evaluation, sampling, and closure behavior

## Constraints

- B-spline curve support must enter as a curve primitive first
- surfaced B-spline patch families are out of scope for this branch

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Feature Spec 01A: B-Spline Curve Primitive Representation and Ownership](feature-01a-b-spline-curve-primitive-representation-and-ownership-v1_0.md)
- [Feature Spec 01B: B-Spline Curve Evaluation, Sampling, and Closure Contract](feature-01b-b-spline-curve-evaluation-sampling-and-closure-contract-v1_0.md)

## Acceptance

This specification is complete when:

- B-spline curve support is broken into executable primitive and behavior
  leaves
- the parent remains a branch container rather than an implementation leaf
