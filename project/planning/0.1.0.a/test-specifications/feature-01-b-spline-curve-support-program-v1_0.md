# Feature Spec 01 Test: B-Spline Curve Support Program

## Overview

This test specification defines verification for the decomposed `0.1.0.a`
B-spline curve support branch.

## Backlink

- [Feature Spec 01: B-Spline Curve Support Program (v1.0)](../specifications/feature-01-b-spline-curve-support-program-v1_0.md)

## Behavior

This parent test branch must verify:

- no executable B-spline primitive work remains hidden in the parent
- no executable evaluation or sampling work remains hidden in the parent
- every final child leaf has a paired test specification

## Child Test Specifications

- [Feature Spec 01A Test: B-Spline Curve Primitive Representation and Ownership](feature-01a-b-spline-curve-primitive-representation-and-ownership-v1_0.md)
- [Feature Spec 01B Test: B-Spline Curve Evaluation, Sampling, and Closure Contract](feature-01b-b-spline-curve-evaluation-sampling-and-closure-contract-v1_0.md)

## Acceptance

This test specification is complete when the child set covers the full branch
without leaving executable work hidden in the parent.
