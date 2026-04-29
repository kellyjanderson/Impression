# Feature Spec 02: Explicit Fit Policy and Diagnostics Program (v1.0)

## Overview

This specification defines the `0.1.0.a` branch for owned fitting policy and
diagnostic reporting.

## Backlink

- [Feature 02 — Explicit Fit Policy And Diagnostics Architecture](../architecture/feature-02-explicit-fit-policy-and-diagnostics-architecture.md)

## Scope

This specification covers:

- fit-policy records
- fit configuration ownership
- residual and acceptance reporting

## Behavior

This branch must define:

- the leaf that owns parameterization and knot policy records
- the leaf that owns residual and acceptance diagnostics

## Constraints

- fitting behavior must not remain hidden helper logic
- later inference branches must be able to consume durable diagnostics

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Feature Spec 02A: Parameterization, Knot, and Fit Configuration Records](feature-02a-parameterization-knot-and-fit-configuration-records-v1_0.md)
- [Feature Spec 02B: Fit Residual, Acceptance, and Refusal Reporting](feature-02b-fit-residual-acceptance-and-refusal-reporting-v1_0.md)

## Acceptance

This specification is complete when:

- fit policy ownership is explicit
- residual and acceptance diagnostics are pushed into final leaves
