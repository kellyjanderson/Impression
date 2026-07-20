# Feature Spec 02A: Parameterization, Knot, and Fit Configuration Records (v1.0)

## Overview

This specification defines the durable configuration-record branch required for
fit-backed inference.

## Backlink

- [Feature Spec 02: Explicit Fit Policy and Diagnostics Program (v1.0)](feature-02-explicit-fit-policy-and-diagnostics-program-v1_0.md)

## Scope

This specification covers:

- parameterization policy records
- knot policy records
- fit configuration records

## Behavior

This branch must define:

- the leaf that owns parameterization policy records
- the leaf that owns knot-count and knot-placement policy records
- the leaf that owns fit configuration record shape

## Constraints

- configuration must be durable and replayable
- policy choices must not be hidden in consumer code

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Feature Spec 02A1: Parameterization Policy Records](feature-02a1-parameterization-policy-records-v1_0.md)
- [Feature Spec 02A2: Knot-Count and Knot-Placement Policy Records](feature-02a2-knot-count-and-knot-placement-policy-records-v1_0.md)
- [Feature Spec 02A3: Fit Configuration Record Contract](feature-02a3-fit-configuration-record-contract-v1_0.md)

## Acceptance

This specification is complete when:

- parameterization, knot, and fit configuration work are split into honest
  executable leaves
- later inference consumers can refer to the same durable configuration shape
