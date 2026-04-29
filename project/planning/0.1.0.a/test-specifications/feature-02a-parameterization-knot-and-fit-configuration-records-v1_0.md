# Feature Spec 02A Test: Parameterization, Knot, and Fit Configuration Records

## Overview

This test specification defines verification for the decomposed fit
configuration-record branch.

## Backlink

- [Feature Spec 02A: Parameterization, Knot, and Fit Configuration Records (v1.0)](../specifications/feature-02a-parameterization-knot-and-fit-configuration-records-v1_0.md)

## Behavior

This parent test branch must verify:

- no executable parameterization-policy work remains hidden in the parent
- no executable knot-policy work remains hidden in the parent
- no executable fit-configuration work remains hidden in the parent
- every final child leaf has a paired test specification

## Child Test Specifications

- [Feature Spec 02A1 Test: Parameterization Policy Records](feature-02a1-parameterization-policy-records-v1_0.md)
- [Feature Spec 02A2 Test: Knot-Count and Knot-Placement Policy Records](feature-02a2-knot-count-and-knot-placement-policy-records-v1_0.md)
- [Feature Spec 02A3 Test: Fit Configuration Record Contract](feature-02a3-fit-configuration-record-contract-v1_0.md)

## Acceptance

This test specification is complete when the child set covers all executable
work in the branch.
