# Feature Spec 02 Test: Explicit Fit Policy and Diagnostics Program

## Overview

This test specification defines verification for the decomposed fit-policy and
diagnostics branch.

## Backlink

- [Feature Spec 02: Explicit Fit Policy and Diagnostics Program (v1.0)](../specifications/feature-02-explicit-fit-policy-and-diagnostics-program-v1_0.md)

## Behavior

This parent test branch must verify:

- no executable fit-policy ownership work remains hidden in the parent
- no executable diagnostics work remains hidden in the parent
- every final child leaf has a paired test specification

## Child Test Specifications

- [Feature Spec 02A Test: Parameterization, Knot, and Fit Configuration Records](feature-02a-parameterization-knot-and-fit-configuration-records-v1_0.md)
- [Feature Spec 02B Test: Fit Residual, Acceptance, and Refusal Reporting](feature-02b-fit-residual-acceptance-and-refusal-reporting-v1_0.md)

## Acceptance

This test specification is complete when the child set covers the full branch
without hidden executable work in the parent.
