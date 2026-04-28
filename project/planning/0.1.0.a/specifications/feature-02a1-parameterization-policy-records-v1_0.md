# Feature Spec 02A1: Parameterization Policy Records (v1.0)

## Overview

This specification defines the durable policy records that assign parameter
values to sampled evidence before fitting.

## Backlink

- [Feature Spec 02A: Parameterization, Knot, and Fit Configuration Records (v1.0)](feature-02a-parameterization-knot-and-fit-configuration-records-v1_0.md)

## Scope

This specification covers:

- parameter-assignment policy records
- parameter-domain normalization posture
- replayable parameterization inputs

## Behavior

This leaf must define:

- what parameterization choices are explicit in the initial fit stack
- how those choices are carried into fit-backed workflows
- how identical evidence plus identical policy reproduce the same parameter
  assignment

## Constraints

- parameterization policy must not remain hidden consumer behavior
- parameterization output must be durable enough for replay and diagnostics

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- parameterization policy record shape is explicit
- replay and determinism posture are explicit
- downstream fit branches can depend on the same contract
