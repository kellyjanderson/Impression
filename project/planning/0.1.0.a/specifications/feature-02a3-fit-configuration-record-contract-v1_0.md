# Feature Spec 02A3: Fit Configuration Record Contract (v1.0)

## Overview

This specification defines the durable fit configuration record that bundles the
relevant fitting-policy inputs for one fit-backed workflow execution.

## Backlink

- [Feature Spec 02A: Parameterization, Knot, and Fit Configuration Records (v1.0)](feature-02a-parameterization-knot-and-fit-configuration-records-v1_0.md)

## Scope

This specification covers:

- fit configuration record shape
- linkage to parameterization and knot policies
- replayable configuration identity

## Behavior

This leaf must define:

- what a fit configuration record contains
- how it references parameterization and knot policies
- how later branches refer back to the exact fit configuration used

## Constraints

- fit configuration must be durable and replayable
- configuration identity must stay stable enough for diagnostics and comparison

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- fit configuration record shape is explicit
- linkage to policy records is explicit
- replay and comparison posture are explicit
