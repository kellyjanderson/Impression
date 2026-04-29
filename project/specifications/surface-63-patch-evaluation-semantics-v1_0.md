# Surface Spec 63: Patch Evaluation Semantics and Parameter Queries (v1.0)

## Overview

This specification defines how callers evaluate a patch at parameter-space
coordinates and what geometric queries are guaranteed.

## Backlink

Parent specification:

- [Surface Spec 21: Surface Patch Base Contract (v1.0)](surface-21-surface-patch-base-contract-v1_0.md)

## Scope

This specification covers:

- point evaluation
- tangent/normal query semantics
- parameter-space query guarantees

## Behavior

This branch must define:

- what evaluating a patch at `(u, v)` means
- which derivative or frame queries are guaranteed
- how evaluation behaves near seams and boundaries

## Constraints

- evaluation semantics must be deterministic
- boundary behavior must be explicit
- guaranteed queries must remain compatible across patch families

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- point evaluation semantics are explicit
- guaranteed geometric queries are explicit
- boundary/seam evaluation behavior is explicit

