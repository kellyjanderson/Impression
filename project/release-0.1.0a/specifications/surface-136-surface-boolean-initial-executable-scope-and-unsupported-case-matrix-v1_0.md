# Surface Spec 136: Surface Boolean Initial Executable Scope and Unsupported-Case Matrix (v1.0)

## Overview

This specification defines the intentionally bounded first executable surfaced
boolean scope and the explicit unsupported-case matrix around it.

## Backlink

- [Surface Spec 120: Surface Boolean Initial Executable Scope and Reference Fixture Matrix (v1.0)](surface-120-surface-boolean-initial-executable-scope-and-reference-fixture-matrix-v1_0.md)

## Scope

This specification covers:

- the initial supported surfaced boolean operations
- the initial supported operand families and trim complexity
- the initial unsupported-case matrix for surfaced execution

## Behavior

This leaf must define:

- what surfaced boolean cases are executable first
- what cases remain explicit surfaced unsupported results
- what family, shell, and trim restrictions bound the initial executable slice

## Constraints

- the initial executable slice must remain intentionally smaller than the full surfaced CSG problem
- unsupported cases must remain explicit rather than silently downgrading to mesh truth
- scope posture must stay aligned with the bounded execution leaves under this branch

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the initial executable surfaced boolean scope is explicit
- the initial unsupported-case matrix is explicit
- the initial operation, family, and trim boundaries are explicit enough to gate implementation and tests
- verification requirements are defined by its paired test specification
