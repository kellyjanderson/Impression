# Surface Spec 137: Surface Boolean Initial Reference Fixture Matrix and Promotion Gates (v1.0)

## Overview

This specification defines the first surfaced CSG named reference-fixture
matrix and the promotion gates required before the bounded executable slice is
treated as ready.

## Backlink

- [Surface Spec 120: Surface Boolean Initial Executable Scope and Reference Fixture Matrix (v1.0)](surface-120-surface-boolean-initial-executable-scope-and-reference-fixture-matrix-v1_0.md)

## Scope

This specification covers:

- the required reference images and STL files for the initial surfaced CSG slice
- the named regression fixture families required for union, difference, and intersection
- the missing-reference and promotion gates that control surfaced CSG promotion

## Behavior

This leaf must define:

- what named fixtures must exist before the initial surfaced CSG slice is considered ready
- what dirty-versus-clean reference artifacts are required for those fixtures
- what promotion and missing-reference failures keep the bounded surfaced lane from being treated as complete

## Constraints

- promotion evidence must use durable rendered and exported reference artifacts
- missing references must fail explicitly unless dirty-reference regeneration was requested intentionally
- fixture evidence must remain bounded to the initial executable scope rather than implying general surfaced CSG completion

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the required surfaced CSG reference-fixture matrix is explicit
- the named union, difference, and intersection fixtures are explicit
- promotion and missing-reference gates are explicit enough to block premature surfaced CSG promotion
- verification requirements are defined by its paired test specification
