# Loft Spec 40: Invalid-Input Versus Underconstrained Taxonomy (v1.0)

## Overview

This specification defines how next-generation loft distinguishes invalid input
from underconstrained but otherwise valid planning input.

## Backlink

Parent specification:

- [Loft Spec 25: Ambiguity, Constraint Request, and Diagnostic Surface (v1.0)](loft-25-ambiguity-constraint-request-and-diagnostic-surface-v1_0.md)

## Scope

This specification covers:

- malformed input classes
- contradictory directional constraint classes
- underconstrained ambiguity classes

## Behavior

Current invalid-input classes include:

- malformed station/frame input
- malformed section topology
- correspondence supplied without topology
- directional correspondence arity mismatch against normalized region count

Current underconstrained input class includes:

- topology ambiguity that requires additional directional correspondence

Current surfaced output difference:

- invalid input continues to raise ordinary `ValueError`
- underconstrained planning raises `LoftPlanningBlockedError`, which carries a
  `LoftAmbiguityRecord` and `LoftConstraintRequest`

## Constraints

- invalid input must not be collapsed into ambiguity
- underconstrained input must not be mislabeled as malformed
- category boundaries must be deterministic

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- invalid-input classes are explicit
- underconstrained classes are explicit
- output-category differences are explicit
