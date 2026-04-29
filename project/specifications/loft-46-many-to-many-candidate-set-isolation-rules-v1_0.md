# Loft Spec 46: Many-to-Many Candidate-Set Isolation Rules (v1.0)

## Overview

This specification defines how next-generation loft isolates unresolved
many-to-many structural subsets before decomposition.

## Backlink

Parent specification:

- [Loft Spec 27: Many-to-Many Decomposition and Automatic Decomposability Gate (v1.0)](loft-27-many-to-many-decomposition-and-decomposability-gate-v1_0.md)

## Scope

This specification covers:

- unresolved subset identification
- boundary between resolved and unresolved structure
- related-subset handling

## Behavior

This branch must define:

- how the candidate set is formed
- how already-resolved neighboring structure is excluded
- how related subsets are isolated without changing deterministic rules

## Constraints

- candidate-set formation must be deterministic
- isolation must remain planner-owned
- related subsets must not trigger a separate ambiguity regime

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- candidate-set formation rules are explicit
- resolved/unresolved boundary rules are explicit
- related-subset handling is explicit
- executable transitions expose deterministic candidate-set records
