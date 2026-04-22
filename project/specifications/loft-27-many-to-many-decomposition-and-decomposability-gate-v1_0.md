# Loft Spec 27: Many-to-Many Decomposition and Automatic Decomposability Gate (v1.0)

## Overview

This specification defines how next-generation loft handles `N -> M` and
`M -> N` structure through deterministic decomposition and constraint-request
escalation.

## Backlink

Parent specification:

- [Loft Spec 21: Next-Gen Loft Evolution Program (v1.0)](loft-21-nextgen-loft-evolution-program-v1_0.md)

## Scope

This specification covers:

- many-to-many candidate-set isolation
- deterministic decomposition order
- automatic decomposability gate
- residual ambiguity escalation for unresolved regions

## Behavior

This branch must define:

- how unresolved many-to-many subsets are isolated
- the deterministic decomposition order within those subsets
- the automatic decomposability gate
- how residual unresolved regions become constraint requests

## Constraints

- decomposition must remain planner-owned
- executor must never receive unresolved many-to-many structure
- related subsets must use the same deterministic reduction rules as unrelated
  subsets

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Loft Spec 46: Many-to-Many Candidate-Set Isolation Rules (v1.0)](loft-46-many-to-many-candidate-set-isolation-rules-v1_0.md)
- [Loft Spec 47: Many-to-Many Deterministic Decomposition Order (v1.0)](loft-47-many-to-many-deterministic-decomposition-order-v1_0.md)
- [Loft Spec 48: Automatic Decomposability Gate Rules (v1.0)](loft-48-automatic-decomposability-gate-rules-v1_0.md)
- [Loft Spec 49: Residual Many-to-Many Constraint Escalation (v1.0)](loft-49-residual-many-to-many-constraint-escalation-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- candidate-set isolation is explicit
- decomposition order is explicit
- decomposability-gate rules are explicit
- residual escalation rules are explicit
