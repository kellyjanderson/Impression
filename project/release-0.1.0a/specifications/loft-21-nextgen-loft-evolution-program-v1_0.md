# Loft Spec 21: Next-Gen Loft Evolution Program (v1.0)

## Overview

This specification defines the first-generation implementation branch for the
next-generation loft architecture.

It translates the stabilized loft architecture into major implementation
concerns that can be refined into executable leaves.

## Backlink

Parent architecture:

- [Loft Evolution System Architecture](../architecture/loft-evolution-system.md)

## Scope

This specification covers the next-generation loft branch at the program level:

- placed topology state inputs
- evolution-plan structure
- planner / executor boundary
- ambiguity and constraint-request handling
- tolerance and degeneracy policy
- many-to-many decomposition

## Behavior

This branch must define:

- the canonical loft input model
- the canonical planner output model
- the executor-facing operator contract
- the ambiguity and diagnostic surface
- the decomposition and escalation model for complex topology change

## Constraints

- this branch must stay surface-first
- this branch must preserve deterministic planning and execution
- this branch must not push surface-kernel seam law back into loft
- this branch must refine broad loft architecture into implementation-sized
  children before execution planning

## Refinement Status

Decomposed into final child leaves.

This parent branch does not yet represent executable work directly.

## Child Specifications

- [Loft Spec 22: Placed Topology State and Directional Correspondence Contract (v1.0)](loft-22-placed-topology-state-and-directional-correspondence-v1_0.md)
- [Loft Spec 23: Evolution Plan and Plan Object Contract (v1.0)](loft-23-evolution-plan-and-plan-object-contract-v1_0.md)
- [Loft Spec 24: Transition Operator and Planner / Executor Boundary (v1.0)](loft-24-transition-operator-and-planner-executor-boundary-v1_0.md)
- [Loft Spec 25: Ambiguity, Constraint Request, and Diagnostic Surface (v1.0)](loft-25-ambiguity-constraint-request-and-diagnostic-surface-v1_0.md)
- [Loft Spec 26: Tolerance and Degeneracy Policy (v1.0)](loft-26-tolerance-and-degeneracy-policy-v1_0.md)
- [Loft Spec 27: Many-to-Many Decomposition and Automatic Decomposability Gate (v1.0)](loft-27-many-to-many-decomposition-and-decomposability-gate-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- each major next-gen loft concern exists as its own child specification
- no child branch still mixes unrelated structural concerns
- the child set covers the stabilized loft architecture breadth-first
