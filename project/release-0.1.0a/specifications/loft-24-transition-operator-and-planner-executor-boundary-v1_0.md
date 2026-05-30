# Loft Spec 24: Transition Operator and Planner / Executor Boundary (v1.0)

## Overview

This specification defines the operator contract that bridges next-generation
loft planning and execution.

## Backlink

Parent specification:

- [Loft Spec 21: Next-Gen Loft Evolution Program (v1.0)](loft-21-nextgen-loft-evolution-program-v1_0.md)

## Scope

This specification covers:

- operator families
- operator composition rules
- planner-owned versus executor-owned responsibilities
- execution-boundary validity conditions

## Behavior

This branch must define:

- the explicit operator types supported by next-gen loft
- what minimum payload each operator carries
- how multi-operator intervals are ordered
- what the executor may assume about resolved operators

## Constraints

- operators must stay explicit enough to prevent executor-side guessing
- operator composition must remain deterministic
- surface-kernel seam law must remain outside this branch

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Loft Spec 35: Transition Operator Family Set (v1.0)](loft-35-transition-operator-family-set-v1_0.md)
- [Loft Spec 36: Transition Operator Payload Contract (v1.0)](loft-36-transition-operator-payload-contract-v1_0.md)
- [Loft Spec 37: Planner / Executor Execution-Boundary Rules (v1.0)](loft-37-planner-executor-execution-boundary-rules-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- operator families are explicit
- operator payload shape is explicit
- execution-boundary rules are explicit
