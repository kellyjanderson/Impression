# Loft Spec 36: Transition Operator Payload Contract (v1.0)

## Overview

This specification defines the minimum payload carried by executor-facing
transition operators.

## Backlink

Parent specification:

- [Loft Spec 24: Transition Operator and Planner / Executor Boundary (v1.0)](loft-24-transition-operator-and-planner-executor-boundary-v1_0.md)

## Scope

This specification covers:

- operator payload fields
- source/target structural references
- placement references
- ordering/composition references

## Behavior

The minimum executor-facing operator payload is exposed through
`PlannedRegionPair.operator_payload`.

Current payload fields are:

- `branch_id`
- `operator_family`
- `action`
- `prev_region_ref`
- `curr_region_ref`
- `loop_pairs`
- `closures`

Source/target structural references are represented by:

- `prev_region_ref`
- `curr_region_ref`
- embedded loop-level refs inside `loop_pairs`

Ordered multi-operator intervals are represented by:

- `PlannedTransition.branch_order`
- `PlannedTransition.executor_operator_payloads`

## Constraints

- payload shape must be explicit
- payload must be sufficient for deterministic execution
- payload must remain geometry-free at the plan layer

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- operator payload fields are explicit
- reference semantics are explicit
- ordering semantics are explicit
