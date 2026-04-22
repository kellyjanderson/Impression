# Loft Spec 37: Planner / Executor Execution-Boundary Rules (v1.0)

## Overview

This specification defines the exact validity conditions for crossing from
next-generation loft planning into execution.

## Backlink

Parent specification:

- [Loft Spec 24: Transition Operator and Planner / Executor Boundary (v1.0)](loft-24-transition-operator-and-planner-executor-boundary-v1_0.md)

## Scope

This specification covers:

- execution-boundary admissibility
- unresolved-state prohibition
- executor assumptions about resolved operators

## Behavior

Execution is allowed only for a validated `LoftPlan` that satisfies
`plan.require_executable()`.

Current admissibility rules:

- the plan must pass `_validate_loft_plan`
- the plan must not carry failed ambiguity intervals
- blocked planning states must surface before execution rather than inside the
  executor

Current executor assumptions:

- operator families and payloads are explicit on resolved region pairs
- branch ordering is explicit per interval
- closure ownership is explicit
- executor does not reinterpret topology or unresolved ambiguity

## Constraints

- executor must never receive unresolved ambiguity
- executor must not reinterpret topology
- admissibility rules must be binary and testable

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- execution-boundary conditions are explicit
- blocking states are explicit
- executor assumptions are explicit
