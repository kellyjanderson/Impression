# Loft Spec 34: Plan Diagnostics and Execution Eligibility Contract (v1.0)

## Overview

This specification defines how diagnostics and execution eligibility are
represented in the next-generation loft plan.

## Backlink

Parent specification:

- [Loft Spec 23: Evolution Plan and Plan Object Contract (v1.0)](loft-23-evolution-plan-and-plan-object-contract-v1_0.md)

## Scope

This specification covers:

- plan-layer diagnostic embedding
- interval execution eligibility
- whole-plan blocking status

## Behavior

Embedded diagnostics remain planner-owned and geometry-free.

Interval-level status is exposed on `PlannedTransition`:

- `ambiguity_class`
- `execution_eligibility`
- `blocking_status`

Current eligibility contract:

- `ambiguity_class="none"` -> `execution_eligibility="executable"`
- any other supported ambiguity class -> `execution_eligibility="blocked_pending_constraint"`

Whole-plan status is exposed on `LoftPlan`:

- `is_executable`
- `blocking_status`

Current whole-plan blocking contract:

- all intervals executable -> `blocking_status="none"`
- any interval blocked -> `blocking_status="constraint_required"`

## Constraints

- diagnostics must not require executor reinterpretation
- execution eligibility must be explicit
- plan-layer blocking status must be deterministic

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- embedded diagnostics are explicit
- interval eligibility states are explicit
- blocking-status semantics are explicit
