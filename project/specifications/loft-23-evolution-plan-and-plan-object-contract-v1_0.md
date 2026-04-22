# Loft Spec 23: Evolution Plan and Plan Object Contract (v1.0)

## Overview

This specification defines the canonical next-generation loft plan object and
its role as the planner / executor handoff.

## Backlink

Parent specification:

- [Loft Spec 21: Next-Gen Loft Evolution Program (v1.0)](loft-21-nextgen-loft-evolution-program-v1_0.md)

## Scope

This specification covers:

- plan header and sequence metadata
- planned states
- interval records
- branch ordering and closure records
- plan-layer diagnostic records

## Behavior

This branch must define:

- the structural sections of the plan object
- which records are executor-facing
- which records are diagnostic-only
- how execution eligibility is represented

## Constraints

- the plan must remain geometry-free
- the plan must be rich enough for deterministic execution
- the executor must not need to reinterpret topology from plan data

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Loft Spec 31: Plan Header and Sequence Metadata Contract (v1.0)](loft-31-plan-header-and-sequence-metadata-contract-v1_0.md)
- [Loft Spec 32: Planned State and Interval Record Contract (v1.0)](loft-32-planned-state-and-interval-record-contract-v1_0.md)
- [Loft Spec 33: Branch and Closure Record Contract (v1.0)](loft-33-branch-and-closure-record-contract-v1_0.md)
- [Loft Spec 34: Plan Diagnostics and Execution Eligibility Contract (v1.0)](loft-34-plan-diagnostics-and-execution-eligibility-contract-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- plan header is explicit
- state and interval records are explicit
- branch and closure records are explicit
- plan diagnostics and execution eligibility are explicit
