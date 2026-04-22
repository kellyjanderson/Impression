# Loft Spec 45: Plan-Validation Tolerance Rules (v1.0)

## Overview

This specification defines the tolerance rules used to validate next-generation
loft plans before execution.

## Backlink

Parent specification:

- [Loft Spec 26: Tolerance and Degeneracy Policy (v1.0)](loft-26-tolerance-and-degeneracy-policy-v1_0.md)

## Scope

This specification covers:

- reference-range validity
- minimum viable sampled structure
- closure consistency checks
- execution-control completeness checks

## Behavior

Current plan-validation checks include:

- `plan.samples >= 3`
- strictly increasing planned-station `t`
- finite planned-station frame values
- metadata completeness
- non-negative ambiguity summary counts
- non-negative fairness objective and diagnostic values
- valid topology-case / ambiguity-class / action enums
- valid branch ordering
- reference-range validity for region and loop refs
- closure ownership consistency

Current execution-blocking invalid states include:

- malformed summary metadata
- out-of-range refs
- invalid closure ownership
- invalid branch ordering
- missing required plan metadata

Plan-validation is separated from planner-entry validity by scope:

- planner-entry validity checks raw authored/planning input
- plan-validation checks the generated `LoftPlan` before execution

## Constraints

- plan validation must occur before execution
- plan-validation rules must be explicit
- executor must not need to rediscover invalid plan structure

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- plan-validation checks are explicit
- execution-blocking validation outcomes are explicit
- distinction from input-validity rules is explicit
