# Loft Spec 35: Transition Operator Family Set (v1.0)

## Overview

This specification defines the explicit operator families available at the
next-generation loft planner / executor boundary.

## Backlink

Parent specification:

- [Loft Spec 24: Transition Operator and Planner / Executor Boundary (v1.0)](loft-24-transition-operator-and-planner-executor-boundary-v1_0.md)

## Scope

This specification covers:

- continuity operator family
- birth/death operator families
- split/merge operator families

## Behavior

The explicit executor-facing operator families are:

- `continuity`
- `split`
- `merge`

Current structural meaning:

- `continuity` covers stable one-to-one continuation
- `split` covers split-match and split-birth transition pairs
- `merge` covers merge-match and merge-death transition pairs

Current executor-facing support is exposed through:

- `PlannedRegionPair.operator_family`
- `PlannedTransition.executor_operator_families`

Unsupported operator families must remain explicit rather than inferred.

## Constraints

- operator families must be explicit
- executor-facing operator meaning must not depend on event guessing
- unsupported families must be explicit

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- operator families are explicit
- structural meaning per family is explicit
- executor-facing support expectations are explicit
