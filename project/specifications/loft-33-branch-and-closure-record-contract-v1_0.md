# Loft Spec 33: Branch and Closure Record Contract (v1.0)

## Overview

This specification defines branch-ordering and closure-ownership records in the
next-generation loft plan.

## Backlink

Parent specification:

- [Loft Spec 23: Evolution Plan and Plan Object Contract (v1.0)](loft-23-evolution-plan-and-plan-object-contract-v1_0.md)

## Scope

This specification covers:

- plan-local branch identifiers
- branch ordering
- loop/region closure records

## Behavior

Branch-local bookkeeping is carried by:

- `PlannedRegionPair.branch_id`
- `PlannedTransition.branch_order`

`branch_id` is plan-local and deterministic within one plan instance.

`branch_order` is the explicit execution order for emitted region pairs inside
one interval and must match region-pair emission order.

Closure ownership is carried by `PlannedClosure`:

- `side`
- `scope`
- optional `loop_index`

Loop closures use:

- `scope="loop"`
- `loop_index` set to the owned loop pair

Region closures use:

- `scope="region"`
- `loop_index=None`

The executor consumes closure ownership directly and must not infer closure from
missing geometry.

## Constraints

- branch bookkeeping must stay plan-local
- closure ownership must be explicit
- executor must not infer closure from missing geometry

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- branch record structure is explicit
- ordering rules are explicit
- closure ownership records are explicit
