# Loft Spec 32: Planned State and Interval Record Contract (v1.0)

## Overview

This specification defines the planned-state and interval record structures in
the next-generation loft plan.

## Backlink

Parent specification:

- [Loft Spec 23: Evolution Plan and Plan Object Contract (v1.0)](loft-23-evolution-plan-and-plan-object-contract-v1_0.md)

## Scope

This specification covers:

- planned state records
- interval records
- state/interval references

## Behavior

The planned state record is `PlannedStation`.

Required fields:

- `station_index`
- `t`
- `origin`
- `u`
- `v`
- `n`
- `regions`

Contract aliases:

- `progression -> t`
- `placement_frame -> (origin, u, v, n)`
- `normalized_regions -> regions`

The interval record is `PlannedTransition`.

Required fields:

- `interval`
- `region_pairs`
- `branch_order`
- `topology_case`
- `ambiguity_class`

Contract aliases:

- `planned_state_indices -> interval`

Intervals refer to planned states by index pair only. References must be
deterministic and range-valid against the enclosing `LoftPlan.stations`.

## Constraints

- state and interval records must be explicit
- references must be deterministic and range-valid
- interval records must remain geometry-free

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- planned-state fields are explicit
- interval-record fields are explicit
- state/interval reference semantics are explicit
