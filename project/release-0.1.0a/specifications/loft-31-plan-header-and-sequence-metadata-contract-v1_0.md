# Loft Spec 31: Plan Header and Sequence Metadata Contract (v1.0)

## Overview

This specification defines the header and sequence-wide metadata of the
next-generation loft plan object.

## Backlink

Parent specification:

- [Loft Spec 23: Evolution Plan and Plan Object Contract (v1.0)](loft-23-evolution-plan-and-plan-object-contract-v1_0.md)

## Scope

This specification covers:

- schema/version metadata
- plan-wide controls
- summary counts
- sequence-level planner metadata

## Behavior

The next-generation loft plan object is `LoftPlan`.

The explicit plan header is:

- `schema_version`
- `planner`
- `samples`
- `station_count`
- `interval_count`

The durable sequence-control metadata is exposed through `sequence_metadata`:

- `split_merge_mode`
- `split_merge_steps`
- `split_merge_bias`
- `ambiguity_mode`
- `ambiguity_cost_profile`
- `ambiguity_max_branches`
- `fairness_mode`
- `fairness_weight`
- `fairness_iterations`
- `skeleton_mode`

The required summary metadata is exposed through `summary_metadata` and
includes:

- ambiguity interval counts
- ambiguity class counts
- region topology/action counts
- fairness objective snapshots
- fairness diagnostics
- convergence status

## Constraints

- header fields must be explicit
- plan-wide controls must be auditable
- sequence metadata must remain geometry-free

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- plan header fields are explicit
- sequence-wide metadata is explicit
- summary metadata responsibilities are explicit
