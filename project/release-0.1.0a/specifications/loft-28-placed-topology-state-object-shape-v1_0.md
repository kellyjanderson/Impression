# Loft Spec 28: Placed Topology State Object Shape (v1.0)

## Overview

This specification defines the object shape of the canonical next-generation
loft input state.

## Backlink

Parent specification:

- [Loft Spec 22: Placed Topology State and Directional Correspondence Contract (v1.0)](loft-22-placed-topology-state-and-directional-correspondence-v1_0.md)

## Scope

This specification covers:

- progression field
- placement/frame payload
- normalized topology payload
- canonical per-sample object structure

## Behavior

The canonical placed-state input object is `Station`.

Required fields:

- `t`
- `origin`
- `u`
- `v`
- `n`
- `section`

Contract aliases:

- `progression -> t`
- `topology_state -> section`
- `placement_frame -> (origin, u, v, n)`

Structural versus placement truth:

- `section` is structural topology truth
- `t`, `origin`, `u`, `v`, and `n` are progression/placement truth

`t` is a monotonic scalar progression parameter.

`origin`, `u`, `v`, and `n` define the placed section frame. The planner
requires `u`, `v`, and `n` to form a valid right-handed orthonormal basis
before execution proceeds.

## Constraints

- the object shape must be explicit
- structure and placement must remain conceptually distinct
- the object must be suitable as the single canonical planner input state

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- placed-state fields are explicit
- structure versus placement boundary is explicit
- canonical input-state shape is explicit
- progression and placement-frame requirements are explicit
