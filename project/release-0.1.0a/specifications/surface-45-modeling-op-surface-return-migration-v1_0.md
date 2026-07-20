# Surface Spec 45: Modeling Operation Surface Return-Type Migration (v1.0)

## Overview

This specification defines the migration of non-primitive modeling operations
to surface-returning behavior.

## Backlink

Parent specification:

- [Surface Spec 15: Modeling API Surface Return-Type Adoption (v1.0)](surface-15-modeling-api-surface-return-type-adoption-v1_0.md)

## Scope

This specification covers:

- modeling operation return-type migration
- operation families included in the first wave
- internal call-site expectations for migrated operations

## Behavior

The first public modeling-operation migration wave covers:

- `linear_extrude`
- `rotate_extrude`

These APIs now support an explicit `backend="surface"` path that returns
`SurfaceBody`.

The compatibility bridge remains explicit:

- `backend="mesh"` returns the legacy mesh path
- the default remains `backend="mesh"` during the compatibility phase
- surfaced outputs must flow through the standard surface tessellation boundary

No internal callers require immediate migration in this phase because the
surface return path is opt-in and the legacy mesh path remains explicit.

## Constraints

- migrated operations must not reintroduce hidden mesh execution
- the included operation set must be explicit
- internal callers must not need ad hoc adapters outside the compatibility plan

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
targets one bounded operation family with one call-site migration concern.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the operation set in scope is explicit
- their new return contract is explicit
- internal caller expectations are explicit
