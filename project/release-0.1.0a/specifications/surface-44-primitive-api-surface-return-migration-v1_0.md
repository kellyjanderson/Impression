# Surface Spec 44: Primitive API Surface Return-Type Migration (v1.0)

## Overview

This specification defines the migration of primitive-creation APIs from
mesh-returning behavior to surface-returning behavior.

## Backlink

Parent specification:

- [Surface Spec 15: Modeling API Surface Return-Type Adoption (v1.0)](surface-15-modeling-api-surface-return-type-adoption-v1_0.md)

## Scope

This specification covers:

- primitive API return types
- compatibility expectations during primitive migration
- primitive-facing documentation updates

## Behavior

The first public primitive migration wave covers:

- `make_box`
- `make_cylinder`
- `make_cone`
- `make_ngon`
- `make_polyhedron`
- `make_nhedron`
- `make_prism`
- `make_sphere`
- `make_torus`

These APIs now support an explicit `backend="surface"` path that returns
`SurfaceBody`.

The compatibility bridge remains explicit:

- `backend="mesh"` returns the legacy mesh path
- the default remains `backend="mesh"` during the compatibility phase
- the surfaced path must flow through standard tessellation rather than hidden
  mesh emitters

## Constraints

- primitive APIs must not silently bifurcate into mesh and surface variants
- migration order must remain compatible with the broader surface rollout
- docs must match the new return contract

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
targets one bounded API family.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the primitive API set in scope is explicit
- their migrated return contract is explicit
- documentation/shim expectations are explicit
