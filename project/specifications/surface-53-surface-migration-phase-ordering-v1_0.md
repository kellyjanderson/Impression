# Surface Spec 53: Surface Migration Phase Ordering (v1.0)

## Overview

This specification defines the high-level ordering of migration phases for the
surface-first program.

## Backlink

Parent specification:

- [Surface Spec 18: Surface Migration Sequencing and Subsystem Order (v1.0)](surface-18-surface-migration-sequencing-v1_0.md)

## Scope

This specification covers:

- named migration phases
- strict ordering between phases
- prohibited out-of-order moves

## Behavior

The ordered migration phases are:

1. surface kernel foundation
2. tessellation and shell-truth boundary
3. scene/adapter adoption
4. public surfaced compatibility paths
5. loft surface track
6. canonical promotion and mesh-first retirement

Phase membership:

- phase 1: surface core data model and patch-family foundation
- phase 2: seam-first tessellation, watertightness, open/closed classification
- phase 3: scene payloads, consumer handoff, adapters, compatibility bridges
- phase 4: public primitive/modeling-op surfaced entry points with explicit mesh
  compatibility
- phase 5: loft surfaced executor, cap construction, patch orchestration,
  consumer handoff
- phase 6: promotion criteria, rollback posture, decommission policy

Prohibited out-of-order moves:

- loft track must not precede phases 1 through 4
- canonical promotion must not precede completion of the loft surface track
- mesh-first decommission must not precede canonical promotion

## Constraints

- the phase sequence must be explicit
- ordering must avoid circular dependency between foundation and adoption work
- loft-track work must remain downstream of surface-foundation phases

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one ordered migration program.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- phases are explicitly named
- their order is explicit
- prohibited out-of-order work is explicit
