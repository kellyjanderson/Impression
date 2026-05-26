# Surface Spec 55: Surface-Foundation to Loft-Track Handoff Gate (v1.0)

## Overview

This specification defines when the surface-foundation program is mature enough
for the dedicated loft surface refactor track to begin.

## Backlink

Parent specification:

- [Surface Spec 18: Surface Migration Sequencing and Subsystem Order (v1.0)](surface-18-surface-migration-sequencing-v1_0.md)

## Scope

This specification covers:

- prerequisites for starting the loft track
- what loft may assume from the surface foundation
- what remains out of scope for loft to invent ad hoc

## Behavior

The loft track may begin only after the following foundation contracts exist:

- explicit `SurfaceBody`, `SurfaceShell`, `SurfacePatch`, `SurfaceSeam`, and
  trim records
- seam-first tessellation with shared-boundary reuse
- open/closed shell classification driven by shell truth
- consumer handoff through standard surface collection/tessellation boundaries
- explicit public/internal boundary for still-private surfaced helpers

Required evidence before loft begins:

- passing shared-boundary and watertight tessellation tests
- passing scene/adapter/compatibility bridge tests
- surfaced public compatibility paths exist for at least one bounded primitive
  family and one bounded modeling-op family

Loft may not invent locally:

- seam ownership rules
- trim semantics
- shell closure rules
- watertightness policy
- hidden mesh repair fallback

## Constraints

- the handoff gate must be explicit
- loft must not become the place where missing surface-kernel semantics are guessed
- the handoff must preserve the program order set by the migration plan

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one cross-program handoff gate.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- loft prerequisites are explicit
- required evidence to start the loft track is explicit
- prohibited loft-side invention of kernel semantics is explicit
