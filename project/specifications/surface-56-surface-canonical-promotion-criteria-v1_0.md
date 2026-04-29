# Surface Spec 56: Surface Canonical Promotion Criteria (v1.0)

## Overview

This specification defines the conditions that must be met before surfaces are
treated as the canonical internal model representation of Impression.

## Backlink

Parent specification:

- [Surface Spec 19: Surface Promotion and Mesh-First Decommission Gate (v1.0)](surface-19-surface-promotion-and-mesh-decommission-v1_0.md)

## Scope

This specification covers:

- promotion prerequisites
- subsystems that must be surface-native before promotion
- explicit non-negotiable conditions for canonical status

## Behavior

Surfaces become canonical only when all of the following are true:

- surface kernel truth is the authoritative internal model for new modeling work
- tessellation is the standard preview/export/render boundary
- scene and consumer handoff paths operate on surfaces
- explicit public surfaced compatibility paths exist for the first-wave
  primitives and modeling ops
- the loft surface track is complete
- legacy mesh use is compatibility-only rather than canonical

Allowed remaining compatibility debt at promotion time:

- explicit `backend="mesh"` compatibility paths may remain
- surface-to-mesh adapters may remain for legacy consumers

Non-negotiable conditions:

- no hidden mesh-first execution on canonical surfaced paths
- no unresolved seam or watertightness ambiguity in the kernel boundary
- promotion must be backed by the verification matrix and evidence burden

## Constraints

- promotion criteria must be explicit and testable
- canonical promotion must not be based on informal confidence alone
- the criteria must align with the migration ordering plan

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one promotion gate.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- mandatory promotion prerequisites are explicit
- tolerated remaining compatibility debt is explicit
- non-negotiable promotion conditions are explicit
