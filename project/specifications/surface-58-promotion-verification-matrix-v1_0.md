# Surface Spec 58: Promotion Verification Matrix and Evidence Burden (v1.0)

## Overview

This specification defines the evidence required to prove that the surface-first
program is ready for canonical promotion.

## Backlink

Parent specification:

- [Surface Spec 19: Surface Promotion and Mesh-First Decommission Gate (v1.0)](surface-19-surface-promotion-and-mesh-decommission-v1_0.md)

## Scope

This specification covers:

- verification categories required for promotion
- evidence ownership
- the minimum proof burden for promotion signoff

## Behavior

The promotion verification matrix categories are:

- kernel topology and seam truth
- tessellation and watertightness
- scene and consumer handoff
- public surfaced API compatibility paths
- loft surfaced execution
- legacy mesh compatibility and rollback posture
- documentation/spec/progression consistency

Required evidence per category:

- passing automated tests where applicable
- explicit progression status
- current specifications and test specifications
- absence of hidden mesh-first fallback on surfaced canonical paths

Evidence ownership:

- modeling kernel/tests own kernel, tessellation, and loft evidence
- public API tests own surfaced compatibility-path evidence
- project docs/specs own progression, promotion, and rollback evidence

## Constraints

- the proof burden must be explicit
- verification categories must map to promotion criteria rather than float independently
- evidence ownership must be unambiguous

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one verification matrix for one promotion gate.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- verification categories are explicit
- required evidence per category is explicit
- evidence ownership is explicit
