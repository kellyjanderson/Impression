# Surface Spec 38: Shared-Boundary Sampling and Edge Agreement Rules (v1.0)

## Overview

This specification defines how tessellation produces identical samples along
shared patch boundaries so adjacent patches agree exactly at their seams.

## Backlink

Parent specification:

- [Surface Spec 13: Seam-Consistent Tessellation and Watertight Output Rules (v1.0)](surface-13-seam-consistent-tessellation-watertightness-v1_0.md)

## Scope

This specification covers:

- seam-edge sampling policy
- shared-boundary vertex agreement
- deterministic ownership of shared sampled edges

## Behavior

This branch must define:

- shared-boundary sampling is seam-first rather than patch-first
- the tessellator computes one canonical sample set from seam-owned 3D boundary
  truth under the active tessellation request
- each participating boundary use remaps that seam sample set into patch-local
  parameter space as needed
- patch tessellation reuses the seam sample set rather than sampling the same
  shared edge independently

## Constraints

- shared-edge sampling must be deterministic
- adjacent patches must not sample shared seams independently in incompatible ways
- edge agreement must hold without best-effort post-stitching

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one seam-sampling contract for tessellation.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- seam sampling rules are explicit
- shared-edge ownership is explicit
- patch agreement rules are explicit

## Current Preferred Answer

Seam sampling is:

- tessellator-owned
- keyed by seam identity and tessellation request
- reused across all participating patch-boundary uses
