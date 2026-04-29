# Surface Spec 39: Closed-Body Watertight Tessellation Contract (v1.0)

## Overview

This specification defines the conditions under which a valid closed surface
body must tessellate to a watertight mesh.

## Backlink

Parent specification:

- [Surface Spec 13: Seam-Consistent Tessellation and Watertight Output Rules (v1.0)](surface-13-seam-consistent-tessellation-watertightness-v1_0.md)

## Scope

This specification covers:

- watertightness preconditions
- watertightness guarantees for closed bodies
- failure behavior when preconditions are not met

## Behavior

This branch must define:

- a closed valid body is determined from shell-level shared/open boundary truth,
  not from post-mesh repair or metadata-only declaration
- watertight tessellation for closed valid bodies depends on seam-first
  shared-boundary sampling and consistent reuse of seam vertices across patches
- tessellation must fail fast when closed-body preconditions are not met rather
  than emitting a supposedly closed but non-watertight mesh

## Constraints

- closed-body guarantees must be binary and testable
- the contract must not rely on later mesh repair
- failure behavior must distinguish invalid input from tessellation defects

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one closed-body output guarantee and its failure gate.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- closed-body preconditions are explicit
- watertight output guarantees are explicit
- fail-fast conditions are explicit

## Current Preferred Answer

For v1:

- shell closure eligibility is kernel-topology truth
- watertightness is a consequence of shared seam truth and seam-first
  tessellation
- mesh repair is verification or fallback tooling, not the primary contract
