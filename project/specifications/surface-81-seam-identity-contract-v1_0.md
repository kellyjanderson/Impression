# Surface Spec 81: Seam Identity Contract (v1.0)

## Overview

This specification defines what makes one seam the same seam across traversal,
serialization, and reuse.

## Backlink

Parent specification:

- [Surface Spec 27: Seam Identity and Ownership Policy (v1.0)](surface-27-seam-identity-ownership-policy-v1_0.md)

## Scope

This specification covers:

- seam identity definition
- identity stability expectations
- seam comparison rules

## Behavior

This branch must define:

- seam identity is determined by stable seam record identity within an owning
  shell
- seam equality compares seam identity, not incidental traversal position
- seam identity changes when seam classification, participating uses, shared 3D
  boundary geometry, or continuity class changes
- attached transforms do not by themselves redefine seam identity unless the
  transform is baked into new kernel geometry/topology

## Constraints

- seam identity must be deterministic
- equality rules must be explicit
- identity must not collapse distinct seams accidentally

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- seam identity definition is explicit
- seam equality rules are explicit
- seam identity stability rules are explicit

## Current Preferred Answer

The preferred minimal seam identity inputs are:

- `seam_id`
- `shell_id`
- `classification`
- participating `uses`
- `shared_geometry_ref`
- `continuity_class`
