# Surface Spec 92: Stable Surface Identity Contract (v1.0)

## Overview

This specification defines what stable identity means for bodies, shells, and
patches.

## Backlink

Parent specification:

- [Surface Spec 31: Stable Identity and Caching Keys for Surface Objects (v1.0)](surface-31-stable-identity-caching-keys-v1_0.md)

## Scope

This specification covers:

- identity definition
- identity scope across structural levels
- identity stability expectations

## Behavior

This branch must define:

- what constitutes identity for bodies, shells, patches, and seam-participating
  kernel records
- what level-specific identities are guaranteed
- how stable identity differs from incidental traversal position
- what topology/geometry changes force identity change

## Constraints

- identity definitions must be explicit
- identity must be deterministic
- identity must not depend on mesh-era assumptions

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- level-specific identity definitions are explicit
- stability expectations are explicit
- non-identity incidental ordering is explicitly separated

## Current Preferred Answer

Identity should change on geometry/topology truth changes, including:

- body shell-membership changes
- shell patch/seam membership changes
- seam classification/use/shared-geometry changes
- boundary-use patch/seam/trim/orientation changes
