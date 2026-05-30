# Surface Spec 97: Loft Split/Merge Surface Patch Orchestration (v1.0)

## Overview

This specification defines how loft split/merge planning stages are converted
into coordinated surface patches rather than directly triangulated branch
geometry.

## Backlink

Parent specification:

- [Surface Spec 06: Loft Surface Refactor Track (v1.0)](surface-06-loft-surface-refactor-track-v1_0.md)

## Scope

This specification covers:

- staged branch execution into surface patches
- synthetic birth/death surface handling
- split/merge seam coordination

## Behavior

This branch must define:

- staged split/merge operators produce coordinated patch groups
- synthetic birth/death closures are represented in surface-native form
- branch staging respects shell-level seam and boundary-use truth

## Constraints

- split/merge execution must remain deterministic
- no branch stage may emit mesh-only geometry on the canonical path
- staged patches must remain valid for seam-first tessellation

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- split/merge patch orchestration rules are explicit
- synthetic branch surface handling is explicit
- seam coordination expectations are explicit
