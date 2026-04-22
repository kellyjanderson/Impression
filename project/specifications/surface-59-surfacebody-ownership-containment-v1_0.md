# Surface Spec 59: SurfaceBody Ownership and Containment Contract (v1.0)

## Overview

This specification defines what a `SurfaceBody` owns and what it means for a
shell to be contained by a body.

## Backlink

Parent specification:

- [Surface Spec 20: Surface Body and Shell Ownership Rules (v1.0)](surface-20-surface-body-shell-ownership-rules-v1_0.md)

## Scope

This specification covers:

- `SurfaceBody` containment of shells
- ownership versus reference semantics
- body-level validity expectations

## Behavior

This branch must define:

- whether a body owns shells directly or references them indirectly
- what containment guarantees a body makes about its shell set
- what body-level validity checks downstream systems may rely on

## Constraints

- ownership semantics must be explicit
- containment must be deterministic
- body validity must not depend on mesh-derived repair

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- body ownership semantics are explicit
- body containment guarantees are explicit
- body-level validity assumptions are explicit

