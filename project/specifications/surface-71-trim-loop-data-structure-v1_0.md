# Surface Spec 71: Trim Loop Data Structure Contract (v1.0)

## Overview

This specification defines the data structure used to represent trim loops.

## Backlink

Parent specification:

- [Surface Spec 24: Trim-Loop Representation and Ownership (v1.0)](surface-24-trim-loop-representation-v1_0.md)

## Scope

This specification covers:

- trim loop geometry representation
- vertex/curve segment representation choices
- deterministic loop ordering rules

## Behavior

This branch must define:

- trim loops are patch-local 2D parameter-space loop structures in v1
- trim loops are represented as ordered closed loops in patch parameter space
- trim loops distinguish `outer` and `inner` role
- loop ordering and closure are explicit and deterministic
- 3D shared-boundary geometry is not stored on the trim loop itself; that truth
  lives on seams when the boundary is shared

## Constraints

- the structure must be deterministic
- representation complexity must stay within v1 scope
- closure and ordering semantics must be explicit

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- trim loop fields are explicit
- representation class is explicit
- ordering and closure semantics are explicit

## Current Preferred Answer

For v1:

- trim loops are parameter-space structures
- patch-local trim truth and shared seam truth are separate but coordinated
