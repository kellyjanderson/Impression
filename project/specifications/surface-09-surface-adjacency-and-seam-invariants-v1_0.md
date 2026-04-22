# Surface Spec 09: Surface Adjacency and Seam Invariants (v1.0)

## Overview

This specification defines the branch responsible for how neighboring patches
relate to one another: adjacency ownership, seam identity, shared-boundary
invariants, and the minimum continuity guarantees exposed by the surface
kernel.

## Backlink

Parent specification:

- [Surface Spec 02: Surface Core Data Model (v1.0)](surface-02-surface-core-data-model-v1_0.md)

## Scope

This specification covers:

- adjacency representation
- seam identity and ownership
- shared-boundary invariants
- continuity classifications needed for tessellation and fairness work
- boundary cases such as open seams and external edges

## Behavior

This branch must define:

- how one patch refers to an adjacent patch
- whether seams are explicit first-class objects
- what it means for two patches to share the same boundary
- what continuity or compatibility metadata is recorded
- what downstream tessellation and modeling systems may assume about a valid seam

## Constraints

- adjacency must be deterministic and index-stable
- seam semantics must be strong enough to support watertight tessellation
- open boundaries and shared boundaries must be distinguishable
- the branch must avoid depending on mesh-first concepts for seam truth

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 26: Patch Adjacency Reference Contract (v1.0)](surface-26-patch-adjacency-reference-contract-v1_0.md)
- [Surface Spec 27: Seam Identity and Ownership Policy (v1.0)](surface-27-seam-identity-ownership-policy-v1_0.md)
- [Surface Spec 28: Shared-Boundary Validity and Continuity Rules (v1.0)](surface-28-shared-boundary-validity-continuity-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- seam ownership is explicit
- adjacency representation is explicit
- shared-boundary validity rules are strong enough for tessellation to use as a
  contract
- the child branches define adjacency and seam concerns as final
  implementation-sized leaves
