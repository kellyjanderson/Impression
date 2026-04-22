# Surface Spec 26: Patch Adjacency Reference Contract (v1.0)

## Overview

This specification defines the branch responsible for how one patch references
another patch as an adjacent neighbor.

## Backlink

Parent specification:

- [Surface Spec 09: Surface Adjacency and Seam Invariants (v1.0)](surface-09-surface-adjacency-and-seam-invariants-v1_0.md)

## Scope

This specification covers:

- adjacency reference structure
- patch-to-patch reference semantics
- deterministic indexing or identity expectations for adjacency

## Behavior

This branch must define:

- what an adjacency reference contains
- how one patch points to another
- what downstream systems may rely on from those references

## Constraints

- adjacency references must be deterministic
- references must remain valid without mesh-derived identity
- the contract must be strong enough for seam and tessellation work

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 77: Patch Adjacency Record Structure (v1.0)](surface-77-patch-adjacency-record-structure-v1_0.md)
- [Surface Spec 78: Adjacency Lookup and Navigation Semantics (v1.0)](surface-78-adjacency-lookup-navigation-v1_0.md)
- [Surface Spec 79: Adjacency Identity and Index Stability Rules (v1.0)](surface-79-adjacency-identity-index-stability-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- adjacency references are explicit
- lookup semantics are explicit
- deterministic identity expectations are explicit
