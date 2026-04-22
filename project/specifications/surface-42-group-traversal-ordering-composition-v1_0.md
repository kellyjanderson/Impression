# Surface Spec 42: Group Traversal, Ordering, and Composition Rules (v1.0)

## Overview

This specification defines how groups of surface-native scene nodes are ordered
and traversed for downstream consumers.

## Backlink

Parent specification:

- [Surface Spec 14: Surface Scene Object and Group Contract (v1.0)](surface-14-surface-scene-object-and-group-contract-v1_0.md)

## Scope

This specification covers:

- deterministic group ordering
- traversal semantics
- composition rules for nested groups

## Behavior

This branch must define:

- how child order is stored and preserved
- how traversal proceeds through nested groups
- which composition operations are structural versus geometric

## Constraints

- traversal order must be deterministic
- nested groups must not create ambiguous flattening order
- structural grouping must remain distinct from tessellation

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
focuses on one traversal and ordering contract.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- child ordering rules are explicit
- traversal semantics are explicit
- structural grouping boundaries are explicit

