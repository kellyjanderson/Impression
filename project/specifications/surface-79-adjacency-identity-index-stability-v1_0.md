# Surface Spec 79: Adjacency Identity and Index Stability Rules (v1.0)

## Overview

This specification defines how adjacency references remain stable across
loading, copying, and traversal.

## Backlink

Parent specification:

- [Surface Spec 26: Patch Adjacency Reference Contract (v1.0)](surface-26-patch-adjacency-reference-contract-v1_0.md)

## Scope

This specification covers:

- identity semantics for adjacency records
- index stability requirements
- persistence expectations for adjacency references

## Behavior

This branch must define:

- whether adjacency identity is record-based, seam-based, or patch-pair based
- what indices must remain stable
- how copies and reordered traversals preserve adjacency meaning

## Constraints

- identity rules must be deterministic
- index stability must be explicit
- persistence expectations must not depend on incidental traversal order

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- adjacency identity rules are explicit
- index stability rules are explicit
- persistence/copy semantics are explicit

