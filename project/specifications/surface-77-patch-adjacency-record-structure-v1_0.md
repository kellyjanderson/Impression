# Surface Spec 77: Patch Adjacency Record Structure (v1.0)

## Overview

This specification defines the data structure used to represent adjacency
between patches.

## Backlink

Parent specification:

- [Surface Spec 26: Patch Adjacency Reference Contract (v1.0)](surface-26-patch-adjacency-reference-contract-v1_0.md)

## Scope

This specification covers:

- adjacency record fields
- source and target patch references
- boundary-reference payloads inside adjacency records

## Behavior

This branch must define:

- adjacency is preferably derived from seam truth rather than stored as the
  only durable kernel record
- if materialized, an adjacency record contains:
  - `shell_id`
  - `patch_pair`
  - `seam_id`
  - `boundary_use_pair`
  - `classification`
- adjacency records point between patches through shared seam membership, not
  duplicated edge truth

## Constraints

- the record structure must be deterministic
- references must be stable enough for seam and tessellation consumers
- boundary payloads must be explicit

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- adjacency record fields are explicit
- patch reference semantics are explicit
- boundary payload semantics are explicit

## Current Preferred Answer

Adjacency is a derived shell-level view over explicit seams and boundary uses.
