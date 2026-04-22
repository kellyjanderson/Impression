# Surface Spec 78: Adjacency Lookup and Navigation Semantics (v1.0)

## Overview

This specification defines how callers navigate from one patch to adjacent
patches using adjacency data.

## Backlink

Parent specification:

- [Surface Spec 26: Patch Adjacency Reference Contract (v1.0)](surface-26-patch-adjacency-reference-contract-v1_0.md)

## Scope

This specification covers:

- adjacency query behavior
- one-step and repeated navigation semantics
- missing-adjacency behavior

## Behavior

This branch must define:

- how callers ask for adjacent patches
- what order multiple neighbors are returned in
- how open boundaries or missing adjacency are represented

## Constraints

- navigation semantics must be deterministic
- missing-adjacency behavior must be explicit
- navigation must not rely on mesh-derived neighborhood discovery

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- adjacency queries are explicit
- result ordering is explicit
- missing/open-boundary behavior is explicit

