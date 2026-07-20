# Surface Spec 61: Deterministic Body/Shell Traversal and Ordering Rules (v1.0)

## Overview

This specification defines deterministic ordering for traversing bodies and
their shells.

## Backlink

Parent specification:

- [Surface Spec 20: Surface Body and Shell Ownership Rules (v1.0)](surface-20-surface-body-shell-ownership-rules-v1_0.md)

## Scope

This specification covers:

- shell ordering within a body
- traversal stability guarantees
- serialization-visible ordering rules

## Behavior

This branch must define:

- how shell order is derived and preserved
- what downstream consumers may assume about traversal stability
- whether ordering participates in identity and cache derivation

## Constraints

- ordering must be deterministic
- traversal must be stable across equivalent loads
- ordering must not depend on incidental memory order

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- shell ordering rules are explicit
- traversal stability guarantees are explicit
- cache/identity implications are explicit

