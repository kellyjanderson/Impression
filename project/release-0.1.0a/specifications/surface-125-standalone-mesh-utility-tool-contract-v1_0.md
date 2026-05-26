# Surface Spec 125: Standalone Mesh Utility Tool Contract (v1.0)

## Overview

This specification defines standalone mesh utilities that may remain valuable in
Impression without being treated as canonical modeling features.

## Backlink

- [Surface Spec 121: Mesh Analysis and Repair Toolchain Program (v1.0)](surface-121-mesh-analysis-and-repair-toolchain-program-v1_0.md)

## Scope

This specification covers:

- standalone mesh CSG as a possible analysis or repair utility
- standalone mesh hull or similar utilities
- other explicit mesh-only helpers that are useful as tools rather than modeling truth

## Behavior

This branch must define:

- what standalone mesh utilities are retained
- what user-facing posture they have
- how they are kept separate from canonical surfaced modeling APIs

## Constraints

- standalone mesh utilities must be clearly marked as tools, not canonical model generators
- retained utilities must have a real analysis, repair, or debugging purpose
- utilities without a durable purpose should be deletion candidates instead

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- retained standalone mesh utilities are explicit
- their boundary from canonical surfaced modeling is explicit
- verification requirements are defined by its paired test specification
