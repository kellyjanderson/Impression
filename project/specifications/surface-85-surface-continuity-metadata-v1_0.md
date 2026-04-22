# Surface Spec 85: Surface Continuity Metadata Contract (v1.0)

## Overview

This specification defines the continuity metadata exposed by the surface
kernel at patch boundaries.

## Backlink

Parent specification:

- [Surface Spec 28: Shared-Boundary Validity and Continuity Rules (v1.0)](surface-28-shared-boundary-validity-continuity-v1_0.md)

## Scope

This specification covers:

- continuity classification categories
- continuity metadata storage
- downstream use of continuity information

## Behavior

This branch must define:

- which continuity classes the kernel exposes in v1
- where continuity metadata is stored
- what tessellation and fairness systems may assume from that metadata

## Constraints

- continuity classes must be explicit and bounded
- metadata storage must be explicit
- downstream assumptions must not exceed the recorded contract

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- continuity classes are explicit
- metadata storage is explicit
- downstream assumptions are explicit

