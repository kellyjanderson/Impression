# Testing Spec 20: Computer Vision Cross-Space Anchoring Contract for Handedness Verification (v1.0)

## Overview

This specification defines the explicit space-alignment contract required
before handedness or mirror classification can run honestly.

## Backlink

- [Testing Spec 07: Computer Vision Handedness and Mirror-Witness Verification (v1.0)](testing-07-computer-vision-handedness-and-mirror-witness-verification-v1_0.md)

## Scope

This specification covers:

- modeling-space, export-space, and viewer-space anchoring
- declared correspondence between those spaces
- prerequisite camera/framing and canonical-view dependencies
- failure posture when anchoring is missing or inconsistent

## Behavior

This leaf must define:

- how a handedness fixture declares the space relationships it depends on
- which space-alignment assumptions are prerequisites for the lane
- how missing or inconsistent anchoring fails before mirror classification
- which upstream contracts this lane depends on explicitly

## Constraints

- handedness verification must not assume implicit space agreement
- anchoring failures must remain distinct from mirrored-shape findings
- this leaf defines prerequisites rather than witness semantics

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- cross-space anchoring requirements are explicit
- prerequisite dependencies are explicit
- anchoring failure posture is explicit
- verification requirements are defined by its paired test specification
