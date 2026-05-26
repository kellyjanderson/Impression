# Surface Spec 103: Surface-Native Threading Replacement (v1.0)

## Overview

This specification defines the surface-first replacement path for deprecated
mesh-primary threading features.

## Backlink

- [Surface Spec 99: Surface-Native Replacement Program (v1.0)](surface-99-surface-native-replacement-program-v1_0.md)

## Scope

This specification covers:

- external threads
- internal threads and cutters
- threaded rods, nuts, and runout relief helpers

## Behavior

This branch must define:

- the canonical surface-native output of thread generation
- how analytic or structured thread surfaces are represented
- how threaded convenience builders terminate in surface-native outputs

## Constraints

- threading replacement must not require mesh as authored truth
- quality and fit behavior must remain explicit
- public threading replacement must have durable docs and examples

## Refinement Status

Decomposed into final child leaves.

This parent branch does not yet represent executable work directly.

## Child Specifications

- [Surface Spec 111: Structured Thread Surface Representation](surface-111-structured-thread-surface-representation-v1_0.md)
- [Surface Spec 112: Surface Thread Convenience Builders](surface-112-surface-thread-convenience-builders-v1_0.md)
- [Surface Spec 113: Thread Fit, Quality, and Regression Verification](surface-113-thread-fit-quality-and-regression-verification-v1_0.md)

## Acceptance

This specification is ready for implementation planning when:

- thread surface representation, convenience builders, and verification concerns are split into final children
- the child set covers deprecated threading replacement breadth-first
- paired verification leaves exist for the final children
