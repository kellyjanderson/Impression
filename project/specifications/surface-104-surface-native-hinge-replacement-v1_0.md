# Surface Spec 104: Surface-Native Hinge Replacement (v1.0)

## Overview

This specification defines the surface-first replacement path for deprecated
mesh-primary hinge generators.

## Backlink

- [Surface Spec 99: Surface-Native Replacement Program (v1.0)](surface-99-surface-native-replacement-program-v1_0.md)

## Scope

This specification covers:

- traditional hinge leaves and pairs
- living hinge panels
- bistable hinge blanks

## Behavior

This branch must define:

- how hinge geometry is assembled from surface-native primitives and ops
- how assembly/group behavior works without mesh as primary document
- how hinge outputs flow through preview/export as boundary tessellation only

## Constraints

- hinge replacement must not depend on mesh-first boolean assembly as the canonical model
- public hinge replacement must have durable docs and showcase examples

## Refinement Status

Decomposed into final child leaves.

This parent branch does not yet represent executable work directly.

## Child Specifications

- [Surface Spec 114: Traditional Hinge Surface Assembly](surface-114-traditional-hinge-surface-assembly-v1_0.md)
- [Surface Spec 115: Living and Bistable Hinge Surface Assembly](surface-115-living-and-bistable-hinge-surface-assembly-v1_0.md)
- [Surface Spec 116: Hinge Public Handoff and Showcase Verification](surface-116-hinge-public-handoff-and-showcase-verification-v1_0.md)

## Acceptance

This specification is ready for implementation planning when:

- hinge replacement is split into traditional, living/bistable, and public-handoff leaves
- the child set covers the deprecated hinge branch without hidden follow-on passes
- paired verification leaves exist for the final children
