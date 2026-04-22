# Surface Spec 15: Modeling API Surface Return-Type Adoption (v1.0)

## Overview

This specification defines the branch responsible for changing modeling APIs so
they return surface-native results rather than meshes.

## Backlink

Parent specification:

- [Surface Spec 04: Scene and Modeling API Surface Adoption (v1.0)](surface-04-scene-and-modeling-api-adoption-v1_0.md)

## Scope

This specification covers:

- return-type expectations for modeling functions
- internal versus public API adoption boundary
- migration shape for primitives and modeling helpers

## Behavior

This branch must define:

- which APIs are expected to return surfaces
- whether transitional adapters are visible or hidden
- how public documentation and internal contracts stay aligned during adoption

## Constraints

- return-type migration must be explicit
- public and internal adoption boundaries must not diverge ambiguously
- the branch must support later tool migration without redefining core rules

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 44: Primitive API Surface Return-Type Migration (v1.0)](surface-44-primitive-api-surface-return-migration-v1_0.md)
- [Surface Spec 45: Modeling Operation Surface Return-Type Migration (v1.0)](surface-45-modeling-op-surface-return-migration-v1_0.md)
- [Surface Spec 46: Public/Internal API Transition and Documentation Boundary (v1.0)](surface-46-public-internal-api-transition-boundary-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- target return types are explicit
- the adoption boundary is explicit
- documentation and API direction are aligned
