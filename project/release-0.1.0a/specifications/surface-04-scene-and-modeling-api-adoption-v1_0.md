# Surface Spec 04: Scene and Modeling API Surface Adoption (v1.0)

## Overview

This specification defines the branch responsible for changing Impression’s
internal modeling contracts so tools, groups, transforms, and scene-level
composition operate on surfaces rather than meshes.

## Backlink

Parent specification:

- [Surface Spec 01: Surface-First Internal Model Program (v1.0)](surface-01-surface-first-internal-model-program-v1_0.md)

## Scope

This specification covers:

- the scene/container contract for surface bodies
- modeling tool return types
- transform behavior on surface-native objects
- color/material/metadata carriage through the surface layer
- internal API expectations for downstream tool composition

## Behavior

This branch must define how:

- primitives return surfaces
- modeling helpers return surfaces
- transforms consume and return surfaces
- groups/scenes hold surfaces
- rendering/export request tessellation from those stored surfaces

The architectural goal is that internal modeling tools no longer require
direct mesh output to interoperate.

## Constraints

- the public and internal API boundary must be explicit during migration
- the scene layer must not become half-mesh and half-surface in an undefined way
- metadata such as color must survive the migration cleanly
- transform semantics must remain deterministic and composable

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 14: Surface Scene Object and Group Contract (v1.0)](surface-14-surface-scene-object-and-group-contract-v1_0.md)
- [Surface Spec 15: Modeling API Surface Return-Type Adoption (v1.0)](surface-15-modeling-api-surface-return-type-adoption-v1_0.md)
- [Surface Spec 16: Surface Composition and Consumer Handoff Rules (v1.0)](surface-16-surface-composition-consumer-handoff-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- it is clear what core modeling APIs are expected to return
- it is clear how groups and transforms operate on those results
- surface storage and scene composition are defined without hidden mesh
  assumptions
- the child branches define API and scene adoption as final
  implementation-sized leaves
