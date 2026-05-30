# Surface Spec 01: Surface-First Internal Model Program (v1.0)

## Overview

This specification defines the program-level implementation branch for moving
Impression from a mesh-first internal model to a surface-first internal model.

It exists to coordinate the first-generation specification branches required to
make surfaces the primary geometric truth of the system.

## Backlink

Parent architecture:

- [Surface-First Internal Model Architecture](../architecture/surface-first-internal-model.md)

## Scope

This specification covers:

- the program boundary for surface-first migration
- the major implementation branches required before execution
- the separation between whole-system surface work and the later dedicated loft
  refactor path

This specification does not itself define final implementation-ready leaf work.

## Behavior

The surface-first program must produce an Impression architecture in which:

- modeling tools consume and produce surface-native geometry
- `Mesh` is a boundary artifact rather than the kernel truth
- preview/render/export request tessellation from surface bodies
- loft is refactored after the surface foundation is in place rather than being
  used to invent the foundation ad hoc

## Constraints

- the migration must preserve deterministic output expectations
- the migration must preserve a compatibility path for current mesh consumers
- the whole-system surface contract must be defined before the loft-specific
  refactor leaf work begins
- the first generation of child specifications must cover the whole relevant
  architecture breadth-first

## Refinement Status

Partially decomposed.

The surface-foundation branches are now decomposed into final child leaves.
Only the dedicated loft refactor track still requires another refinement round.

## Child Specifications

- [Surface Spec 02: Surface Core Data Model (v1.0)](surface-02-surface-core-data-model-v1_0.md)
- [Surface Spec 03: Tessellation Boundary and Rendering Contract (v1.0)](surface-03-tessellation-boundary-v1_0.md)
- [Surface Spec 04: Scene and Modeling API Surface Adoption (v1.0)](surface-04-scene-and-modeling-api-adoption-v1_0.md)
- [Surface Spec 05: Migration and Compatibility Path (v1.0)](surface-05-migration-and-compatibility-path-v1_0.md)
- [Surface Spec 06: Loft Surface Refactor Track (v1.0)](surface-06-loft-surface-refactor-track-v1_0.md)

## Acceptance

This parent specification branch is considered properly established when:

- the surface-first architecture has a durable parent document
- the work is decomposed into first-generation child specifications that cover
  the major system branches
- the loft refactor is explicitly represented as a downstream dedicated track
  rather than being mixed into the surface foundation ambiguously
