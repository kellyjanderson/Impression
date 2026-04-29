# Surface Spec 16: Surface Composition and Consumer Handoff Rules (v1.0)

## Overview

This specification defines the branch responsible for how composed surface
results are handed off to downstream consumers such as preview, export, and
analysis.

## Backlink

Parent specification:

- [Surface Spec 04: Scene and Modeling API Surface Adoption (v1.0)](surface-04-scene-and-modeling-api-adoption-v1_0.md)

## Scope

This specification covers:

- composed-object handoff semantics
- downstream consumer access patterns
- the boundary between composition and tessellation requests

## Behavior

This branch must define:

- how consumers receive one or more surface bodies
- how composed structures are flattened or traversed
- where tessellation handoff begins

## Constraints

- consumer handoff must be deterministic
- composition rules must not reintroduce mesh-first coupling
- the boundary between composition and tessellation must remain explicit

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 47: Surface Collection Consumer Interface (v1.0)](surface-47-surface-collection-consumer-interface-v1_0.md)
- [Surface Spec 48: Composition Flattening and Traversal Rules (v1.0)](surface-48-composition-flattening-traversal-v1_0.md)
- [Surface Spec 49: Composition-to-Tessellation Invocation Boundary (v1.0)](surface-49-composition-to-tessellation-boundary-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- consumer handoff is explicit
- composition traversal is explicit
- the composition/tessellation boundary is explicit
