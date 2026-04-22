# Surface Spec 14: Surface Scene Object and Group Contract (v1.0)

## Overview

This specification defines the branch responsible for how scene containers,
groups, and other composition structures hold surface-native objects.

## Backlink

Parent specification:

- [Surface Spec 04: Scene and Modeling API Surface Adoption (v1.0)](surface-04-scene-and-modeling-api-adoption-v1_0.md)

## Scope

This specification covers:

- scene/container storage of surface bodies
- group composition rules
- the minimum scene-level object contract needed for downstream consumers

## Behavior

This branch must define:

- what kind of object a scene/group stores
- how grouped surface objects preserve transforms and metadata
- how scene containers hand composed objects to tessellation consumers

## Constraints

- scene storage must not quietly fall back to mesh-native truth
- group behavior must remain deterministic
- object composition must preserve identity and metadata contracts

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 41: Scene Node Surface Payload Contract (v1.0)](surface-41-scene-node-surface-payload-v1_0.md)
- [Surface Spec 42: Group Traversal, Ordering, and Composition Rules (v1.0)](surface-42-group-traversal-ordering-composition-v1_0.md)
- [Surface Spec 43: Scene-to-Tessellation Consumer Handoff Contract (v1.0)](surface-43-scene-to-tessellation-handoff-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- scene storage contracts are explicit
- group behavior is explicit
- composed surface objects can be handed off without hidden mesh assumptions
