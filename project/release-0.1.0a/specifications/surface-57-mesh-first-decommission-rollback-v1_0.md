# Surface Spec 57: Mesh-First Decommission and Rollback Policy (v1.0)

## Overview

This specification defines when mesh-first internal paths may be removed and
what rollback posture must remain available until that point.

## Backlink

Parent specification:

- [Surface Spec 19: Surface Promotion and Mesh-First Decommission Gate (v1.0)](surface-19-surface-promotion-and-mesh-decommission-v1_0.md)

## Scope

This specification covers:

- decommission triggers for mesh-first internals
- rollback policy before decommission
- rollback removal conditions after decommission

## Behavior

Mesh-first internal paths may be removed only after canonical promotion is
explicitly achieved and the verification matrix has been satisfied.

Required rollback posture before decommission:

- explicit `backend="mesh"` compatibility paths for bounded public APIs
- working surface-to-mesh adapter path for legacy mesh consumers
- documented ability to keep preview/export operating through tessellation even
  while mesh compatibility remains

Rollback support may be retired only after:

- no active core subsystem depends on mesh-first internal truth
- the surfaced path has remained stable through a full verification cycle
- rollback retirement is explicitly approved rather than assumed

## Constraints

- decommission triggers must be explicit
- rollback posture must remain available until the trigger is met
- rollback retirement must not happen implicitly

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one decommission/rollback policy pair.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- decommission triggers are explicit
- rollback requirements are explicit
- rollback retirement conditions are explicit
