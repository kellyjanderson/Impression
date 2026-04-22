# Surface Spec 43: Scene-to-Tessellation Consumer Handoff Contract (v1.0)

## Overview

This specification defines where scene traversal ends and tessellation consumer
handoff begins.

## Backlink

Parent specification:

- [Surface Spec 14: Surface Scene Object and Group Contract (v1.0)](surface-14-surface-scene-object-and-group-contract-v1_0.md)

## Scope

This specification covers:

- scene traversal output shape
- consumer handoff boundary
- handoff guarantees provided to preview/export/analysis

## Behavior

This branch must define:

- what traversal yields to downstream tessellation consumers
- whether handoff is node-based, body-based, or collection-based
- what ordering and metadata guarantees the handoff preserves

## Constraints

- handoff must not implicitly tessellate
- consumer input shape must be deterministic
- ordering and metadata must survive handoff explicitly

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one scene-to-consumer boundary contract.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- handoff payload shape is explicit
- the scene/tessellation boundary is explicit
- ordering and metadata preservation guarantees are explicit

