# Surface Spec 41: Scene Node Surface Payload Contract (v1.0)

## Overview

This specification defines what a scene node stores when the scene graph
becomes surface-native.

## Backlink

Parent specification:

- [Surface Spec 14: Surface Scene Object and Group Contract (v1.0)](surface-14-surface-scene-object-and-group-contract-v1_0.md)

## Scope

This specification covers:

- scene node payload shape
- node-level linkage to surface objects
- node-level transform and metadata references

## Behavior

This branch must define:

- the minimum scene node data required to reference a surface object
- whether nodes own or reference surface bodies
- the node-level fields downstream traversal may rely on

## Constraints

- scene nodes must not fall back to mesh-native payloads
- payload ownership/reference semantics must be explicit
- node fields must remain stable enough for traversal and caching

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one node payload contract.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- node payload fields are explicit
- payload ownership/reference semantics are explicit
- downstream reliance points are explicit

