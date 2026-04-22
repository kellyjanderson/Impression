# Surface Spec 86: Default Attached-Transform Policy (v1.0)

## Overview

This specification defines whether transforms are attached to surface objects by
default rather than baked into geometry immediately.

## Backlink

Parent specification:

- [Surface Spec 29: Transform Attachment Versus Baked Geometry Policy (v1.0)](surface-29-transform-attachment-vs-baked-policy-v1_0.md)

## Scope

This specification covers:

- default transform attachment behavior
- transform representation at rest
- default caller-visible object state

## Behavior

This branch must define:

- transforms are attached by default
- attached transform state lives with surface objects rather than forcing eager
  geometry duplication
- downstream systems must assume non-baked objects remain geometry/topology-
  first with placement carried separately
- baking occurs only when required by downstream consumers or explicit
  structural rewrite

## Constraints

- default policy must be explicit
- attached-transform state must be deterministic
- callers must not need to guess whether geometry is baked

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- default transform policy is explicit
- transform storage location is explicit
- downstream assumptions for non-baked objects are explicit

## Current Preferred Answer

The v1 policy is attached-transform first, baked-geometry second.
