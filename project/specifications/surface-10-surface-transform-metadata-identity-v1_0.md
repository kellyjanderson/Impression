# Surface Spec 10: Surface Transform, Metadata, and Identity Policy (v1.0)

## Overview

This specification defines the branch responsible for how surface-native
objects carry transforms, colors/material metadata, stable identity, and other
non-topological properties across the modeling stack.

## Backlink

Parent specification:

- [Surface Spec 02: Surface Core Data Model (v1.0)](surface-02-surface-core-data-model-v1_0.md)

## Scope

This specification covers:

- baked versus attached transform policy
- metadata carriage across bodies/shells/patches
- stable identity rules for traversal, caching, and planning
- the minimum object-level properties that the rest of Impression may rely on

## Behavior

This branch must define:

- whether transforms are attached to surface objects or immediately baked
- where color and material-like metadata live
- how identity is preserved across composition and transformation
- what metadata is kernel-native versus consumer-specific

These rules must be strong enough to support:

- transforms
- scene/group composition
- deterministic caching and tessellation requests
- future modeling and rendering integration

## Constraints

- transform policy must be explicit and consistent
- metadata ownership must not be ambiguous between body, shell, and patch levels
- identity rules must be deterministic enough for caching and planning
- this branch must avoid accidental dependence on mesh IDs or renderer-specific
  concepts

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 29: Transform Attachment Versus Baked Geometry Policy (v1.0)](surface-29-transform-attachment-vs-baked-policy-v1_0.md)
- [Surface Spec 30: Surface Metadata Placement Contract (v1.0)](surface-30-surface-metadata-placement-contract-v1_0.md)
- [Surface Spec 31: Stable Identity and Caching Keys for Surface Objects (v1.0)](surface-31-stable-identity-caching-keys-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- transform policy is explicit
- metadata ownership is explicit
- identity rules are bounded well enough for scene and tessellation consumers to
  rely on
- the child branches define transform, metadata, and identity concerns as final
  implementation-sized leaves
