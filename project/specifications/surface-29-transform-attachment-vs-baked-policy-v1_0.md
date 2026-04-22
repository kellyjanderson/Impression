# Surface Spec 29: Transform Attachment Versus Baked Geometry Policy (v1.0)

## Overview

This specification defines the branch responsible for deciding whether surface
objects carry attached transforms, baked geometry, or both.

## Backlink

Parent specification:

- [Surface Spec 10: Surface Transform, Metadata, and Identity Policy (v1.0)](surface-10-surface-transform-metadata-identity-v1_0.md)

## Scope

This specification covers:

- attached-transform policy
- baked-geometry policy
- where the boundary between those approaches lives

## Behavior

This branch must define:

- whether transforms are attached or baked by default
- when baking is permitted or required
- what downstream systems may assume about transformed objects

## Constraints

- transform policy must be explicit
- the branch must avoid hidden eager baking
- downstream consumers must not need to guess object state

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 86: Default Attached-Transform Policy (v1.0)](surface-86-default-attached-transform-policy-v1_0.md)
- [Surface Spec 87: Geometry Baking Triggers and Required Cases (v1.0)](surface-87-geometry-baking-triggers-v1_0.md)
- [Surface Spec 88: Downstream Transformed-Object Assumptions (v1.0)](surface-88-downstream-transformed-object-assumptions-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- default transform policy is explicit
- baking rules are explicit
- downstream assumptions are explicit
