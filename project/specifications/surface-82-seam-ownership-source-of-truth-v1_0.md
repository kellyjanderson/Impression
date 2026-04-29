# Surface Spec 82: Seam Ownership and Source-of-Truth Policy (v1.0)

## Overview

This specification defines which kernel object owns seam truth and how that
ownership interacts with patches and adjacency records.

## Backlink

Parent specification:

- [Surface Spec 27: Seam Identity and Ownership Policy (v1.0)](surface-27-seam-identity-ownership-policy-v1_0.md)

## Scope

This specification covers:

- seam ownership
- source-of-truth location
- mutation/update responsibility for seam data

## Behavior

This branch must define:

- seam truth lives on explicit seam objects owned in shell context
- `SurfaceShell` owns seam membership and adjacency truth
- patches refer to seams through oriented boundary-use records
- derived adjacency views may be cached for lookup or tooling, but they do not
  supersede seam truth
- seam updates are shell-governed topology updates rather than patch-local edge
  rewrites

## Constraints

- source-of-truth must be explicit
- update responsibility must be unambiguous
- cached derived views must not supersede source-of-truth ownership

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- seam source-of-truth is explicit
- update responsibility is explicit
- derived-view limits are explicit

## Current Preferred Answer

For v1:

- shell owns seam truth
- seam is the boundary source of truth
- adjacency views are derived from seams
