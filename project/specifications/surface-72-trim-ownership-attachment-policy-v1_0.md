# Surface Spec 72: Trim Ownership and Attachment Policy (v1.0)

## Overview

This specification defines which kernel object owns trim loops and how trims
attach to patches.

## Backlink

Parent specification:

- [Surface Spec 24: Trim-Loop Representation and Ownership (v1.0)](surface-24-trim-loop-representation-v1_0.md)

## Scope

This specification covers:

- trim ownership
- trim attachment to patches or shells
- lifetime rules for owned trims

## Behavior

This branch must define:

- trims are owned by patches rather than shells
- trims attach to the surface they constrain through patch-local 2D trim
  references in parameter space
- shared boundaries use seam truth for 3D/topological meaning while retaining
  patch-owned trim references for local parameter-space meaning
- when patches are transformed by attached transforms, trims remain attached to
  the patch; when geometry is baked or structurally rewritten, trim validity
  must be re-evaluated

## Constraints

- ownership must be explicit
- attachment semantics must be deterministic
- lifetime rules must not create dangling boundary truth

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- trim ownership is explicit
- trim attachment rules are explicit
- trim lifetime rules are explicit

## Current Preferred Answer

For v1:

- trims are patch-owned
- seams are shell-owned
- a boundary that is both trimmed and shared carries:
  - one seam-owned 3D boundary truth
  - one patch-owned 2D trim reference per use
