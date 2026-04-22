# Surface Spec 122: Mesh Capability Retention and Deletion Matrix (v1.0)

## Overview

This specification defines which mesh-era capabilities are retained as toolchain
and which are targeted for deletion.

## Backlink

- [Surface Spec 121: Mesh Analysis and Repair Toolchain Program (v1.0)](surface-121-mesh-analysis-and-repair-toolchain-program-v1_0.md)

## Scope

This specification covers:

- retained mesh toolchain capability
- removable mesh-era modeling capability
- the required inventory shape for tracking both

## Behavior

This branch must define:

- the criteria for retaining a mesh capability
- the criteria for deleting a mesh capability
- the canonical inventory fields used to track file, location, role, and intended end state

## Constraints

- retained mesh capability must have an explicit toolchain purpose
- deletion candidates must not be hidden inside vague “legacy” labels
- the retention/deletion matrix must stay consistent with surfaced architecture

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- retention and deletion criteria are explicit
- the required inventory format is explicit
- verification requirements are defined by its paired test specification
