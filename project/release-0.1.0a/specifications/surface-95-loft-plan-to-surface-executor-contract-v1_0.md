# Surface Spec 95: Loft Plan-to-Surface Executor Contract (v1.0)

## Overview

This specification defines how the next-generation loft executor consumes a
resolved loft plan and produces a surface-native result instead of a mesh.

## Backlink

Parent specification:

- [Surface Spec 06: Loft Surface Refactor Track (v1.0)](surface-06-loft-surface-refactor-track-v1_0.md)

## Scope

This specification covers:

- executor target type
- mapping from plan operators to surface-kernel objects
- shell/seam/boundary-use creation responsibilities

## Behavior

This branch must define:

- loft execution terminates in `SurfaceBody` rather than direct mesh emission
- executor consumes only resolved plan records plus surface-kernel contracts
- executor creates patches, seams, boundary uses, and shells needed for a valid
  loft-produced body

## Constraints

- executor must not infer missing topology policy
- executor must not bypass seam/boundary-use rules with patch-local mesh hacks
- output must be valid for downstream seam-first tessellation

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- loft executor output type is explicit
- mapping from resolved plan records to surface-kernel objects is explicit
- seam/shell ownership expectations are explicit
