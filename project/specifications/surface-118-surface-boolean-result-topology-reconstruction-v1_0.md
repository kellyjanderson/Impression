# Surface Spec 118: Surface Boolean Result Topology Reconstruction (v1.0)

## Overview

This specification defines how surfaced boolean fragments are reconstructed into
result `SurfaceBody` topology.

## Backlink

- [Surface Spec 102: Surface-Body Boolean Replacement (v1.0)](surface-102-surface-body-boolean-replacement-v1_0.md)

## Scope

This specification covers:

- assembling classified surface fragments into result shells
- reconstructing trim loops from cut boundaries
- rebuilding seam, open-boundary, and boundary-use truth
- determining empty, single-shell, and multi-shell surfaced results

## Behavior

This branch must define:

- when surviving source fragments are reused versus rebuilt
- how new cut boundaries become trim loops and seam records
- how shell membership and adjacency are determined on the result body

The reconstruction stage is expected to:

- reuse surviving source fragments whenever their patch family and parameter
  meaning remain valid
- rebuild trim loops around cut regions rather than inventing mesh-owned
  replacement boundaries
- create new seam truth whenever reconstructed boundaries are shared across
  adjacent result patches

## Constraints

- result topology must remain surface-native and seam-aware
- reconstruction must not rely on post-mesh stitching as the primary result law
- shell and seam ownership must be explicit enough to support deterministic tessellation

## Refinement Status

Decomposed into final child leaves.

This parent branch does not yet represent executable work directly.

## Child Specifications

- [Surface Spec 130: Surface Boolean No-Cut and Exact-Reuse Result Reconstruction](surface-130-surface-boolean-no-cut-and-exact-reuse-result-reconstruction-v1_0.md)
- [Surface Spec 131: Surface Boolean Overlap Cut-Boundary Trim-Loop Reconstruction for Initial Box Slice](surface-131-surface-boolean-overlap-cut-boundary-trim-loop-reconstruction-for-initial-box-slice-v1_0.md)
- [Surface Spec 132: Surface Boolean Overlap Shell and Seam Assembly for Initial Box Slice](surface-132-surface-boolean-overlap-shell-and-seam-assembly-for-initial-box-slice-v1_0.md)
- [Surface Spec 133: Surface Boolean Result Outcome Classification for the Initial Slice](surface-133-surface-boolean-result-outcome-classification-for-the-initial-slice-v1_0.md)

## Acceptance

This specification is ready for implementation planning when:

- no-cut reuse, overlap trim reconstruction, shell/seam assembly, and result classification concerns are separated into bounded final leaves
- the child set makes result topology reconstruction explicit at trim, shell, seam, and classification level
- paired verification leaves exist for the final children
