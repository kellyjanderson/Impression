# Surface Spec 132: Surface Boolean Overlap Shell and Seam Assembly for Initial Box Slice (v1.0)

## Overview

This specification defines how reconstructed overlap fragments are assembled
into result shells, seams, and boundary-use truth for the initial box slice.

## Backlink

- [Surface Spec 118: Surface Boolean Result Topology Reconstruction (v1.0)](surface-118-surface-boolean-result-topology-reconstruction-v1_0.md)

## Scope

This specification covers:

- assembling reconstructed overlap fragments into result shells
- creating deterministic seam truth on shared cut boundaries
- determining whether reconstructed boundaries remain shared seams or open boundaries

## Behavior

This leaf must define:

- how reconstructed overlap fragments are grouped into shells
- how shared cut boundaries become seam records with explicit ownership
- how boundary-use truth is determined when a reconstructed boundary is not shared

## Constraints

- shell and seam assembly must remain surface-native and seam-aware
- overlap assembly must not rely on post-mesh stitching as the primary result law
- shared-versus-open boundary truth must remain explicit enough to support deterministic tessellation

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- overlap shell assembly is explicit for the initial box slice
- seam and boundary-use ownership are explicit on reconstructed cut boundaries
- the resulting shell/seam structure is explicit enough to support deterministic tessellation
- verification requirements are defined by its paired test specification
