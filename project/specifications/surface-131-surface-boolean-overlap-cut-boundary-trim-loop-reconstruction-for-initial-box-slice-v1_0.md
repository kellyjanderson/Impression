# Surface Spec 131: Surface Boolean Overlap Cut-Boundary Trim-Loop Reconstruction for Initial Box Slice (v1.0)

## Overview

This specification defines how overlap cut boundaries become trim loops on
surviving surfaced fragments for the initial box/box boolean slice.

## Backlink

- [Surface Spec 118: Surface Boolean Result Topology Reconstruction (v1.0)](surface-118-surface-boolean-result-topology-reconstruction-v1_0.md)

## Scope

This specification covers:

- rebuilding trim loops around surviving overlap regions
- reusing unaffected planar fragments while introducing new cut-bounded trims where needed
- deterministic trim-loop orientation and categorization for the initial box slice

## Behavior

This leaf must define:

- how cut boundaries and surviving patch regions become result trim loops
- when surviving source fragments remain valid versus when new trimmed fragments are rebuilt
- how outer-versus-inner trim meaning is determined on reconstructed overlap fragments

## Constraints

- trim reconstruction must stay surfaced and parameter-space aware
- trim loops must not be replaced by mesh-owned boundary stitching
- trim-loop orientation and categorization must remain deterministic

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- overlap cut-boundary trim reconstruction is explicit for the initial box slice
- reuse-versus-rebuild rules for overlap fragments are explicit
- deterministic trim-loop orientation and categorization are explicit
- verification requirements are defined by its paired test specification
