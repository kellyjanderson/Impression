# Surface Spec 127: Surface Boolean Initial Box/Box Cut-Curve Discovery (v1.0)

## Overview

This specification defines the first bounded surfaced cut-curve discovery lane
for overlapping axis-aligned planar box-style operands.

## Backlink

- [Surface Spec 117: Surface Boolean Intersection, Classification, and Operand Splitting (v1.0)](surface-117-surface-boolean-intersection-classification-and-operand-splitting-v1_0.md)

## Scope

This specification covers:

- patch/patch intersection discovery for the initial box/box slice
- canonical 3D cut-curve output before trim-space mapping
- deterministic cut-curve ordering and identity for the bounded initial slice

## Behavior

This leaf must define:

- which planar patch pairs are intersected in the initial box/box lane
- what 3D line-segment records are canonical for those cuts
- how overlapping, boundary-touch, or coplanar cases remain explicit when the bounded lane does not support them

## Constraints

- cut discovery must not rely on mesh clipping truth
- equal inputs and equal request state must produce identical cut-curve identities and ordering
- unsupported overlap shapes must remain explicit rather than silently approximated

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the initial box/box cut-curve discovery lane is explicit
- canonical 3D cut-curve output and ordering are explicit
- unsupported cut-discovery cases for the bounded lane remain explicit
- verification requirements are defined by its paired test specification
