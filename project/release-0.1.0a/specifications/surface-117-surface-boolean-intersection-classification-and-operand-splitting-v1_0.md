# Surface Spec 117: Surface Boolean Intersection, Classification, and Operand Splitting (v1.0)

## Overview

This specification defines the surfaced boolean execution stage that discovers
operand intersections, classifies cut fragments, and splits operands into
boolean-ready surface fragments.

## Backlink

- [Surface Spec 102: Surface-Body Boolean Replacement (v1.0)](surface-102-surface-body-boolean-replacement-v1_0.md)

## Scope

This specification covers:

- patch/patch intersection discovery between prepared operands
- mapping intersection curves into patch-local trim space
- fragment classification as inside, outside, or on the opposing operand
- deterministic operand splitting into temporary boolean fragments

## Behavior

This branch must define:

- what intersection outputs are canonical for the boolean executor
- how split boundaries are represented before result reconstruction
- how union, difference, and intersection share or differ in fragment classification

The canonical surfaced boolean intersection output should include, at minimum:

- canonical 3D cut-curve geometry
- per-patch trim-space boundary fragments for affected patches
- deterministic fragment labels relative to the opposing operand

## Constraints

- boolean execution must not demote operands to mesh-primary clipping truth
- fragment classification must remain deterministic for equal operands and equal request state
- unsupported intersection cases must remain explicit rather than silently approximated

## Refinement Status

Decomposed into final child leaves.

This parent branch does not yet represent executable work directly.

## Child Specifications

- [Surface Spec 126: Surface Boolean Exact Body-Relation Classification and No-Cut Gate](surface-126-surface-boolean-exact-body-relation-classification-and-no-cut-gate-v1_0.md)
- [Surface Spec 127: Surface Boolean Initial Box/Box Cut-Curve Discovery](surface-127-surface-boolean-initial-box-box-cut-curve-discovery-v1_0.md)
- [Surface Spec 128: Surface Boolean Patch-Local Trim-Fragment Mapping for Initial Box Slice](surface-128-surface-boolean-patch-local-trim-fragment-mapping-for-initial-box-slice-v1_0.md)
- [Surface Spec 129: Surface Boolean Initial Box Slice Fragment Classification and Split Records](surface-129-surface-boolean-initial-box-slice-fragment-classification-and-split-records-v1_0.md)

## Acceptance

This specification is ready for implementation planning when:

- no-cut gating, cut-curve discovery, trim-fragment mapping, and split-record concerns are separated into bounded final leaves
- the child set makes the initial surfaced intersection stage explicit without hiding another full implementation round
- paired verification leaves exist for the final children
