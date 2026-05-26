# Surface Spec 117 Test: Surface Boolean Intersection, Classification, and Operand Splitting

## Overview

This test specification defines verification for surfaced boolean intersection,
classification, and operand splitting.

## Backlink

- [Surface Spec 117: Surface Boolean Intersection, Classification, and Operand Splitting (v1.0)](../specifications/surface-117-surface-boolean-intersection-classification-and-operand-splitting-v1_0.md)

## Refinement Status

Decomposed into final child verification leaves.

This parent test branch does not yet represent executable verification work directly.

## Child Test Specifications

- [Surface Spec 126 Test: Surface Boolean Exact Body-Relation Classification and No-Cut Gate](surface-126-surface-boolean-exact-body-relation-classification-and-no-cut-gate-v1_0.md)
- [Surface Spec 127 Test: Surface Boolean Initial Box/Box Cut-Curve Discovery](surface-127-surface-boolean-initial-box-box-cut-curve-discovery-v1_0.md)
- [Surface Spec 128 Test: Surface Boolean Patch-Local Trim-Fragment Mapping for Initial Box Slice](surface-128-surface-boolean-patch-local-trim-fragment-mapping-for-initial-box-slice-v1_0.md)
- [Surface Spec 129 Test: Surface Boolean Initial Box Slice Fragment Classification and Split Records](surface-129-surface-boolean-initial-box-slice-fragment-classification-and-split-records-v1_0.md)

## Acceptance

- the child verification leaves cover no-cut gating, cut-curve discovery, trim-fragment mapping, and split-record verification
- the child set verifies the bounded surfaced intersection stage without relying on mesh fallback
