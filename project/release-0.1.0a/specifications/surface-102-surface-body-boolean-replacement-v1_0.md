# Surface Spec 102: Surface-Body Boolean Replacement (v1.0)

## Overview

This specification defines the surface-body boolean replacement path for
deprecated mesh-first CSG operations.

## Backlink

- [Surface Spec 99: Surface-Native Replacement Program (v1.0)](surface-99-surface-native-replacement-program-v1_0.md)

## Scope

This specification covers:

- union
- difference
- intersection
- composition replacement for legacy `union_meshes`

## Behavior

This branch must define:

- boolean inputs in terms of `SurfaceBody` and shell/seam/trim truth
- boolean outputs as surface-native results
- the migration posture from mesh CSG to surface-body booleans

## Constraints

- boolean replacement must not rely on mesh as the primary modeling truth
- compatibility tessellation may exist only at consumer boundaries
- public boolean replacement must have durable docs and migration notes

## Refinement Status

Decomposed into final child leaves.

This parent branch does not yet represent executable work directly.

## Child Specifications

- [Surface Spec 108: Surface Boolean Input Eligibility and Canonicalization](surface-108-surface-boolean-input-eligibility-and-canonicalization-v1_0.md)
- [Surface Spec 109: Surface Boolean Result Contract and Failure Modes](surface-109-surface-boolean-result-contract-and-failure-modes-v1_0.md)
- [Surface Spec 110: Surface Boolean Public API Migration and Reference Verification](surface-110-surface-boolean-public-api-migration-and-reference-verification-v1_0.md)
- [Surface Spec 117: Surface Boolean Intersection, Classification, and Operand Splitting](surface-117-surface-boolean-intersection-classification-and-operand-splitting-v1_0.md)
- [Surface Spec 118: Surface Boolean Result Topology Reconstruction](surface-118-surface-boolean-result-topology-reconstruction-v1_0.md)
- [Surface Spec 119: Surface Boolean Validity, Healing Limits, and Metadata Propagation](surface-119-surface-boolean-validity-healing-limits-and-metadata-propagation-v1_0.md)
- [Surface Spec 120: Surface Boolean Initial Executable Scope and Reference Fixture Matrix](surface-120-surface-boolean-initial-executable-scope-and-reference-fixture-matrix-v1_0.md)

## Acceptance

This specification is ready for implementation planning when:

- boolean work is separated into input, execution, result, and public-migration leaves
- the child set covers deprecated CSG replacement without hidden follow-on passes
- paired verification leaves exist for the final children
