# Surface Spec 120: Surface Boolean Initial Executable Scope and Reference Fixture Matrix (v1.0)

## Overview

This specification defines the intentionally bounded first executable slice of
surfaced boolean operations together with the regression fixture set required to
promote that slice.

## Backlink

- [Surface Spec 102: Surface-Body Boolean Replacement (v1.0)](surface-102-surface-body-boolean-replacement-v1_0.md)

## Scope

This specification covers:

- the initial supported surfaced boolean operation set
- the initial supported operand families and trim complexity
- explicit unsupported surfaced boolean cases
- required reference images, reference STL files, and regression fixtures

## Behavior

This branch must define:

- what surfaced boolean cases are executable first
- what cases remain structured unsupported results
- what fixture matrix proves surfaced CSG is geometrically stable enough to promote

The initial executable surfaced boolean scope should be intentionally bounded
to:

- exactly two operands
- single-shell, connected, closed-valid `SurfaceBody` operands
- the operation families `union`, `difference`, and `intersection`
- operand bodies composed only of the currently required V1 patch families
  already present in the surface kernel
- simple closed shells without authored multi-shell assemblies

The initial unsupported surfaced boolean set should remain explicit for cases
such as:

- multi-shell operands
- open-surface operands
- self-intersecting operands
- cases requiring broad healing beyond the bounded cleanup allowed by surfaced
  boolean validity rules

The first named reference-fixture matrix should include at least:

- `surfacebody/csg_union_box_post`
- `surfacebody/csg_difference_slot`
- `surfacebody/csg_intersection_box_sphere`

## Constraints

- the initial executable slice must be intentionally smaller than the full general CSG problem
- unsupported cases must remain explicit rather than silently downgrading to mesh truth
- promotion evidence must include durable rendered and exported references

## Refinement Status

Decomposed into final child leaves.

This parent branch does not yet represent executable work directly.

## Child Specifications

- [Surface Spec 136: Surface Boolean Initial Executable Scope and Unsupported-Case Matrix](surface-136-surface-boolean-initial-executable-scope-and-unsupported-case-matrix-v1_0.md)
- [Surface Spec 138: Surface Boolean Reference Fixture Composition and Slice Verification](surface-138-surface-boolean-reference-fixture-composition-and-slice-verification-v1_0.md)
- [Surface Spec 137: Surface Boolean Initial Reference Fixture Matrix and Promotion Gates](surface-137-surface-boolean-initial-reference-fixture-matrix-and-promotion-gates-v1_0.md)

## Acceptance

This specification is ready for implementation planning when:

- executable scope and unsupported-case posture are isolated from reference-fixture and promotion-gate work
- the child set makes the first promotable surfaced CSG slice explicit without hiding another full implementation round
- paired verification leaves exist for the final children
