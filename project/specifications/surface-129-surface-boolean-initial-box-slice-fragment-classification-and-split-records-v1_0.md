# Surface Spec 129: Surface Boolean Initial Box Slice Fragment Classification and Split Records (v1.0)

## Overview

This specification defines deterministic fragment classification and
executor-internal split records for the initial box/box surfaced boolean slice.

## Backlink

- [Surface Spec 117: Surface Boolean Intersection, Classification, and Operand Splitting (v1.0)](surface-117-surface-boolean-intersection-classification-and-operand-splitting-v1_0.md)

## Scope

This specification covers:

- classifying affected source fragments as inside, outside, or on the opposing operand
- operation-aware fragment selection for union, difference, and intersection
- canonical split-record payloads before result reconstruction

## Behavior

This leaf must define:

- how the initial box slice derives fragment classifications from the cut and no-cut stage outputs
- how union, difference, and intersection interpret those classifications
- how surviving and removed fragments are represented in deterministic split records

## Constraints

- fragment labels must remain deterministic for equal operands and equal request state
- split records must remain surfaced executor state rather than mesh-owned topology truth
- unsupported fragment-classification ambiguity must remain explicit

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- fragment classification is explicit for the initial box slice
- operation-aware split-record semantics are explicit
- canonical split records are explicit enough to drive result reconstruction
- verification requirements are defined by its paired test specification
