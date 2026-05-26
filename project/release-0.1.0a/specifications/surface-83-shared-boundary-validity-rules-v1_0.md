# Surface Spec 83: Shared-Boundary Validity Rules (v1.0)

## Overview

This specification defines when two patches are considered to share a valid
boundary.

## Backlink

Parent specification:

- [Surface Spec 28: Shared-Boundary Validity and Continuity Rules (v1.0)](surface-28-shared-boundary-validity-continuity-v1_0.md)

## Scope

This specification covers:

- shared-boundary validity conditions
- tolerated mismatch versus invalid seam conditions
- validation timing

## Behavior

This branch must define:

- what conditions a shared boundary must satisfy to be valid
- which mismatches are tolerated versus rejected
- when shared-boundary validity is checked

## Constraints

- validity rules must be explicit
- tolerance policy must be explicit
- invalid boundaries must not silently degrade into best-effort stitching

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- shared-boundary validity conditions are explicit
- tolerated mismatch rules are explicit
- validation timing is explicit

