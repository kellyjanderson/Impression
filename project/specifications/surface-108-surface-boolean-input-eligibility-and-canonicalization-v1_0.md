# Surface Spec 108: Surface Boolean Input Eligibility and Canonicalization (v1.0)

## Overview

This specification defines the canonical input contract for surface-body boolean work.

## Backlink

- [Surface Spec 102: Surface-Body Boolean Replacement (v1.0)](surface-102-surface-body-boolean-replacement-v1_0.md)

## Scope

This specification covers:

- boolean-eligible `SurfaceBody` inputs
- shell/seam/trim preconditions
- canonicalization required before boolean execution

## Behavior

This branch must define:

- what body classifications are accepted for boolean input
- what normalization or validation occurs before boolean execution
- what incompatibilities block execution versus downgrade to compatibility paths

## Constraints

- mesh tessellation may not become the primary boolean modeling truth
- input rejection rules must be explicit and deterministic
- required canonicalization must be documented for public callers

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- boolean input eligibility is explicit for `SurfaceBody` callers
- canonicalization or validation steps are explicit
- verification requirements are defined by its paired test specification

