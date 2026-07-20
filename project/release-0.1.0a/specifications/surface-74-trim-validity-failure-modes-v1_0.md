# Surface Spec 74: Trim Validity Conditions and Failure Modes (v1.0)

## Overview

This specification defines when a trim is valid and how invalid trims fail.

## Backlink

Parent specification:

- [Surface Spec 25: Trim Validity, Orientation, and Boundary Semantics (v1.0)](surface-25-trim-validity-orientation-boundary-v1_0.md)

## Scope

This specification covers:

- validity conditions for trims
- invalid trim detection
- failure behavior for invalid trims

## Behavior

This branch must define:

- the conditions a trim must satisfy to be valid
- when invalidity is detected
- whether invalidity is rejected eagerly or deferred to specific consumers

## Constraints

- validity conditions must be explicit
- failure behavior must be deterministic
- invalid trims must not silently degrade into approximate behavior

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- validity conditions are explicit
- invalidity detection stage is explicit
- failure behavior is explicit

