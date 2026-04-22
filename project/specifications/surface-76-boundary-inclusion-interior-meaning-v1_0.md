# Surface Spec 76: Boundary Inclusion and Interior Meaning Rules (v1.0)

## Overview

This specification defines how trim boundaries determine retained interior
regions of a patch.

## Backlink

Parent specification:

- [Surface Spec 25: Trim Validity, Orientation, and Boundary Semantics (v1.0)](surface-25-trim-validity-orientation-boundary-v1_0.md)

## Scope

This specification covers:

- boundary inclusion semantics
- retained interior meaning
- hole versus outside interpretation

## Behavior

This branch must define:

- whether boundaries are inclusive, exclusive, or tolerance-qualified
- how outer and inner trims combine to define retained patch area
- what region semantics downstream tessellation uses

## Constraints

- retained-area meaning must be explicit
- boundary inclusion semantics must be deterministic
- hole/outside semantics must not depend on mesh heuristics

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- boundary inclusion semantics are explicit
- retained interior meaning is explicit
- hole versus outside interpretation is explicit

