# Surface Spec 75: Trim Orientation Semantics (v1.0)

## Overview

This specification defines what trim orientation means in parameter space and
how downstream systems interpret that orientation.

## Backlink

Parent specification:

- [Surface Spec 25: Trim Validity, Orientation, and Boundary Semantics (v1.0)](surface-25-trim-validity-orientation-boundary-v1_0.md)

## Scope

This specification covers:

- trim winding/orientation meaning
- orientation invariants for outer and inner trims
- downstream use of orientation

## Behavior

This branch must define:

- what orientation is measured against
- how orientation differs between outer and inner trims
- what tessellation and capping systems may assume from orientation

## Constraints

- orientation semantics must be explicit
- orientation rules must be deterministic
- downstream consumers must not reinterpret orientation independently

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- trim orientation meaning is explicit
- category-specific orientation rules are explicit
- downstream orientation assumptions are explicit

