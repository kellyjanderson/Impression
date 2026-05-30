# Loft Spec 38: Ambiguity Record Minimal Locator Contract (v1.0)

## Overview

This specification defines the minimal locator payload carried by next-
generation loft ambiguity records.

## Backlink

Parent specification:

- [Loft Spec 25: Ambiguity, Constraint Request, and Diagnostic Surface (v1.0)](loft-25-ambiguity-constraint-request-and-diagnostic-surface-v1_0.md)

## Scope

This specification covers:

- blocked interval locator
- topology-state locator
- ambiguous-region locator
- optional relationship-group locator

## Behavior

The minimal ambiguity locator is currently represented by `LoftAmbiguityRecord`.

Current fields are:

- `interval`
- `topology_state_index`
- `ambiguous_region_indices`
- `ambiguity_class`
- optional `relationship_group`

The topology state holding the ambiguity is currently identified by
`topology_state_index`.

Ambiguous regions are represented as region indices local to that topology
state. When a finer-grained region subset is not yet available, the current
record may carry an empty tuple rather than inventing false precision.

## Constraints

- ambiguity payload must remain minimal
- locator fields must be machine-consumable
- diagnostic shape must not over-specify internal planner detail

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- interval, topology, and region locators are explicit
- relationship-group locator rules are explicit
- minimal payload boundaries are explicit
