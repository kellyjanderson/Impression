# Loft Spec 30: Directional Correspondence Field Contract (v1.0)

## Overview

This specification defines how authored directional correspondence is expressed
for next-generation loft.

## Backlink

Parent specification:

- [Loft Spec 22: Placed Topology State and Directional Correspondence Contract (v1.0)](loft-22-placed-topology-state-and-directional-correspondence-v1_0.md)

## Scope

This specification covers:

- `predecessor_ids`
- `successor_ids`
- authored correspondence semantics

## Behavior

Directional correspondence is currently represented on `Station` as:

- `predecessor_ids`
- `successor_ids`

Each is stored as one normalized id-set per normalized region. Empty sets mean
no authored directional constraint for that region on that side.

The station exposes these aligned fields through:

- `predecessor_ids`
- `successor_ids`
- `directional_correspondence`

What they constrain:

- authored region-level correspondence intent across progression
- deterministic tie-breaking inputs once the planner begins consuming them

What they do not constrain:

- full region lineage
- execution geometry directly
- standalone authored region identity

Current contradiction behavior:

- directional correspondence without topology is invalid
- arity mismatch against normalized region count is invalid
- richer cross-station contradiction classification remains planner work

## Constraints

- authored correspondence must remain relationship-first
- standalone authored region IDs must not be required
- directional fields must be explicit enough for deterministic planner use

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- directional field semantics are explicit
- constraint scope is explicit
- contradiction behavior is explicit
