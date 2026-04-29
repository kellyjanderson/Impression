# Loft Spec 29: Topology State Normalization Invariants (v1.0)

## Overview

This specification defines the normalization invariants that topology must
satisfy before next-generation loft planning begins.

## Backlink

Parent specification:

- [Loft Spec 22: Placed Topology State and Directional Correspondence Contract (v1.0)](loft-22-placed-topology-state-and-directional-correspondence-v1_0.md)

## Scope

This specification covers:

- normalization prerequisites
- containment and polarity invariants
- progression-order validity prerequisites

## Behavior

Planner entry requires a canonicalized topology state per station.

Current normalization guarantees:

- `Station.section` is canonicalized on construction when present
- regions are deterministically ordered by loft loop sort key
- holes are deterministically ordered by the same sort key within each region
- outer and inner loops are anchored deterministically for stable downstream
  signatures
- the canonicalized topology is exposed through `normalized_topology_state`

Placed-state versus topology-state responsibilities:

- placed-state validation owns progression/frame validity and correspondence
  arity checks
- topology-state normalization owns region ordering, hole ordering, and loop
  anchoring

Rejected malformed/non-normalized cases at planner entry include:

- invalid section-station ordering or frame validity
- correspondence arrays whose lengths do not match the normalized region count
- directional correspondence provided without section topology

## Constraints

- invalid input must be rejected before structural interpretation
- normalization requirements must be explicit
- planner entry invariants must be deterministic

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- planner-entry invariants are explicit
- rejection conditions are explicit
- placed-state versus topology-state normalization responsibilities are explicit
- deterministic loop-order and loop-anchor invariants are explicit
