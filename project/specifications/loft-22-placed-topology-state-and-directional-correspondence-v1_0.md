# Loft Spec 22: Placed Topology State and Directional Correspondence Contract (v1.0)

## Overview

This specification defines the canonical next-generation loft input object and
the authored directional correspondence model.

## Backlink

Parent specification:

- [Loft Spec 21: Next-Gen Loft Evolution Program (v1.0)](loft-21-nextgen-loft-evolution-program-v1_0.md)

## Scope

This specification covers:

- placed topology state structure
- progression and placement payload
- topology-state normalization requirements
- predecessor/successor correspondence fields

## Behavior

This branch must define:

- the fields required for a placed topology state
- which parts of that object are structural versus placement truth
- how directional correspondence is expressed
- what invariants must hold before planning begins

## Constraints

- structure and placement must remain conceptually distinct even when wrapped in
  one canonical input object
- standalone authored region IDs must not be reintroduced without new
  architectural justification
- directional correspondence must remain relationship-first

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Loft Spec 28: Placed Topology State Object Shape (v1.0)](loft-28-placed-topology-state-object-shape-v1_0.md)
- [Loft Spec 29: Topology State Normalization Invariants (v1.0)](loft-29-topology-state-normalization-invariants-v1_0.md)
- [Loft Spec 30: Directional Correspondence Field Contract (v1.0)](loft-30-directional-correspondence-field-contract-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- input object shape is explicit
- normalization invariants are explicit
- directional correspondence fields are explicit
