# Topology Spec 08: Loft Module Narrowing

## Goal

Narrow `loft.py` to loft orchestration responsibilities and remove non-loft topology ownership.

## loft.py Should Own

- station sequencing and pair iteration
- frame generation/transport along path
- section placement in 3D
- section-to-section surface bridging
- loft-level orchestration and error propagation

## loft.py Should Not Own

- generic loop anchoring algorithms
- generic winding/containment/classification helpers
- generic path-to-region topology transforms
- generic triangulation primitives

## Endcap Contract

Endcap generation may remain orchestrated by loft, but topology operations used by endcaps must route through topology layer.

## Deliverables

- ownership matrix in docs
- migration of generic helpers out of loft
- reduced internal helper surface in `loft.py`

## Completion Criteria

- `loft.py` reads as a geometry orchestration module, not a topology utility module.
- topology helpers are reused by loft and other features uniformly.
