> Status: Deprecated historical spec.
> Active work now lives in the surface-first specification tree and the current
> implementation. Retained for project history only.

# Topology Spec 06: ops Boundary and Domain Split

## Goal

Stop `ops.py` from mixing unrelated responsibilities (2D topology ops and 3D mesh ops) without an ownership model.

## Current State

`ops.py` currently includes:

- 2D profile/path operations (`offset`, 2D `hull`)
- 3D mesh hull
- manifold conversion adapters

## Target Direction

### Short term

- Keep `ops.py` public API stable.
- Route all loop/containment/classification logic through `topology.py`.

### Medium term

- Split implementation internally:
  - planar operations module (topology-backed)
  - mesh operations module (manifold-backed)
- `ops.py` becomes façade/re-export layer.

## Rules

1. 2D loop math must not be reimplemented in `ops.py`.
2. Backend conversion code stays isolated from topology algorithms.
3. Public function names remain stable until explicit API revision.

## Deliverables

- internal ownership split plan
- topology dependency for planar helpers
- façade contract for compatibility

## Completion Criteria

- `ops.py` is thin and non-duplicative.
- planar-topology and mesh-backend logic are separately testable.
