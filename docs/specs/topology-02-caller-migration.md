# Topology Spec 02: Caller Migration

## Goal

Move feature modules off scattered private planar helpers and onto `topology.py`.

## Target Callers

- `loft.py`
- `extrude.py`
- `ops.py`
- `text.py`
- `morph.py`

## Migration Rules

1. Prefer `topology.py` names over `legacy_planar_adapter` private names.
2. Keep compatibility wrappers while migration is in progress.
3. Do not mix duplicated point-in-polygon/winding implementations per module.
4. Keep backend plumbing (`earcut`, `clipper`, `manifold`) behind topology-facing functions where possible.

## Deliverables

- Each caller imports reusable topology helpers.
- Local duplicate helpers are removed or reduced to thin wrappers.
- Behavior parity tests remain green.

## Completion Criteria

- No caller owns bespoke loop winding/resampling/containment code unless feature-specific.
- `legacy_planar_adapter.py` is no longer a functional dependency for active codepaths (adapter-only).
