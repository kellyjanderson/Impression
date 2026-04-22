> Status: Deprecated historical spec.
> Active work now lives in the surface-first specification tree and the current
> implementation. Retained for project history only.

# Topology Spec 05: drawing2d Boundary Contract

## Goal

Keep `drawing2d` as the authored-geometry API while removing kernel-topology ownership from it.

## drawing2d Owns

- authored primitives and segments:
  - `Line2D`, `Arc2D`, `Bezier2D`
  - `Path2D`
- user-facing constructors and convenience factories:
  - `make_rect`, `make_circle`, `make_polygon`, `make_polyline`, `make_ngon`

## drawing2d Does Not Own

- region/section topology classification
- winding normalization policies beyond local helper use
- triangulation kernels
- correspondence/matching logic

## Re-export Rule

`drawing2d` may re-export topology-native constructs for ergonomics, but ownership remains in `topology.py`.

## Deliverables

- Doc-level ownership matrix.
- No new topology algorithms added in `drawing2d`.
- Existing topology logic routed through topology helpers.

## Completion Criteria

- `drawing2d` is clearly discoverable as authored geometry surface.
- topology algorithms are discoverable in one place (`topology.py`).
