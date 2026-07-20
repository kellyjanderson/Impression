# B-Spline Path And Trajectory Architecture

## Status

`0.1.0.a` feature-path architecture branch.

Parent architecture:

- [B-Spline Implementation Architecture](b-spline-implementation-architecture.md)

Related research:

- [Trajectory-Guided Loft Representation](../../research/2026-04-23-trajectory-guided-loft-representation.md)
- [External Trajectory-Guided Loft](../../research/2026-04-23-external-trajectory-guided-loft.md)
- [Modeling — Path3D](../../../docs/modeling/path3d.md)

## Purpose

This branch defines where B-spline should be used for path and trajectory
authoring in Impression.

It covers:

- `Path3D` integration
- explicit trajectory-guided loft inputs
- future path-aware modeling workflows

## Why This Branch Exists

`Path3D` already gives Impression a natural home for smooth 3D guidance, but it
currently exposes:

- `Line3D`
- `Arc3D`
- `Bezier3D`

That is useful, but it leaves the system without a canonical curve type for:

- longer smooth guides
- compact high-control path authoring
- deterministic fit-generated trajectory results

## Core Decision

B-spline should become a first-class path segment or path-owned curve type in
the path and trajectory layer.

The initial architectural target is:

- `BSpline3D` usable as a path segment or path element

This should be explicit in the public API rather than hidden behind automatic
conversion.

## Path3D Integration

The clean path-level model is:

```text
Path3D
-> segments:
   -> Line3D
   -> Arc3D
   -> Bezier3D
   -> BSpline3D
```

That preserves the current Path3D posture while adding a richer smooth-guide
option.

Important consequences:

- `Path3D.sample()` must sample B-splines deterministically
- path-derived station placement must preserve stable ordering
- path utilities must not reduce B-spline to polyline-only truth internally
  unless explicitly crossing a consumer boundary

## Trajectory-Guided Loft Usage

Trajectory-guided loft is the strongest early consumer of `BSpline3D`.

The recommended first attachment level remains:

- one shared whole-loft trajectory

So the first clear use case is something like:

```text
Loft(..., trajectory=BSpline3D(...))
```

or:

```text
Loft(..., trajectory=Path3D([... BSpline3D(...) ...]))
```

The architectural rule is:

- stations remain hard anchors
- B-spline trajectory influences in-between travel
- topology planning remains owned by the loft planner

## Where It Helps

This branch should benefit:

- explicit smooth loft guidance
- future sweep/pipe-like workflows
- authored guide paths that are too rich for simple beziers
- later region-level trajectory guidance if that branch is promoted

## What It Should Not Do

This branch should not:

- silently replace station placement
- bypass ambiguity handling
- turn `Path3D` into a hidden fitting system
- imply that B-spline guidance overrides topology interpretation

## Delivery Order

Recommended order inside this branch:

1. `BSpline3D` support in `Path3D`
2. deterministic path sampling and evaluation updates
3. explicit path/trajectory docs
4. first whole-loft trajectory-guidance consumer

## Open Questions

- Should `Path3D` expose direct factory helpers for B-spline creation?
- Should B-spline trajectory input be accepted directly by loft, or only via
  `Path3D`?
- How should mixed segment paths behave when some segments are B-spline and
  others are line/arc/bezier?
- Should trajectory-fitting output be normalized to `BSpline3D` or to `Path3D`
  containing B-spline segments?

## Architectural Conclusion

The first practical usage of B-spline in Impression should be at the path and
trajectory layer, with `BSpline3D` becoming the canonical high-control smooth
guide representation for future loft-aware workflows.
