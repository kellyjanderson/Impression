# Priority 03 — Path And Trajectory Integration Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document covers Priority Read item `3` from
[Low-Level Construct Gap Report](../../../research/2026-04-25-low-level-construct-gap-report.md).

Related architecture:

- [B-Spline Path And Trajectory Architecture](b-spline-path-and-trajectory-architecture.md)
- [Priority 01 — B-Spline Curve Constructs Architecture](priority-01-b-spline-curve-constructs-architecture.md)
- [Priority 02 — Parameterization, Knot, And Fit Policy Architecture](priority-02-parameterization-knot-and-fit-policy-architecture.md)

## Purpose

This branch defines how first-class B-spline curves become real consumers in the
existing path and trajectory layer, rather than remaining isolated primitives.

## Core Need

The research argues that the cleanest first consumer of B-spline support is the
path layer, because Impression already has:

- `Path3D`
- line, arc, and bezier segments

But it lacks:

- a high-control smooth segment type
- a progression object that owns path semantics directly
- explicit trajectory attachment records
- deterministic mixed-segment path behavior that includes B-spline

## Branch Split

This priority item actually has two tightly related concerns:

1. `Path3D` segment integration
2. progression-as-semantic-wrapper integration
3. trajectory attachment integration

All three belong together because a path is not enough on its own; the system
also needs:

- a durable semantic owner for path-backed travel along loft
- durable answers to what the path is attached to

## Path-Level Additions

The branch should add a B-spline-capable path layer with:

- `BSpline3D` as a path segment or path-owned curve element
- deterministic mixed-segment sampling rules
- path normalization rules for B-spline-containing paths
- explicit closure handling for mixed segment paths

The path layer should preserve authored curve truth until an explicit
consumer-boundary sampling operation is requested.

## Progression-Level Additions

The current loft code still treats progression primarily as scalar `t` values
running in parallel with station placement.

This branch should instead move toward:

- `Path3D` as the geometric spine
- `Progression` as the semantic wrapper around that spine

That progression object should own at least:

- underlying path or spine reference
- parameter domain
- station attachment semantics
- transport or frame policy
- twist and scale law slots
- inferred vs explicit provenance

The architectural rule is:

- stations attach to progression
- progression owns travel semantics
- `Path3D` remains the geometric carrier, not the whole semantic contract

## Trajectory Attachment Additions

The branch should define attachment records for:

- whole-loft shared trajectory
- region-level trajectory
- track-level trajectory

Likely minimal records:

- `SharedTrajectoryInput`
- `RegionTrajectoryAttachment`
- `TrackTrajectoryAttachment`
- `TrajectoryAttachmentResolution`

These records should own:

- target identifier
- attachment level
- sampling or evaluation intent
- deterministic resolution order

## Architectural Rules

This branch should preserve:

1. stations remain hard structural anchors
2. progression owns path-backed travel semantics rather than loose scalar arrays
3. trajectory guidance influences in-between travel, not topology truth
4. ambiguity handling remains planner-owned
5. path or trajectory input must be explicit, never guessed from station shape
6. attachment identity must be durable enough for diagnostics and replay

## Scope Boundary

This branch should not define:

- control-station result classification
- surfaced B-spline patches
- section-derived reconstruction

It is the usage bridge between the primitive and later loft-aware consumers.

## Delivery Guidance

Recommended implementation order:

1. `Path3D` support for `BSpline3D`
2. path-backed progression object or equivalent semantic wrapper
3. mixed-segment deterministic sampling rules
4. shared whole-loft trajectory attachment
5. later region-level and track-level attachments

## Architectural Conclusion

Priority `3` is where B-spline becomes operational in the modeling layer.

It should first appear as:

- a real `Path3D`-compatible smooth segment
- a progression model that wraps path semantics explicitly
- an explicit trajectory attachment contract

That gives the project a clean first consumer without prematurely expanding into
surface refit work.
