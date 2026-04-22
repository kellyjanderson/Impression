# Control Station Inference Architecture

## Status

Future feature.

This document captures an architectural direction that is worth preserving, but
is not yet part of the active implementation architecture or specification
tree.

## Idea

Current loft behavior treats station-to-station evolution as piecewise linear
unless the author provides many closely spaced stations.

That means a user who wants a shape with meaningful non-linear section
evolution often approximates it by:

1. authoring many stations
2. making each linear span very small
3. relying on dense sampling rather than a richer progression model

This future feature introduces a user-facing tool that works in the opposite
direction:

1. start from a loft described in dense linear station space
2. analyze the point-to-point and loop-to-loop relationships across that dense
   sequence
3. infer a reduced set of control stations
4. return a new progression made of:
   - topology stations
   - control stations

The intent is not merely "remove stations."
The intent is to infer a more meaningful loft representation that preserves the
shape with fewer authored progression samples.

## Why It Matters

- reduces the number of stations needed to represent smooth non-linear shape
  change
- separates topology-critical authored structure from shape-driving control
  structure
- creates a user-facing optimization and authoring tool rather than requiring
  manual brute-force station densification
- opens a middle layer between explicit section stations and low-level surface
  patch construction
- gives loft a path toward higher-order evolution without collapsing into a
  generic spline or mesh-morph system

## Architectural Placement

This concept sits between:

- explicit placed topology stations
- downstream executor surface realization

It also sits adjacent to, but distinct from:

- user-authored control sections
- low-level surface patch families

The intended layering is:

```text
dense linear loft stations
-> control-station inference tool
-> reduced progression
   -> topology stations
   -> control stations
-> loft planner
-> loft executor
-> surface body
```

Or, if exposed as a user-facing authoring flow:

```text
user-authored dense station sequence
-> infer_control_stations(...)
-> review / pin / reject proposed retained stations
-> accepted reduced progression
-> canonical loft authoring input
```

## Core Distinction

The returned progression should distinguish between two kinds of retained
stations.

### Topology Stations

Topology stations are required structural anchors.

They exist where:

- region structure changes
- hole birth or hole death matters
- correspondence constraints must remain explicit
- topology interpretation would become ambiguous without that station

These are not optional compression artifacts.
They are structural truth.

### Control Stations

Control stations are shape-driving anchors that preserve non-linear evolution
without requiring every intermediate dense station to remain authored.

They exist where:

- shape curvature or rate-of-change meaningfully changes
- a span cannot be represented well enough by simple direct interpolation
- the user or tool wants a sparse but expressive control set

These are not the same thing as topology stations, even though both live on the
same progression axis.

## What The Tool Infers

The tool should not be thought of as only fitting raw point trajectories.

A stronger architectural direction is to infer higher-level relationship
structure such as:

- loop evolution
- correspondence-field evolution
- region or boundary motion patterns
- span-local non-linear progression behavior

Points are likely a downstream realization detail rather than the primary
architectural truth.

This matters because the goal is not to turn loft into a generic point-cloud
curve fitter.
The goal is to preserve topology-aware shape evolution.

## Proposed User-Facing Contract

At a high level, the user-facing tool contract could look like:

```text
infer_control_stations(dense_progression, dense_stations, topology)
-> reduced_progression
   -> retained topology stations
   -> inferred control stations
   -> fit metadata
   -> error metrics
```

Important outputs would likely include:

- reduced progression values
- retained station classification
- interval fit metadata
- approximation or drift metrics relative to the dense input
- enough diagnostic output for the user to inspect or override the proposed
  reduction

## Relationship To Existing Loft Architecture

This idea is compatible with the current loft architectural bias that loft is:

- topology-aware
- deterministic
- planner-driven

It is not the same as:

- making loft a spline system
- replacing the planner with curve fitting
- turning the executor into a heuristic approximator

The cleanest interpretation is:

- control-station inference is a preprocessing or authoring-stage tool
- the result becomes a better loft input
- the existing planner/executor architecture remains intact downstream

That keeps the current core architectural statement valid:

> Loft constructs surfaces by evolving topology over progression.

This feature proposes a richer way to author or infer that progression, not a
replacement for the planner/executor split.

## Open Questions

- What is the primary inferred truth:
  - point trajectories
  - loop trajectories
  - correspondence-field evolution
  - interval-local span operators
- What makes a station topology-critical versus shape-control-only?
- How should the tool expose error metrics and acceptance thresholds?
- Should control stations be editable first-class authored objects after
  inference?
- How should user-pinned stations interact with inferred control reduction?
- Does the inferred progression remain fully deterministic under repeated runs?
- How should this concept relate to any future control-section authoring model?
- Should the first implementation be offline simplification only, or also offer
  interactive refinement?

## Why Preserve This Now

This idea emerged from a real limitation in the current loft workflow:

- dense linear stations can brute-force shape
- but they are not always the best authored representation of that shape

Capturing this now preserves a possible path toward:

- sparser loft authoring
- more expressive progression modeling
- better separation of structural and shape-driving stations

without forcing immediate implementation while the core loft system is still
stabilizing.

