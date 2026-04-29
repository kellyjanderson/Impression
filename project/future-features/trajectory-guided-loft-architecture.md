# Trajectory-Guided Loft Architecture

## Status

Future feature.

This document preserves a loft direction that is not yet part of the active
architecture or specification tree.

## Idea

Current loft derives vertical shape evolution primarily from:

- placed stations
- correspondence between sections
- interpolation across the progression axis

That works well for many cases, but it does not give the author an explicit way
to say:

- this feature should travel through space along a curve
- this region should bow outward and then return
- this side should rise differently from that side
- this dense set of tiny stations is really trying to describe one smooth curve

This future feature introduces the idea that loft should be able to understand
curve intent explicitly or infer it from dense authored input.

## Two Complementary Lanes

This idea has two related lanes.

### 1. Explicit Trajectory Guidance

The author supplies curve intent directly.

Examples:

- one shared `Path3D` describing a symmetric or globally coherent vertical
  curve
- one path per region family
- one path per correspondence track or node family

This gives loft a way to evolve mapped features along explicit spatial
trajectories instead of only inferring their motion from section placement.

### 2. Inferred Curve Intent From Dense Facets

The author supplies many closely spaced linear stations, but the density and
resulting shape strongly imply that the true intent is curved rather than
piecewise linear.

Examples:

- a dense hourglass-like stack where radius change is communicated by both
  position and frequency of stations
- repeated small station changes that visually reconstruct a smooth bowed
  surface
- an authored faceted progression where the final realized result clearly reads
  as one smooth curve

This gives loft a way to infer that the input is communicating curve intent,
not merely many unrelated tiny linear spans.

## Why It Matters

- gives loft a more expressive way to represent vertical curvature
- reduces the need to brute-force curved evolution with many small stations
- provides a path between simple station interpolation and fully patch-authored
  surfaces
- creates a way to preserve author intent that currently gets hidden inside
  dense progression samples
- may allow fewer authored stations and fewer synthetic helper stations while
  still producing visibly curved results

## Core Distinction

Stations define sectional truth.

Correspondence defines what maps to what across progression.

Trajectory guidance would define how mapped features move through space.

That means loft could evolve from:

```text
stations -> interpolate
```

toward:

```text
stations -> correspondence -> trajectory intent -> execute surface evolution
```

## Possible Attachment Levels

The main architectural question is what the curve is attached to.

### Loft-Level Path

One shared path describes a global vertical curvature pattern for the whole
loft.

This is a good fit for:

- symmetric vessels
- globally coherent ducts
- simple organic sweeps

### Region-Level Path

Different regions of the same station may carry different curve guidance.

This is a good fit for:

- multi-lobe objects
- pillar or branch families
- asymmetric split-region shapes

### Track-Level Path

A path is attached to a node family, loop sample track, or correspondence
trajectory.

This is the richest version and the closest to explicit trajectory authoring.

This is a good fit for:

- strongly asymmetric forms
- locally expressive curvature
- high-control loft authoring

## Inference Signals

If curve intent is inferred from dense authored input, likely signals include:

- station frequency over distance
- station placement clustering near high curvature areas
- smooth continuity of cross-section drift across many small spans
- repeated local evolution that looks like one larger algorithmic curve rather
  than unrelated linear steps
- consistency of node or loop motion across a run of stations

The important idea is that authors may already be communicating curve intent
without having a first-class curve-intent tool.

Dense stations can act as a faceted description of a curve.

## Relationship To Existing Future Ideas

This idea is related to, but distinct from:

- [Control Station Inference Architecture](control-station-inference-architecture.md)
- [Spanwise Loft Consolidation Architecture](spanwise-loft-consolidation-architecture.md)

Control-station inference asks:

- which stations are really needed

Trajectory-guided loft asks:

- what curve behavior is those stations trying to communicate

Spanwise loft consolidation asks:

- how should a wider run of stations collapse into a better realized surface
  structure

These ideas fit together well:

- control-station inference reduces authored station density
- trajectory guidance preserves non-linear vertical intent
- spanwise consolidation reduces realized surface over-segmentation

## Possible User-Facing Shapes

The simplest future user-facing forms might be:

```text
Loft(..., trajectory=shared_path)
```

```text
Loft(..., region_trajectories={region_id: path3d})
```

```text
Loft(..., track_trajectories={track_id: path3d})
```

Or as preprocessing / analysis tools:

```text
infer_loft_trajectory_intent(dense_progression, dense_stations, topology)
```

```text
fit_track_trajectories(stations, correspondences)
```

The likely right staging is:

1. shared-path guidance
2. region-level guidance
3. track-level guidance
4. inference from dense faceted station sets

## Open Questions

- What is the first useful attachment level:
  - whole loft
  - region
  - track
- How should trajectory guidance interact with explicit station placement?
- Should explicit trajectory guidance constrain station origins, or only the
  evolution between them?
- What is the right inferred signal for "this dense progression is really a
  curve"?
- How should trajectory inference interact with control-station inference?
- Can trajectory fitting remain deterministic and topology-aware?
- How much of this belongs inside the planner versus a preprocessing tool?

## Why Preserve This Now

This idea emerged from real loft authoring work where curved vertical behavior
was being communicated indirectly through:

- station placement
- station frequency
- repeated small faceted transitions

The resulting shape already reads as curved, which suggests the authored input
is expressing more intent than the current loft representation can capture
directly.

Preserving this now keeps open a path toward loft that can understand not just
what sections exist, but how those sections are intended to travel through
space.
