# Spanwise Loft Consolidation Architecture

## Status

Future feature.

This document preserves a higher-level loft idea that is not yet part of the
active architecture or specification tree.

## Idea

Current loft primarily reasons locally from one station to the next.

That local station-to-station view is often correct, but it can also produce a
surface decomposition that is finer than necessary when a shape was authored
with many stations.

From a wider spanwise view, a long run of locally valid station intervals may
actually describe one larger coherent surface phenomenon that could be
represented as:

- one larger patch
- one higher-order span
- or one consolidated run of fewer surface elements

The core idea is therefore:

> allow loft or a loft-adjacent tool to reason over larger station spans than
> the immediate interval and consolidate over-segmented local loft structure.

## Why It Matters

- dense station authoring can produce more local intervals and more local patch
  boundaries than the final shape really needs
- local interval correctness does not guarantee good higher-level surface
  decomposition
- a wider-span view could reduce realized patch count while preserving the same
  visible shape
- this creates a second compression direction alongside control-station
  inference:
  - control-station inference reduces authored progression complexity
  - spanwise loft consolidation reduces realized surface complexity

## Core Architectural Question

Should loft optimize only for:

- correct local transitions

or also for:

- best larger-span surface decomposition across a run of transitions

This feature explores the second possibility without abandoning the current
planner/executor foundation.

## Three Architectural Branches

This future idea naturally splits into three branches:

### 1. Inline Loft Enhancement

The planner recognizes longer span structure before execution and emits a
larger-span realization directly.

Architecture note:

- [Spanwise Loft Inline Enhancement Architecture](spanwise-loft-inline-enhancement-architecture.md)

### 2. Post-Loft Optimization Tool

Loft first builds its normal local result, then a postprocessing tool analyzes
adjacent patches and consolidates compatible spans into a simpler surfacebody
representation.

Architecture note:

- [Spanwise Loft Postprocessing Optimization Architecture](spanwise-loft-postprocessing-optimization-architecture.md)

### 3. Repair Tool

A repair-oriented tool uses the same larger-span consolidation idea when
recovering or simplifying damaged or over-segmented loft-like geometry.

Architecture note:

- [Spanwise Loft Repair Tool Architecture](spanwise-loft-repair-tool-architecture.md)

## Relationship To Control Station Inference

This idea is related to, but distinct from:

- [Control Station Inference Architecture](control-station-inference-architecture.md)

Control-station inference focuses on reducing and reclassifying the authored
progression.

Spanwise loft consolidation focuses on reducing and improving the realized
surface decomposition over a longer run of stations.

Both ideas move away from brute-force density, but they do so at different
layers of the loft system.

## Open Questions

- What qualifies a run of local intervals as one larger coherent span?
- Should consolidation target:
  - fewer patches
  - different patch families
  - better seam placement
  - or all three?
- How should consolidation respect topology events that are locally real but
  globally over-segmenting?
- Should the first implementation live inside the planner or outside it?
- How should the tool report approximation or loss when consolidation is not
  exact?

