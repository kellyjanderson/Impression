# B-Spline Implementation Architecture

## Status

`0.1.0.a` feature-path architecture.

This document defines the core architectural shape for adding first-class
B-spline support to Impression.

Related architecture branches:

- [B-Spline Path And Trajectory Architecture](b-spline-path-and-trajectory-architecture.md)
- [B-Spline Control-Station Inference Architecture](b-spline-control-station-inference-architecture.md)
- [B-Spline Surface And Reconstruction Architecture](b-spline-surface-and-reconstruction-architecture.md)

Related research:

- [External Control Station Inference](../../../research/2026-04-23-external-control-station-inference.md)
- [Trajectory-Guided Loft Representation](../../../research/2026-04-23-trajectory-guided-loft-representation.md)
- [External Trajectory-Guided Loft](../../../research/2026-04-23-external-trajectory-guided-loft.md)

## Why B-Spline

Impression currently has:

- `Path3D`
- line, arc, and bezier 3D segments
- polyline/spline-like helpers
- surfaced loft planning that is topology-aware rather than spline-first

What it does not yet have is a first-class B-spline representation for curve or
surface-level modeling work.

That leaves several future directions without a canonical smooth-control model:

- compact curve fitting from dense stations
- explicit trajectory guidance richer than piecewise line/arc/bezier chains
- future surfaced reconstruction and refit workflows

## Core Architectural Decision

B-spline should enter the system in two stages:

1. first-class B-spline curve support
2. later, only if justified, B-spline surface support

The first stage is the primary `0.1.0.a` focus.

The second stage remains an extension branch rather than a required first
milestone.

## Scope Boundary

The initial architecture should cover:

- representation of B-spline curves
- deterministic evaluation and sampling
- use in path and trajectory workflows
- use in fitting/inference workflows
- serialization and durable parameter ownership

The initial architecture should not assume:

- full OCC-backed surfacing
- automatic replacement of loft planning with spline fitting
- immediate NURBS-everywhere migration
- hidden B-spline behavior inside existing APIs without explicit opt-in

## System Placement

The clean architectural placement is:

```text
authoring / analysis input
-> B-spline curve object
-> deterministic parameterization + knot vector ownership
-> evaluation / sampling / fitting helpers
-> downstream consumers
   -> Path3D / trajectory workflows
   -> control-station inference
   -> future surface/reconstruction tooling
```

This keeps B-spline as a reusable modeling primitive rather than tying it to one
consumer too early.

## Required Core Objects

The first architectural layer should define a first-class curve object with at
least:

- degree
- control points
- knot vector
- open/closed state or closure policy
- evaluation
- derivative / tangent access
- deterministic sampling
- metadata / color compatibility where appropriate

Likely object families:

- `BSpline2D`
- `BSpline3D`

`BSpline3D` is the more urgent object because it serves `Path3D`,
trajectory-guided loft, and reconstruction work directly.

## Parameterization Ownership

Parameterization and knot placement should be treated as first-class owned
subproblems, not hidden implementation details.

That means the architecture should distinguish between:

- explicit authored B-spline curves
- fitted B-spline curves inferred from samples

For authored curves, the user or importing tool owns:

- control points
- degree
- knot vector

For fitted curves, the fitting subsystem owns:

- parameter assignment
- knot count / knot placement choice
- fit diagnostics and residuals

This distinction matters because fitting logic should not leak silently into the
basic authored-curve object model.

## Determinism Rules

B-spline behavior must preserve the current project posture toward determinism.

The architecture should require:

- deterministic evaluation
- deterministic sampling for identical inputs
- deterministic fitting output when fitting configuration is unchanged
- stable ordering of control points and knots
- explicit rather than implicit approximation settings

## Relationship To Existing Modeling Posture

B-spline should complement, not replace, the current surface-first and
topology-aware modeling posture.

That means:

- loft remains planner-driven and topology-owned
- B-spline curves become tools for guidance, fitting, and compact representation
- downstream tessellation remains an explicit consumer boundary

The architecture should avoid turning B-spline into a stealth alternate kernel
that bypasses the planner or the surfaced ownership model.

## Initial Delivery Order

Recommended architectural delivery order:

1. `BSpline3D` core curve representation
2. deterministic evaluation and sampling helpers
3. `Path3D` integration
4. fitting-oriented helpers for inference tooling
5. consumer-specific branches
6. only later, evaluate surfaced B-spline patch families

## Open Questions

- Should `BSpline2D` and `BSpline3D` be introduced together or stage 3D first?
- How should closure be represented:
  - repeated control points
  - explicit periodic mode
  - explicit open/closed policy object
- What minimum fitting helper belongs in the first milestone?
- How much import/export surface area should be public before a surfaced patch
  family exists?
- When, if ever, should NURBS be introduced relative to plain B-splines?

## Architectural Conclusion

`0.1.0.a` should treat B-spline primarily as:

- a first-class curve primitive
- a reusable control representation
- a dependency of future loft/path/inference/reconstruction features

It should not yet be treated as a mandatory surfaced patch family or as a new
dominant kernel.
