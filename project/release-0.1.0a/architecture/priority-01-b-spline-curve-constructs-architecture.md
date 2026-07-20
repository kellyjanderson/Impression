# Priority 01 — B-Spline Curve Constructs Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document covers Priority Read item `1` from
[Low-Level Construct Gap Report](../../research/2026-04-25-low-level-construct-gap-report.md).

Related architecture:

- [B-Spline Implementation Architecture](b-spline-implementation-architecture.md)
- [Priority 02 — Parameterization, Knot, And Fit Policy Architecture](priority-02-parameterization-knot-and-fit-policy-architecture.md)
- [Priority 03 — Path And Trajectory Integration Architecture](priority-03-path-and-trajectory-integration-architecture.md)

## Purpose

This branch defines the first-class B-spline curve primitives that Impression
needs before higher-level fitting, trajectory, or surfaced refit work can be
implemented honestly.

## Core Need

The current repo has first-class curve-like constructs for:

- line
- arc
- bezier

But the research repeatedly points toward a missing compact smooth-control
primitive that owns:

- degree
- control points
- knot vector
- periodic or closed behavior
- deterministic evaluation and derivatives

That missing primitive should be filled explicitly rather than hidden behind
sampling helpers or importer-only shims.

## Primary Objects

The initial branch should define:

- `BSpline2D`
- `BSpline3D`

With owned fields for at least:

- control point sequence
- degree
- knot vector
- closure policy
- parameter-domain bounds
- optional metadata attachment compatible with existing modeling posture

`BSpline3D` is the first practical consumer-facing target.

`BSpline2D` should be introduced either alongside it or as an immediate follow
on, because fitting and section work will quickly want the same primitive in 2D
space.

## Behavioral Contract

The curve objects should own:

- point evaluation
- first derivative evaluation
- tangent extraction
- deterministic sampling over a requested parameter domain
- explicit closure interpretation

The curve objects should not own:

- fitting policy
- knot-placement heuristics
- trajectory attachment semantics
- surfaced patch behavior

Those belong to later architecture branches.

## Representation Rules

The branch should establish these rules:

1. authored B-spline truth is explicit
2. knot vectors are durable user-visible data, not hidden implementation detail
3. closure must be expressed by policy, not guessed from repeated end points
4. evaluation order and sampling order must remain stable for identical inputs
5. tessellation remains a consumer-boundary operation

## System Placement

```text
authoring or imported curve truth
-> BSpline2D / BSpline3D
-> deterministic evaluation and sampling
-> downstream consumer
   -> Path3D
   -> fitting diagnostics
   -> future section or reconstruction workflows
```

## Scope Boundary

This branch is only about the primitive itself.

It should not silently pull in:

- fitted-curve policy
- path integration
- trajectory attachments
- surfaced B-spline patches

Those remain separate architecture branches so the primitive can stay small and
reusable.

## Delivery Guidance

Recommended implementation order:

1. define the core object shape
2. implement deterministic evaluation and sampling
3. add derivative and tangent access
4. add serialization-friendly durable ownership
5. only then wire consumers onto the primitive

## Architectural Conclusion

Priority `1` is the project’s first B-spline milestone:

- first-class curve primitives
- durable owned spline parameters
- deterministic evaluation behavior

Everything else in the low-level gap report depends on this branch existing
first.
