# B-Spline Control-Station Inference Architecture

## Status

`0.1.0.a` feature-path architecture branch.

Parent architecture:

- [B-Spline Implementation Architecture](b-spline-implementation-architecture.md)

Related research:

- [Control Station Inference Semantics](../../research/2026-04-23-control-station-inference-semantics.md)
- [Control Station Inference Workflow](../../research/2026-04-23-control-station-inference-workflow.md)
- [External Control Station Inference](../../research/2026-04-23-external-control-station-inference.md)
- [External Curve Intent From Dense Stations](../../research/2026-04-23-external-curve-intent-from-dense-stations.md)

## Purpose

This branch defines where and how B-spline should be used in future
control-station inference work.

It covers:

- compact representation of dense station evolution
- fitted smooth explanations for dense progression samples
- the relationship between retained stations and fitted curves

## Why This Branch Exists

The control-station inference research already points toward a compact fitted
representation problem, not just a decimation problem.

The strongest external cues are:

- parameterization matters
- knot placement matters
- iterative fitting matters

That makes B-spline a natural architectural tool for this branch, because it
offers:

- compact smooth representation
- explicit control allocation
- a direct place for parameterization and knot decisions to live

## Core Decision

B-spline should be used in control-station inference as a fitting and reduction
tool, not as the final authored truth that replaces topology-aware stations.

The architectural model is:

```text
dense topology-aware stations
-> region / loop / correspondence analysis
-> candidate reduced progression
-> B-spline-backed smooth fit diagnostics
-> retained station classification
   -> topology stations
   -> control stations
```

The retained station set remains the primary result.

The B-spline fit is the compact explanatory / diagnostic model that helps decide
which stations are needed and where reduced control should remain.

## What B-Spline Owns In This Branch

In this branch, B-spline should own:

- smooth-fit representation of dense evolution
- parameterization choices
- knot-count / knot-placement choices
- residual diagnostics against dense input

It should not own:

- topology interpretation
- ambiguity handling
- final planner truth

That remains with the loft planner and the retained topology-aware station set.

## Relationship To Topology Stations

Topology stations are hard structural anchors.

They should not be removable just because a B-spline fit looks smooth enough.

The architectural rule is:

- B-spline fit may justify reducing shape-control density
- B-spline fit may not erase topology-critical planner facts

That keeps the feature aligned with the control-station inference research:

- loop / region evolution first
- point-level or curve-level fitting second

## Relationship To Control Stations

Control stations are the likely places where the B-spline fit needs local
control to stay accurate enough.

That means this branch should likely use B-spline diagnostics to answer
questions such as:

- where does curvature behavior change enough to need explicit retention
- where does a reduced smooth model drift too far from the dense input
- where should user pins stay because they encode intentional local control

## Delivery Order

Recommended order inside this branch:

1. use B-spline only as an offline fitting/diagnostic representation
2. keep retained stations as the explicit result
3. expose parameterization / knot decisions in diagnostics
4. later evaluate whether accepted inferred control stations should preserve
   attached B-spline metadata

## Open Questions

- Should one B-spline be fit per loop family, per track family, or per other
  descriptor band?
- How should knot-placement diagnostics be exposed to the user?
- Should fit residuals become part of the control-station acceptance contract?
- How much of the fitted B-spline representation should be serialized with the
  reduced progression result?

## Architectural Conclusion

B-spline belongs in control-station inference as the compact smooth fitting tool
behind station reduction decisions, not as a replacement for topology-aware loft
inputs.
