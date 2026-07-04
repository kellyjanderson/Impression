# 0.1.0.a Feature List

## Version Theme

`0.1.0.a` should be an inference-and-curve-fitting version, centered on making
loft smarter without prematurely turning the project into a surfaced B-spline
reconstruction release.

The core idea is:

- infer more intent from dense loft evidence
- keep topology truth app-owned
- reduce brute-force station density with hidden internal structure
- make fitting and inference explainable rather than magical

## Included Feature Direction

### 1. First-Class B-Spline Curve Support

Add first-class B-spline curve constructs as low-level modeling primitives.

Expected scope:

- `BSpline2D`
- `BSpline3D`
- explicit degree and knot-vector ownership
- deterministic evaluation, tangent, and sampling behavior

This is an enabling feature, not the headline by itself.

Architecture coverage for this branch already exists in:

- [B-Spline Implementation Architecture](../architecture/b-spline-implementation-architecture.md)
- [Priority 01 — B-Spline Curve Constructs Architecture](../architecture/priority-01-b-spline-curve-constructs-architecture.md)

### 2. Explicit Fit Policy And Diagnostics

Introduce owned fitting configuration and reporting rather than hiding those
choices inside helper code.

Expected scope:

- parameterization policy
- knot-count and knot-placement policy
- fit configuration record
- residual and acceptance reports

This feature exists to make future inference durable and inspectable.

### 3. Curve Fitting From Dense Loft Evidence

Fit smooth curve explanations to dense loft evidence instead of treating station
density as the only way to communicate curvature.

Expected scope:

- dense-station curve fitting
- trajectory fitting for likely smooth vertical evolution
- diagnostics that show where fit succeeds or drifts

This should begin as offline or planner-support tooling rather than hidden
automatic mutation.

### 4. Non-User-Facing Control Stations

Introduce internal control-station structure that the planner can use without
requiring a new authored public API on day one.

Expected scope:

- hidden retained control-station classification
- hard distinction between topology stations and control stations
- planner-owned internal use
- durable diagnostics and provenance

This should remain non-user-facing in the first version slice unless a later
branch proves a public control-station authoring API is necessary.

### 5. Control-Station Inference

Reduce dense linear station sets into a more compact internal representation
without sacrificing topology truth.

Expected scope:

- topology-station retention
- hidden control-station retention
- reduction refusal when structure would be lost
- structural preservation reporting

The primary result is not “fewer points.” It is a durable reduced progression
with explainable retained structure.

### 6. Curve-Intent Inference

Infer likely smooth curve intent from dense station evidence.

Expected scope:

- station-density-over-distance signals
- loop and correspondence continuity signals
- smooth size/centroid/anisotropy change signals
- candidate curve-intent reporting

This is the bridge between dense brute-force station authoring and a more
compact smooth explanation.

### 7. Shared Trajectory Inference And Optional Guidance

Use inferred smooth travel paths where the evidence supports them, starting with
whole-loft shared trajectory interpretation before region-level or track-level
guidance.

Expected scope:

- whole-loft shared trajectory candidates
- optional explicit `Path3D` or future `BSpline3D` guidance consumption
- deterministic attachment resolution

This is a consumer of the curve-fitting stack, not a separate foundation.

### 8. Progression Model Upgrade

Upgrade progression from a mostly scalar sequencing concept into a richer
semantic object built on `Path3D`.

Expected scope:

- path-backed progression as the canonical loft travel model
- station attachment to progression rather than loose parallel progression
  arrays
- owned parameterization and transport semantics
- room for twist, scale, and inferred-trajectory metadata
- exact vs inferred progression provenance

This is an architectural improvement because it matches the real role
progression is already trying to play in loft and gives the inference work a
cleaner home than raw scalar lists.

This progression upgrade is also the intended replacement path for generic
path-driven body-construction cases. `0.1.0.a` should prefer loft enhancement
over creating separate sweep/pipe feature tracks.

### 9. Inference Diagnostics And Explainability

Make the internal decisions inspectable so reduction and fitting are trustworthy
rather than opaque.

Expected scope:

- retained vs dropped station explanation
- fit drift reporting
- topology-preservation reporting
- refusal diagnostics
- inferred-trajectory and inferred-curve evidence bundles

This feature is required for confidence, testing, and future UI review tools.

## Explicitly Excluded From 0.1.0.a Core

These are important later branches, but they should not define this version’s
center of gravity:

- surfaced B-spline patch families as a required milestone
- approximate spanwise patch refit as a core release feature
- heavy mesh-repair or surfaced reconstruction features
- public user-authored control-station APIs as the first milestone
- separate sweep/pipe/path-driven body-construction feature tracks outside loft
  enhancement

## Recommended Implementation Order

1. first-class B-spline curve support
2. fit policy and diagnostics
3. curve fitting from dense loft evidence
4. hidden control-station structure
5. control-station inference results
6. curve-intent inference
7. shared trajectory inference
8. progression model upgrade
9. explainability and diagnostics hardening

## Version Story

The clearest release story for `0.1.0.a` is:

> Loft becomes smarter about smooth shape intent. Dense station stacks can be
> interpreted, reduced, and explained through fitted curves and hidden internal
> control structure while preserving topology-aware surfaced planning.

## Out Of Scope But Adjacent

These remain strongly related and should stay visible in planning, but outside
the first version slice:

- spanwise postprocess consolidation
- planner-time span promotion
- repair-oriented reconstruction intermediates
- surfaced B-spline patch families

## Conclusion

`0.1.0.a` should focus on:

- curve fitting
- non-user-facing control stations
- inference across dense loft evidence

That is the highest-leverage path supported by both the internal and external
research sets.
