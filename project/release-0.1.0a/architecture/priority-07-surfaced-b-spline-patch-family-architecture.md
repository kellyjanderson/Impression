# Priority 07 — Surfaced B-Spline Patch Family Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document covers Priority Read item `7` from
[Low-Level Construct Gap Report](../../research/2026-04-25-low-level-construct-gap-report.md).

Related architecture:

- [B-Spline Surface And Reconstruction Architecture](b-spline-surface-and-reconstruction-architecture.md)
- [Priority 05 — Spanwise Grouping And Compatibility Architecture](priority-05-spanwise-grouping-and-compatibility-architecture.md)
- [Priority 06 — Reconstruction And Repair Intermediates Architecture](priority-06-reconstruction-and-repair-intermediates-architecture.md)

## Purpose

This branch defines the later surfaced patch-family expansion that may follow
once the lower-level B-spline, fitting, and reconstruction branches are mature.

## Core Need

The research points toward future surfaced work needing patch families beyond
the current practical surfaced set, especially for:

- approximate spanwise refit
- section-driven reconstruction
- local repair patches
- curve-network or guide-driven smooth patch construction

But it also argues that these patch families should come later than curve-level
B-spline support and the reconstruction intermediates that would justify them.

## Candidate Patch Families

The strongest candidate additions are:

- native B-spline surface patch
- curve-network-derived patch family
- reconstruction-oriented local smooth patch family
- possibly developable or guided-surface families for narrower use cases

This branch should evaluate them as distinct surfaced-native families rather
than collapsing everything into one vague “smooth patch” concept.

## Architectural Rules

This branch should enforce:

1. surfaced B-spline patches are not the first B-spline milestone
2. exact surfaced-native outcomes remain preferred where available
3. smooth approximate patch families must report their approximation posture
4. a new patch family needs a real consumer before becoming active work
5. patch-family expansion should follow, not precede, reconstruction and
   compatibility infrastructure

## Preconditions

Before this branch becomes implementation work, the project should already have:

1. first-class B-spline curve objects
2. explicit fit-policy and residual reporting
3. path or trajectory B-spline consumers
4. grouping and compatibility records for spanwise work
5. reconstruction and repair intermediates

## System Placement

```text
curve primitives and fit policy
-> grouping or reconstruction evidence
-> candidate smooth patch family
-> exact or approximate surfaced result
-> surfaced result taxonomy and diagnostics
```

## Scope Boundary

This branch should not decide that every smooth reconstruction problem needs a
B-spline surface.

Its real purpose is to define:

- when a surfaced B-spline family is justified
- what other smooth patch families may coexist beside it
- how those families stay subordinate to the app-owned surfaced model

## Delivery Guidance

Recommended implementation order:

1. identify the first real consumer
2. choose the smallest justified patch family
3. define exact-vs-approximate surfaced reporting
4. add patch-family-specific diagnostics and promotion gates

## Architectural Conclusion

Priority `7` is a later expansion branch. It matters, but only after the
project has built the lower-level curve, fit, trajectory, grouping, and repair
foundations that would make a surfaced B-spline patch family honest.
