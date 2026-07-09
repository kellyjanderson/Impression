# Feature 08 — Progression Model Upgrade Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document corresponds directly to Feature `8` in
[0.1.0.a Feature List](../planning/feature-list.md).

Related architecture:

- [Priority 03 — Path And Trajectory Integration Architecture](priority-03-path-and-trajectory-integration-architecture.md)
- [Feature 07 — Shared Trajectory Inference And Guidance Architecture](feature-07-shared-trajectory-inference-and-guidance-architecture.md)

## Purpose

Define the architectural correction that upgrades progression from parallel
scalar arrays into a semantic object built on `Path3D`.

## Core Need

Current loft code still treats progression mainly as ordered scalar `t` values,
while path and station placement semantics are carried elsewhere.

This feature should unify that split.

## Included Scope

- path-backed progression as canonical loft travel model
- station attachment to progression
- owned parameterization semantics
- frame transport semantics
- twist and scale law slots
- exact-vs-inferred provenance

## Core Rule

`Path3D` should remain the geometric spine.

`Progression` should become the semantic wrapper that owns:

- how loft moves along that spine
- how stations attach to it
- whether the progression is explicit or inferred

## Relationship To Other Features

This feature is the intended replacement path for generic path-driven body
construction cases.

That means:

- no separate sweep/pipe feature branch for `0.1.0.a`
- loft enhancement absorbs those cases through better progression semantics

## System Role

```text
Path3D
-> Progression
   -> parameterization
   -> station attachment
   -> transport policy
   -> twist/scale slots
   -> provenance
-> Loft
```

## Architectural Conclusion

Feature `08` is a structural correction to the loft architecture and one of the
most important coherence improvements in the `0.1.0.a` plan.
