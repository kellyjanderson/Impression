# Feature 07 — Shared Trajectory Inference And Guidance Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document corresponds directly to Feature `7` in
[0.1.0.a Feature List](../planning/feature-list.md).

Related architecture:

- [B-Spline Path And Trajectory Architecture](b-spline-path-and-trajectory-architecture.md)
- [Priority 03 — Path And Trajectory Integration Architecture](priority-03-path-and-trajectory-integration-architecture.md)
- [Feature 08 — Progression Model Upgrade Architecture](feature-08-progression-model-upgrade-architecture.md)

## Purpose

Define the feature branch that infers or consumes shared loft travel paths
without splitting loft into a separate sweep/pipe product line.

## Included Scope

- whole-loft shared trajectory candidates
- optional explicit trajectory guidance consumption
- deterministic attachment resolution
- compatibility with later region or track trajectory expansion

## Core Rule

This branch should start with shared whole-loft trajectory guidance first.

Region-level and track-level guidance remain later extensions unless the first
whole-loft lane proves insufficient.

## Relationship To Loft

This feature is a loft enhancement branch.

It should:

- improve how loft understands in-between travel
- preserve topology-aware planner ownership
- avoid creating a separate sweep/pipe modeling family for `0.1.0.a`

## Output Contract

The feature should yield:

- inferred or explicit shared trajectory truth
- durable attachment metadata
- supporting fit and confidence diagnostics

## Architectural Conclusion

Feature `07` is where path-backed behavior becomes explicit in loft while still
staying subordinate to the main loft architecture.
