# Feature 04 — Non-User-Facing Control Stations Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document corresponds directly to Feature `4` in
[0.1.0.a Feature List](../planning/feature-list.md).

Related architecture:

- [B-Spline Control-Station Inference Architecture](b-spline-control-station-inference-architecture.md)
- [Feature 05 — Control-Station Inference Architecture](feature-05-control-station-inference-architecture.md)

## Purpose

Define the internal-only control-station layer that lets loft retain shape
control without forcing a new authored public API in the first milestone.

## Included Scope

- hidden control-station classification
- planner-owned control-station consumption
- distinction between topology stations and control stations
- retained provenance and diagnostic metadata

## Core Rule

Control stations should first exist as internal planner or inference structures,
not as mandatory new user-facing authored objects.

That keeps `0.1.0.a` focused on better inference and better loft behavior rather
than on immediate public authoring redesign.

## System Role

```text
dense authored stations
-> inference and fit analysis
-> retained topology stations
-> hidden internal control stations
-> planner consumption
```

## Excluded Scope

- public user-authored control-station API
- surfaced patch-family work
- non-loft modeling consumers

## Architectural Conclusion

Feature `04` is the safe middle layer between brute-force dense station authoring
and a more explicit public control-station model that may or may not come later.
