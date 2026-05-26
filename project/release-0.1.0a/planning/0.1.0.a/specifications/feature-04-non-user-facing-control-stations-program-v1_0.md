# Feature Spec 04: Non-User-Facing Control Stations Program (v1.0)

## Overview

This specification defines the `0.1.0.a` branch for internal-only control
station structure.

## Backlink

- [Feature 04 — Non-User-Facing Control Stations Architecture](../architecture/feature-04-non-user-facing-control-stations-architecture.md)

## Scope

This specification covers:

- hidden control-station structure
- planner-owned internal control-station consumption

## Behavior

This branch must define:

- the leaf that owns internal control-station representation and provenance
- the leaf that owns planner-consumption boundaries for hidden control stations

## Constraints

- this branch must not require a public authored control-station API
- topology and control classifications must remain distinct

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Feature Spec 04A: Internal Control-Station Representation and Provenance](feature-04a-internal-control-station-representation-and-provenance-v1_0.md)
- [Feature Spec 04B: Planner Consumption Boundary for Hidden Control Stations](feature-04b-planner-consumption-boundary-for-hidden-control-stations-v1_0.md)

## Acceptance

This specification is complete when:

- internal representation and planner-consumption boundaries are explicit
- the branch preserves the non-user-facing posture
