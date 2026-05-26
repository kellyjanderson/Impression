# Feature Spec 04B Test: Planner Consumption Boundary for Hidden Control Stations

## Overview

This test specification defines verification for planner consumption boundaries
around hidden control stations.

## Backlink

- [Feature Spec 04B: Planner Consumption Boundary for Hidden Control Stations (v1.0)](../specifications/feature-04b-planner-consumption-boundary-for-hidden-control-stations-v1_0.md)

## Automated Smoke Tests

- planner stages that consume hidden control stations are explicit and
  inspectable
- topology-owned truth remains visible alongside hidden control consumption

## Automated Acceptance Tests

- hidden control stations do not override topology truth
- planner-consumption boundaries remain explicit and deterministic
- public API posture remains non-user-facing
