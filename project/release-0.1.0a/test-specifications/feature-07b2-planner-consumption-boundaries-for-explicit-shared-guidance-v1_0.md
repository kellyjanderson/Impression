# Feature Spec 07B2 Test: Planner Consumption Boundaries for Explicit Shared Guidance

## Overview

This test specification defines verification for planner consumption of explicit
shared guidance.

## Backlink

- [Feature Spec 07B2: Planner Consumption Boundaries for Explicit Shared Guidance (v1.0)](../specifications/feature-07b2-planner-consumption-boundaries-for-explicit-shared-guidance-v1_0.md)

## Automated Smoke Tests

- planner stages that consume explicit shared guidance are inspectable
- topology truth remains visible alongside guidance consumption

## Automated Acceptance Tests

- explicit shared guidance does not override topology-owned planning truth
- planner-consumption behavior remains deterministic
- guidance influence stays bounded to in-between travel semantics
