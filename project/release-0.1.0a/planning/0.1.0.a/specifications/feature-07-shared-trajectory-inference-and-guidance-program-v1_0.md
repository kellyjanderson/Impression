# Feature Spec 07: Shared Trajectory Inference and Guidance Program (v1.0)

## Overview

This specification defines the `0.1.0.a` branch for shared trajectory
inference and optional explicit guidance consumption inside loft.

## Backlink

- [Feature 07 — Shared Trajectory Inference And Guidance Architecture](../architecture/feature-07-shared-trajectory-inference-and-guidance-architecture.md)

## Scope

This specification covers:

- whole-loft shared trajectory inference
- explicit shared guidance attachment and consumption

## Behavior

This branch must define:

- the leaf that owns shared trajectory candidate inference
- the leaf that owns explicit guidance attachment and consumption rules

## Constraints

- this branch must remain a loft enhancement path
- it must not become a separate sweep/pipe feature line

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Feature Spec 07A: Shared Whole-Loft Trajectory Candidate Inference](feature-07a-shared-whole-loft-trajectory-candidate-inference-v1_0.md)
- [Feature Spec 07B: Explicit Shared Guidance Attachment and Consumption Rules](feature-07b-explicit-shared-guidance-attachment-and-consumption-rules-v1_0.md)

## Acceptance

This specification is complete when:

- inference and explicit guidance behavior are split into final leaves
- the branch remains clearly inside loft enhancement
