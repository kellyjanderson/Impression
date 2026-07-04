# Feature Spec 07A: Shared Whole-Loft Trajectory Candidate Inference (v1.0)

## Overview

This specification defines the whole-loft shared trajectory inference lane.

## Backlink

- [Feature Spec 07: Shared Trajectory Inference and Guidance Program (v1.0)](feature-07-shared-trajectory-inference-and-guidance-program-v1_0.md)

## Scope

This specification covers:

- whole-loft trajectory candidate generation
- supporting confidence or refusal posture

## Behavior

This leaf must define:

- the child leaf that owns whole-loft shared trajectory candidate generation
- the child leaf that owns confidence and refusal posture for those candidates

## Constraints

- whole-loft shared trajectory is the initial scope
- region-level or track-level guidance remains out of initial scope

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Feature Spec 07A1: Shared Whole-Loft Trajectory Candidate Generation](feature-07a1-shared-whole-loft-trajectory-candidate-generation-v1_0.md)
- [Feature Spec 07A2: Shared Whole-Loft Trajectory Confidence and Refusal Posture](feature-07a2-shared-whole-loft-trajectory-confidence-and-refusal-posture-v1_0.md)

## Acceptance

This specification is complete when:

- whole-loft shared trajectory candidate generation is explicit
- confidence or refusal posture is explicit
- scope remains limited to the initial shared-trajectory lane
