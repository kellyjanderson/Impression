# Feature Spec 06: Curve-Intent Inference Program (v1.0)

## Overview

This specification defines the `0.1.0.a` branch for inferring likely smooth
curve intent from dense loft evidence.

## Backlink

- [Feature 06 — Curve-Intent Inference Architecture](../architecture/feature-06-curve-intent-inference-architecture.md)

## Scope

This specification covers:

- descriptor-level evidence
- intent candidate classification

## Behavior

This branch must define:

- the leaf that owns descriptor and evidence records
- the leaf that owns intent candidate reporting and posture

## Constraints

- the feature must infer likely intent from durable descriptor evidence
- it must stay distinct from pure geometric curve fitting

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Feature Spec 06A: Descriptor and Evidence Records for Curve-Intent Inference](feature-06a-descriptor-and-evidence-records-for-curve-intent-inference-v1_0.md)
- [Feature Spec 06B: Curve-Intent Candidate Classification and Confidence Posture](feature-06b-curve-intent-candidate-classification-and-confidence-posture-v1_0.md)

## Acceptance

This specification is complete when:

- descriptor evidence and intent candidate reporting are split into executable
  leaves
- the feature remains semantically distinct from pure fit generation
