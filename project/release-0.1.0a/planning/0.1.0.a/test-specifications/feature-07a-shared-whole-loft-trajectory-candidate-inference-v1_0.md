# Feature Spec 07A Test: Shared Whole-Loft Trajectory Candidate Inference

## Overview

This test specification defines verification for shared whole-loft trajectory
candidate inference.

## Backlink

- [Feature Spec 07A: Shared Whole-Loft Trajectory Candidate Inference (v1.0)](../specifications/feature-07a-shared-whole-loft-trajectory-candidate-inference-v1_0.md)

## Behavior

This parent test branch must verify:

- no executable shared-trajectory candidate-generation work remains hidden in
  the parent
- no executable shared-trajectory confidence or refusal work remains hidden in
  the parent
- every final child leaf has a paired test specification

## Child Test Specifications

- [Feature Spec 07A1 Test: Shared Whole-Loft Trajectory Candidate Generation](feature-07a1-shared-whole-loft-trajectory-candidate-generation-v1_0.md)
- [Feature Spec 07A2 Test: Shared Whole-Loft Trajectory Confidence and Refusal Posture](feature-07a2-shared-whole-loft-trajectory-confidence-and-refusal-posture-v1_0.md)

## Acceptance

This test specification is complete when the child set fully covers the branch.
