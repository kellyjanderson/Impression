# Feature Spec 07 Test: Shared Trajectory Inference and Guidance Program

## Overview

This test specification defines verification for the decomposed shared
trajectory branch.

## Backlink

- [Feature Spec 07: Shared Trajectory Inference and Guidance Program (v1.0)](../specifications/feature-07-shared-trajectory-inference-and-guidance-program-v1_0.md)

## Behavior

This parent test branch must verify:

- no executable shared-trajectory inference work remains hidden in the parent
- no executable explicit-guidance work remains hidden in the parent
- every final child leaf has a paired test specification

## Child Test Specifications

- [Feature Spec 07A Test: Shared Whole-Loft Trajectory Candidate Inference](feature-07a-shared-whole-loft-trajectory-candidate-inference-v1_0.md)
- [Feature Spec 07B Test: Explicit Shared Guidance Attachment and Consumption Rules](feature-07b-explicit-shared-guidance-attachment-and-consumption-rules-v1_0.md)

## Acceptance

This test specification is complete when the child set fully covers the branch.
