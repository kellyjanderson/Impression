# Feature Spec 06 Test: Curve-Intent Inference Program

## Overview

This test specification defines verification for the decomposed curve-intent
inference branch.

## Backlink

- [Feature Spec 06: Curve-Intent Inference Program (v1.0)](../specifications/feature-06-curve-intent-inference-program-v1_0.md)

## Behavior

This parent test branch must verify:

- no executable descriptor-evidence work remains hidden in the parent
- no executable candidate-classification work remains hidden in the parent
- every final child leaf has a paired test specification

## Child Test Specifications

- [Feature Spec 06A Test: Descriptor and Evidence Records for Curve-Intent Inference](feature-06a-descriptor-and-evidence-records-for-curve-intent-inference-v1_0.md)
- [Feature Spec 06B Test: Curve-Intent Candidate Classification and Confidence Posture](feature-06b-curve-intent-candidate-classification-and-confidence-posture-v1_0.md)

## Acceptance

This test specification is complete when the child set fully covers the branch.
