# Feature Spec 07A2 Test: Shared Whole-Loft Trajectory Confidence and Refusal Posture

## Overview

This test specification defines verification for confidence, uncertainty, and
refusal behavior around shared whole-loft trajectory candidates.

## Backlink

- [Feature Spec 07A2: Shared Whole-Loft Trajectory Confidence and Refusal Posture (v1.0)](../specifications/feature-07a2-shared-whole-loft-trajectory-confidence-and-refusal-posture-v1_0.md)

## Automated Smoke Tests

- weak or conflicting evidence produces explicit confidence or refusal posture
- confidence posture is attached to generated shared-trajectory candidates

## Automated Acceptance Tests

- weak evidence is not silently promoted to accepted trajectory truth
- refusal or uncertainty remain first-class outputs
- scope remains limited to whole-loft shared trajectories
