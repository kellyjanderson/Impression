# Feature Spec 07A1: Shared Whole-Loft Trajectory Candidate Generation (v1.0)

## Overview

This specification defines how the initial shared whole-loft trajectory lane
generates candidate trajectories.

## Backlink

- [Feature Spec 07A: Shared Whole-Loft Trajectory Candidate Inference (v1.0)](feature-07a-shared-whole-loft-trajectory-candidate-inference-v1_0.md)

## Scope

This specification covers:

- derivation of whole-loft shared trajectory candidates
- relationship to fitted curve evidence
- candidate output shape for later consumers

## Behavior

This leaf must define:

- how whole-loft shared trajectory candidates are derived
- how fitted curve evidence contributes to candidate generation
- how later consumers receive trajectory candidates

## Constraints

- generation scope remains limited to whole-loft shared trajectories
- this leaf must not own confidence or refusal posture

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- whole-loft shared trajectory candidate generation is explicit
- candidate output shape is explicit
- relationship to fitted curve evidence is explicit
