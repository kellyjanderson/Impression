# Feature Spec 07A2: Shared Whole-Loft Trajectory Confidence and Refusal Posture (v1.0)

## Overview

This specification defines how confidence, uncertainty, and refusal are
communicated for whole-loft shared trajectory candidates.

## Backlink

- [Feature Spec 07A: Shared Whole-Loft Trajectory Candidate Inference (v1.0)](feature-07a-shared-whole-loft-trajectory-candidate-inference-v1_0.md)

## Scope

This specification covers:

- confidence posture for whole-loft shared trajectory candidates
- explicit refusal posture when evidence is weak or conflicting
- reporting of weak or conflicting evidence for the shared trajectory lane

## Behavior

This leaf must define:

- how confidence is recorded for shared trajectory candidates
- how weak or conflicting evidence produces explicit refusal or uncertainty
- how candidate-generation outputs are carried into this posture layer

## Constraints

- weak evidence must not be silently promoted to accepted trajectory truth
- region-level or track-level guidance remains out of initial scope

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- confidence posture is explicit
- refusal or uncertainty posture is explicit
- weak evidence is not silently promoted
