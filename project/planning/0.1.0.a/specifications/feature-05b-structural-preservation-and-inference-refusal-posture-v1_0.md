# Feature Spec 05B: Structural Preservation and Inference Refusal Posture (v1.0)

## Overview

This specification defines how control-station inference protects topology truth
and refuses unsafe reductions.

## Backlink

- [Feature Spec 05: Control-Station Inference Program (v1.0)](feature-05-control-station-inference-program-v1_0.md)

## Scope

This specification covers:

- structural preservation reporting
- refusal triggers
- relation between residual diagnostics and refusal

## Behavior

This leaf must define:

- when a reduction is refused
- how structural preservation is evaluated
- how refusal causes are carried durably

## Constraints

- topology-critical structure must not be dropped silently
- refusal must remain a first-class valid outcome

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- structural preservation posture is explicit
- refusal triggers are explicit
- refusal reporting is explicit
