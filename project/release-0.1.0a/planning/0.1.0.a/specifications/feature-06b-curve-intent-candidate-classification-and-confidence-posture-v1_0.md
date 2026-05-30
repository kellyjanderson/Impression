# Feature Spec 06B: Curve-Intent Candidate Classification and Confidence Posture (v1.0)

## Overview

This specification defines how curve-intent candidates are reported from dense
loft evidence.

## Backlink

- [Feature Spec 06: Curve-Intent Inference Program (v1.0)](feature-06-curve-intent-inference-program-v1_0.md)

## Scope

This specification covers:

- candidate intent reporting
- confidence posture
- refusal or indeterminate posture

## Behavior

This leaf must define:

- what a curve-intent candidate report contains
- how confidence or uncertainty is expressed
- how indeterminate or conflicting evidence is handled

## Constraints

- weak evidence must not be overstated as strong inferred intent
- refusal or indeterminate posture must remain a first-class output

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- candidate report shape is explicit
- confidence posture is explicit
- indeterminate or refusal behavior is explicit
