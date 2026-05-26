# Feature Spec 03 Test: Curve Fitting From Dense Loft Evidence Program

## Overview

This test specification defines verification for the decomposed dense-evidence
curve-fitting branch.

## Backlink

- [Feature Spec 03: Curve Fitting From Dense Loft Evidence Program (v1.0)](../specifications/feature-03-curve-fitting-from-dense-loft-evidence-program-v1_0.md)

## Behavior

This parent test branch must verify:

- no executable descriptor-preparation work remains hidden in the parent
- no executable station-derived fit work remains hidden in the parent
- no executable shared-trajectory fit work remains hidden in the parent
- every final child leaf has a paired test specification

## Child Test Specifications

- [Feature Spec 03A Test: Dense Loft Evidence Descriptor Preparation for Curve Fitting](feature-03a-dense-loft-evidence-descriptor-preparation-for-curve-fitting-v1_0.md)
- [Feature Spec 03B Test: Station-Derived Candidate Curve-Fit Generation, Comparison, and Refusal Posture](feature-03b-station-derived-candidate-curve-fit-generation-comparison-and-refusal-posture-v1_0.md)
- [Feature Spec 03C Test: Shared-Trajectory Candidate Curve-Fit Generation, Comparison, and Refusal Posture](feature-03c-shared-trajectory-candidate-curve-fit-generation-comparison-and-refusal-posture-v1_0.md)

## Acceptance

This test specification is complete when the child set fully covers the branch.
