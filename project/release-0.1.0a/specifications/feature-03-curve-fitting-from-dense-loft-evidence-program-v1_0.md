# Feature Spec 03: Curve Fitting From Dense Loft Evidence Program (v1.0)

## Overview

This specification defines the `0.1.0.a` feature branch for fitting smooth
curve explanations to dense loft evidence.

## Backlink

- [Feature 03 — Curve Fitting From Dense Loft Evidence Architecture](../architecture/feature-03-curve-fitting-from-dense-loft-evidence-architecture.md)

## Scope

This specification covers:

- dense-evidence descriptor preparation
- candidate curve-fit generation and comparison

## Behavior

This branch must define:

- the leaf that prepares dense loft evidence for fit-backed analysis
- the leaf that generates and evaluates station-derived fitted curves
- the leaf that generates and evaluates shared-trajectory fitted curves

## Constraints

- this branch must stay diagnostic and explanatory first
- it must not silently mutate authored loft truth on its own

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Feature Spec 03A: Dense Loft Evidence Descriptor Preparation for Curve Fitting](feature-03a-dense-loft-evidence-descriptor-preparation-for-curve-fitting-v1_0.md)
- [Feature Spec 03B: Station-Derived Candidate Curve-Fit Generation, Comparison, and Refusal Posture](feature-03b-station-derived-candidate-curve-fit-generation-comparison-and-refusal-posture-v1_0.md)
- [Feature Spec 03C: Shared-Trajectory Candidate Curve-Fit Generation, Comparison, and Refusal Posture](feature-03c-shared-trajectory-candidate-curve-fit-generation-comparison-and-refusal-posture-v1_0.md)

## Acceptance

This specification is complete when:

- dense-evidence preparation and candidate fitting are split into executable
  leaves
- the branch stays explanatory rather than silently destructive
