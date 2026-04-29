# Testing Spec 09 Test: Reference Artifact Shared Harness and Lifecycle Program

## Overview

This test specification defines verification for the decomposed top-level
reference-artifact tooling branch.

## Backlink

- [Testing Spec 09: Reference Artifact Shared Harness and Lifecycle Program (v1.0)](../specifications/testing-09-reference-artifact-shared-harness-and-lifecycle-program-v1_0.md)

## Scope

This test specification covers:

- child-leaf completeness for reference-artifact lifecycle and grouped
  completeness rules
- paired verification coverage for final child leaves
- the boundary between top-level tooling and feature consumption

## Behavior

This parent test branch must verify:

- no executable reference-artifact lifecycle work remains hidden in the parent
- no executable grouped completeness work remains hidden in the parent
- every final child leaf has a paired test specification

## Refinement Status

Decomposed with the parent specification.

This parent test branch does not represent one executable test lane directly.

## Child Test Specifications

- [Testing Spec 10 Test: Reference Artifact Baseline Lifecycle and Invalidation Contract](testing-10-reference-artifact-baseline-lifecycle-and-invalidation-contract-v1_0.md)
- [Testing Spec 11 Test: Reference Artifact Grouped Model-Output Completeness and Bootstrap Rules](testing-11-reference-artifact-grouped-model-output-completeness-and-bootstrap-rules-v1_0.md)

## Acceptance

This test specification is complete when:

- the child set covers the reusable reference-artifact tooling lanes
- every final child leaf has a paired test specification
- the parent remains a container rather than an executable leaf
