# Testing Spec 07 Test: Computer Vision Handedness and Mirror-Witness Verification

## Overview

This test specification defines verification for the decomposed handedness and
mirror-witness parent branch in the CV verification subtree.

## Backlink

- [Testing Spec 07: Computer Vision Handedness and Mirror-Witness Verification (v1.0)](../specifications/testing-07-computer-vision-handedness-and-mirror-witness-verification-v1_0.md)

## Scope

This test specification covers:

- child-leaf completeness for cross-space anchoring and witness-based
  classification
- paired verification coverage for final child leaves
- the boundary between space-alignment contracts and handedness result
  semantics

## Behavior

This parent test branch must verify:

- no executable cross-space anchoring or handedness-classification work remains
  hidden in the parent
- the child set covers both anchoring and witness/result taxonomy

## Refinement Status

Decomposed with the parent specification.

This parent test branch does not represent one executable test lane directly.

## Child Test Specifications

- [Testing Spec 20 Test: Computer Vision Cross-Space Anchoring Contract for Handedness Verification](testing-20-computer-vision-cross-space-anchoring-contract-for-handedness-verification-v1_0.md)
- [Testing Spec 21 Test: Computer Vision Handedness Witness Adequacy and Classification Taxonomy](testing-21-computer-vision-handedness-witness-adequacy-and-classification-taxonomy-v1_0.md)

## Acceptance

This test specification is complete when:

- the handedness child leaves both exist
- every final child leaf has a paired test specification
- the parent remains a container rather than an executable leaf
