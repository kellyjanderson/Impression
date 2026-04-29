# Testing Spec 04 Test: Computer Vision Slice Silhouette and Orientation-Witness Verification

## Overview

This test specification defines verification for the decomposed slice
silhouette and orientation-witness parent branch in the CV verification
subtree.

## Backlink

- [Testing Spec 04: Computer Vision Slice Silhouette and Orientation-Witness Verification (v1.0)](../specifications/testing-04-computer-vision-slice-silhouette-and-orientation-witness-verification-v1_0.md)

## Scope

This test specification covers:

- child-leaf completeness for slice artifact production and slice comparison
  semantics
- paired verification coverage for final child leaves
- the boundary between expected/actual slice generation and semantic
  classification

## Behavior

This parent test branch must verify:

- no executable slice-artifact or slice-comparison work remains hidden in the
  parent
- the child set covers both normalization/extraction and comparison/taxonomy

## Refinement Status

Decomposed with the parent specification.

This parent test branch does not represent one executable test lane directly.

## Child Test Specifications

- [Testing Spec 16 Test: Computer Vision Slice Artifact Frame, Extraction, and Normalization Contract](testing-16-computer-vision-slice-artifact-frame-extraction-and-normalization-contract-v1_0.md)
- [Testing Spec 17 Test: Computer Vision Slice Silhouette Comparison Method and Orientation-Class Taxonomy](testing-17-computer-vision-slice-silhouette-comparison-method-and-orientation-class-taxonomy-v1_0.md)

## Acceptance

This test specification is complete when:

- the slice child leaves both exist
- every final child leaf has a paired test specification
- the parent remains a container rather than an executable leaf
