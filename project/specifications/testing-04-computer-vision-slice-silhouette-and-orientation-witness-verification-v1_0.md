# Testing Spec 04: Computer Vision Slice Silhouette and Orientation-Witness Verification (v1.0)

## Overview

This specification defines the parent branch for architecture-level
slice-silhouette comparison tooling used to verify 3D output in a fixture-local
frame, including the orientation-witness specialization used for asymmetric
notched fixtures.

## Backlink

- [Testing Spec 01: Testing Tooling and Verification Program (v1.0)](testing-01-testing-tooling-and-verification-program-v1_0.md)

## Scope

This specification covers:

- fixture-local slice frames and slice-plane declarations
- expected silhouette sources and normalization policy
- the slice-comparison tooling contract that consumes those artifacts
- grouped expected/actual/diff slice artifacts
- orientation-required versus orientation-irrelevant fixture policy
- witness adequacy rules for asymmetric notch or equivalent cues

## Behavior

This branch must define:

- how expected and actual slice silhouettes are recovered into a shared local
  comparison frame
- the tooling boundary between slice-artifact generation and slice-classifier
  evaluation
- the supported result classes for canonical slice comparison
- how orientation-sensitive fixtures declare and justify their witness cue
- how grouped slice artifacts participate in review and reference lifecycle
  behavior

## Constraints

- slice comparison must stay local to the declared fixture frame
- scale and translation normalization must not hide genuine contour drift
- symmetric fixtures must not claim orientation-sensitive proof
- mirror-specific classification is optional unless a fixture explicitly opts
  into that stronger lane
- this leaf defines reusable verification tooling rather than feature-specific
  loft or boolean behavior

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Testing Spec 16: Computer Vision Slice Artifact Frame, Extraction, and Normalization Contract](testing-16-computer-vision-slice-artifact-frame-extraction-and-normalization-contract-v1_0.md)
- [Testing Spec 17: Computer Vision Slice Silhouette Comparison Method and Orientation-Class Taxonomy](testing-17-computer-vision-slice-silhouette-comparison-method-and-orientation-class-taxonomy-v1_0.md)

## Acceptance

This specification is complete when:

- slice artifact production and slice interpretation semantics are separated
  into honest executable child leaves
- the parent remains a container rather than an executable implementation leaf
- verification requirements are pushed down into the paired child test
  specifications
