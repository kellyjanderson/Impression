# Testing Spec 17: Computer Vision Slice Silhouette Comparison Method and Orientation-Class Taxonomy (v1.0)

## Overview

This specification defines the comparison semantics for slice-silhouette
verification after canonical slice artifacts have been prepared.

## Backlink

- [Testing Spec 04: Computer Vision Slice Silhouette and Orientation-Witness Verification (v1.0)](testing-04-computer-vision-slice-silhouette-and-orientation-witness-verification-v1_0.md)

## Scope

This specification covers:

- the initial silhouette comparison method
- supported same-shape, transformed, and different-shape classes
- orientation-required versus orientation-irrelevant posture
- asymmetric witness adequacy rules

## Behavior

This leaf must define:

- the first comparison method used to classify expected versus actual slices
- how same-shape, rotated-same-shape, and different-shape outcomes are decided
- how orientation-sensitive fixtures declare and justify their witness cue
- whether mirror classification is out of scope or explicitly opted in

## Constraints

- symmetric fixtures must not claim orientation-sensitive proof
- transformed-shape classes must stay distinct from different-shape
- the initial method must stay bounded rather than trying to solve every slice
  case at once

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the initial comparison method is explicit
- result classes and orientation policy are explicit
- witness adequacy posture is explicit
- verification requirements are defined by its paired test specification
