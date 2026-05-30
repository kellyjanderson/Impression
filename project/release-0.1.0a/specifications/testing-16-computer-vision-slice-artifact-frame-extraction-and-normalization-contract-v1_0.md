# Testing Spec 16: Computer Vision Slice Artifact Frame, Extraction, and Normalization Contract (v1.0)

## Overview

This specification defines the shared local-frame, extraction, and
normalization contract for slice-based CV verification.

## Backlink

- [Testing Spec 04: Computer Vision Slice Silhouette and Orientation-Witness Verification (v1.0)](testing-04-computer-vision-slice-silhouette-and-orientation-witness-verification-v1_0.md)

## Scope

This specification covers:

- fixture-local slice frame declaration
- expected silhouette source declaration
- slice extraction and projection into a comparison frame
- scale and translation normalization posture

## Behavior

This leaf must define:

- which local-frame fields a slice fixture must declare
- how expected and actual slices are projected into one comparison frame
- which normalization steps are allowed and which would hide true drift
- the grouped expected, actual, and diff artifact set for slice lanes

## Constraints

- slice comparison must remain local to the declared fixture frame
- normalization must not erase genuine contour change
- expected and actual slice products must be reproducible and reviewable

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the slice-frame contract is explicit
- extraction and normalization boundaries are explicit
- grouped slice artifact expectations are explicit
- verification requirements are defined by its paired test specification
