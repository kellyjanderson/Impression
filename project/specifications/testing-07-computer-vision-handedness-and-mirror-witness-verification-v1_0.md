# Testing Spec 07: Computer Vision Handedness and Mirror-Witness Verification (v1.0)

## Overview

This specification defines the parent branch for architecture-level handedness
and mirror-witness tooling used to verify left/right preservation across
modeling, export, and viewing spaces.

## Backlink

- [Testing Spec 01: Testing Tooling and Verification Program (v1.0)](testing-01-testing-tooling-and-verification-program-v1_0.md)

## Scope

This specification covers:

- asymmetric witness requirements for handedness or mirror verification
- result classes for same handedness, mirrored output, and unknown orientation
- mirror and handedness interpretation across modeling, export, and viewer
  spaces
- the witness-classification tooling contract for those spaces

## Behavior

This branch must define:

- which witness features make handedness verification meaningful
- the tooling boundary between witness-bearing canonical products and
  handedness/mirror classification
- how mirror or handedness outcomes are classified
- how ambiguity is surfaced when a fixture is too symmetric or the witness is
  insufficient

## Constraints

- symmetric fixtures must not claim handedness proof
- ambiguity must remain explicit rather than guessed into a success class
- witness features must survive the transforms and artifact products used by
  the lane
- this leaf defines reusable tooling rather than feature-level export behavior

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Testing Spec 20: Computer Vision Cross-Space Anchoring Contract for Handedness Verification](testing-20-computer-vision-cross-space-anchoring-contract-for-handedness-verification-v1_0.md)
- [Testing Spec 21: Computer Vision Handedness Witness Adequacy and Classification Taxonomy](testing-21-computer-vision-handedness-witness-adequacy-and-classification-taxonomy-v1_0.md)

## Acceptance

This specification is complete when:

- cross-space anchoring and handedness classification are separated into honest
  executable child leaves
- the parent remains a container rather than an executable implementation leaf
- verification requirements are pushed down into the paired child test
  specifications
