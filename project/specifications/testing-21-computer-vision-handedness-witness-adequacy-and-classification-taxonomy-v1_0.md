# Testing Spec 21: Computer Vision Handedness Witness Adequacy and Classification Taxonomy (v1.0)

## Overview

This specification defines the witness requirements and result taxonomy for the
handedness verification lane after cross-space anchoring is satisfied.

## Backlink

- [Testing Spec 07: Computer Vision Handedness and Mirror-Witness Verification (v1.0)](testing-07-computer-vision-handedness-and-mirror-witness-verification-v1_0.md)

## Scope

This specification covers:

- asymmetric witness adequacy rules
- same-handedness, mirrored, and unknown classes
- ambiguity posture for insufficient witnesses
- survival of witness features through the chosen artifact products

## Behavior

This leaf must define:

- which witness features make handedness verification meaningful
- how preserved-handedness, mirrored, and unknown outcomes are classified
- how insufficient-witness cases fail the claimed proof contract honestly
- how witness survival is checked through the products used by the lane

## Constraints

- symmetric fixtures must not claim handedness proof
- ambiguity must remain explicit rather than guessed into success
- witness adequacy must be evaluated against the actual artifact products used

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- witness adequacy rules are explicit
- handedness result classes are explicit
- ambiguity posture is explicit
- verification requirements are defined by its paired test specification
