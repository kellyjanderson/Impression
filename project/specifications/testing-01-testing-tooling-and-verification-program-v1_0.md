# Testing Spec 01: Testing Tooling and Verification Program (v1.0)

## Overview

This specification defines the top-level testing-tooling and verification
program derived from the testing architecture branch.

## Backlink

- [Testing Architecture](../architecture/testing-architecture.md)

## Scope

This specification covers:

- top-level reusable testing tooling
- reusable verification harness structure
- reference-artifact and CV tooling branches owned by testing rather than by
  feature trunks

## Behavior

This branch must define:

- the child tooling branches that implement reusable testing infrastructure
- the boundary between top-level testing tools and feature trunks that consume
  them

## Constraints

- reusable testing tools must live under the testing program, not under feature
  programs
- feature trunks may depend on testing tools but must not own their structure

## Refinement Status

Decomposed into child branches.

This parent branch does not represent executable work directly.

## Child Specifications

- [Testing Spec 09: Reference Artifact Shared Harness and Lifecycle Program](testing-09-reference-artifact-shared-harness-and-lifecycle-program-v1_0.md)
- [Testing Spec 02: Computer Vision Shared Fixture Contract and Harness Products](testing-02-computer-vision-shared-fixture-contract-and-harness-products-v1_0.md)
- [Testing Spec 03: Computer Vision Text OCR and Glyph Verification](testing-03-computer-vision-text-ocr-and-glyph-verification-v1_0.md)
- [Testing Spec 04: Computer Vision Slice Silhouette and Orientation-Witness Verification](testing-04-computer-vision-slice-silhouette-and-orientation-witness-verification-v1_0.md)
- [Testing Spec 05: Computer Vision Camera and Framing Contract Compliance](testing-05-computer-vision-camera-and-framing-contract-compliance-v1_0.md)
- [Testing Spec 06: Computer Vision Canonical Object-View Render Products and View-Space Verification](testing-06-computer-vision-canonical-object-view-render-products-and-view-space-verification-v1_0.md)
- [Testing Spec 07: Computer Vision Handedness and Mirror-Witness Verification](testing-07-computer-vision-handedness-and-mirror-witness-verification-v1_0.md)
- [Testing Spec 08: Computer Vision Diagnostic Triptych and Panel-Region Presentation](testing-08-computer-vision-diagnostic-triptych-and-panel-region-presentation-v1_0.md)

## Acceptance

This specification is complete when:

- the reusable CV tooling branches are represented as child testing specs
- the testing/program boundary is explicit
- executable work is pushed down into final child leaves
