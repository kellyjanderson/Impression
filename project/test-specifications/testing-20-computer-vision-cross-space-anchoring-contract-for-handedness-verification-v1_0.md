# Testing Spec 20 Test: Computer Vision Cross-Space Anchoring Contract for Handedness Verification

## Overview

This test specification defines verification for the space-alignment
prerequisites of the handedness lane.

## Backlink

- [Testing Spec 20: Computer Vision Cross-Space Anchoring Contract for Handedness Verification (v1.0)](../specifications/testing-20-computer-vision-cross-space-anchoring-contract-for-handedness-verification-v1_0.md)

## Automated Smoke Tests

- representative handedness fixtures declare their modeling/export/viewer space
  relationships
- the lane can reject fixtures that omit required anchoring declarations

## Automated Acceptance Tests

- missing or inconsistent anchoring fails before handedness classification runs
- prerequisite camera/framing and view dependencies remain explicit
- anchoring failure stays distinct from mirrored-result findings
- declared cross-space relationships remain inspectable and stable
