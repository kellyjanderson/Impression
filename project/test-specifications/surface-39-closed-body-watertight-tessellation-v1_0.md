# Surface Spec 39 Test: Closed-Body Watertight Tessellation

## Overview

This test specification defines verification for watertight tessellation of
valid closed surface bodies.

## Backlink

- [Surface Spec 39: Closed-Body Watertight Tessellation Contract (v1.0)](../specifications/surface-39-closed-body-watertight-tessellation-v1_0.md)

## Manual Smoke Check

- Tessellate a valid closed body and inspect the result.
- Confirm the output appears closed and exports without obvious seam artifacts.

## Automated Smoke Tests

- valid closed bodies tessellate to a non-empty mesh
- open or invalid bodies do not incorrectly report closed-valid success

## Automated Acceptance Tests

- closed-valid bodies produce meshes with zero boundary edges
- closed-valid bodies produce meshes with zero nonmanifold edges
- closed-valid bodies produce meshes with zero degenerate faces
- watertight success depends on valid upstream seam/boundary truth rather than
  repair passes

## Notes

- Use mesh analysis as the source of truth for watertightness.
