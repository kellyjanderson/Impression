# Surface Spec 35 Test: Preview Tessellation Policy

## Overview

This test specification defines verification for preview-mode tessellation of
surface-native geometry.

## Backlink

- [Surface Spec 35: Preview Tessellation Policy Contract (v1.0)](../specifications/surface-35-preview-tessellation-policy-v1_0.md)

## Manual Smoke Check

- Preview a representative `SurfaceBody`.
- Confirm tessellation is fast, non-empty, and visually stable across repeated
  refreshes.

## Automated Smoke Tests

- preview request normalization yields a bounded tessellation request
- preview tessellation returns a non-empty mesh for valid closed and open bodies

## Automated Acceptance Tests

- preview tessellation remains deterministic for identical input and request
- preview density settings stay within documented preview bounds
- preview tessellation does not mutate surface-kernel state

## Notes

- Use one closed body and one open surface fixture.
