# Surface Spec 36 Test: Export and Analysis Tessellation Policy

## Overview

This test specification defines verification for export and analysis mode
tessellation of surface-native geometry.

## Backlink

- [Surface Spec 36: Export and Analysis Tessellation Policy Contract (v1.0)](../specifications/surface-36-export-analysis-tessellation-policy-v1_0.md)

## Manual Smoke Check

- Export or analyze a representative closed body.
- Confirm the resulting mesh is denser than preview where expected and suitable
  for downstream analysis/export use.

## Automated Smoke Tests

- export and analysis requests normalize correctly
- export/analysis tessellation returns a valid mesh for valid bodies

## Automated Acceptance Tests

- export/analysis modes are deterministic for identical input and requests
- mode-specific density or tolerance policy is reflected in mesh output
- export/analysis requests preserve modeled meaning while differing from preview
  only in allowed tessellation policy

## Notes

- Pair with cross-mode drift tests where available.
