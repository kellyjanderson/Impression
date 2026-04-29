# Surface Spec 102 Test: Surface-Body Boolean Replacement

## Overview

This test specification defines verification for surface-body boolean replacements of deprecated mesh CSG operations.

## Backlink

- [Surface Spec 102: Surface-Body Boolean Replacement (v1.0)](../specifications/surface-102-surface-body-boolean-replacement-v1_0.md)

## Manual Smoke Check

- Run representative union, difference, and intersection cases on surface-native inputs.
- Confirm the outputs remain canonical surface-native results before tessellation.

## Automated Smoke Tests

- boolean operations accept surface-native inputs
- boolean outputs remain valid surface-native results

## Automated Acceptance Tests

- representative boolean cases produce valid preview/export tessellation
- regression fixtures cover holes, seams, and trimmed boundaries
- representative results gain reference images and STL artifacts once implemented
