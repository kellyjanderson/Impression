# Surface Spec 101 Test: Surface-Native Text Replacement

## Overview

This test specification defines verification for the surface-first replacement of deprecated mesh text generation.

## Backlink

- [Surface Spec 101: Surface-Native Text Replacement (v1.0)](../specifications/surface-101-surface-native-text-replacement-v1_0.md)

## Manual Smoke Check

- Build representative raised or inset text through the surface-first path.
- Confirm the result can be previewed and exported through standard consumer boundaries.

## Automated Smoke Tests

- text replacement terminates in canonical non-mesh-first outputs
- text topology stages remain valid and deterministic

## Automated Acceptance Tests

- surfaced text bodies tessellate non-empty and structurally valid
- representative text examples gain reference images and STL artifacts once implemented

## Notes

- Include examples that prove public text docs remain accurate.
