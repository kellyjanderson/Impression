# Surface Spec 106 Test: Reference Artifact Regression Suite

## Overview

This test specification defines verification for the durable image and STL reference-artifact system.

## Backlink

- [Surface Spec 106: Reference Artifact Regression Suite (v1.0)](../specifications/surface-106-reference-artifact-regression-suite-v1_0.md)

## Manual Smoke Check

- Generate dirty reference artifacts for representative surfacebody and loft cases.
- Inspect that images are viewable and STL files open in a 3D viewer.
- Promote to clean only after explicit visual inspection.

## Automated Smoke Tests

- reference image tests render non-empty model-related images
- reference STL tests export non-empty model-related STL files
- first-run bootstrap writes dirty artifacts under stable project paths for new fixtures

## Automated Acceptance Tests

- clean references are preferred over dirty references when present
- dirty references detect change without silently promoting themselves
- subsequent runs compare against clean when present, otherwise against dirty
- image and STL diffs fail when the fresh output no longer matches the selected baseline
- when a fixture's reference-test contract changes, the old dirty and clean
  references are invalidated and the next intentional run bootstraps a new
  dirty baseline
- representative surfacebody and loft fixtures produce both image and STL references
- model-outputting capabilities are not treated as complete without at least one
  durable named reference fixture
- when a fixture opts into canonical slice verification, the test classifies the
  expected and actual silhouette as same-shape-same-orientation,
  same-shape-rotated, or different-shape
- fixtures that declare orientation-sensitive slice truth fail when the
  comparison falls into the rotated-shape class

## Notes

- Project-specific artifact lifecycle rules live in `project/agents/`.
