# Loft Spec 51 Test: Canonical Station-Slice Silhouette Classification

## Overview

This test specification defines verification for canonical station-slice
silhouette classification in the loft correspondence regression lane.

## Backlink

- [Loft Spec 51: Canonical Station-Slice Silhouette Classification (v1.0)](../specifications/loft-51-canonical-station-slice-silhouette-classification-v1_0.md)

## Automated Smoke Tests

- representative loft fixtures emit expected, actual, and diff station-slice
  bitmaps in a shared comparison frame
- silhouette comparison returns one of the supported relationship classes
- synthetic classifier fixtures run without requiring the full loft reference
  pipeline

## Automated Acceptance Tests

- scaled and translated versions of the same notched silhouette classify as
  `same_shape_same_orientation`
- rotated versions of the same silhouette classify as `same_shape_rotated`
- materially different silhouettes classify as `different_shape`
- representative loft station fixtures fail only when silhouette comparison
  reports `different_shape`, unless the fixture explicitly requires orientation
- expected, actual, and diff station artifacts follow the documented dirty/clean
  reference-image lifecycle
