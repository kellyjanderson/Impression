# Surface Spec 96 Test: Loft Surface-Native Cap Construction

## Overview

This test specification defines verification for loft cap construction on the
surface-native path.

## Backlink

- [Surface Spec 96: Loft Surface-Native Cap Construction (v1.0)](../specifications/surface-96-loft-surface-native-cap-construction-v1_0.md)

## Manual Smoke Check

- Run capped loft fixtures through the surfaced loft path.
- Confirm end caps and closure caps are emitted as surface patches.
- Confirm cap patches attach through explicit seam/trim records rather than
  mesh-native closure shortcuts.

## Automated Smoke Tests

- flat start/end caps emit planar surface patches
- loop and region closure ownership emits planar closure-cap patches
- cap patches participate in seam matching against loft sidewalls

## Automated Acceptance Tests

- capped simple lofts classify closed and tessellate watertight
- cap trim boundaries and sidewall seam boundaries align without hidden mesh repair
- non-flat cap modes enter this suite once implemented

## Notes

- This suite should stay open until the surfaced loft path supports the
  intended non-flat cap families as well as the current flat/closure cases.
