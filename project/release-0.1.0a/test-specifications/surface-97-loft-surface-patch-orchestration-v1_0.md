# Surface Spec 97 Test: Loft Split/Merge Surface Patch Orchestration

## Overview

This test specification defines verification for staged split/merge execution
on the surfaced loft path.

## Backlink

- [Surface Spec 97: Loft Split/Merge Surface Patch Orchestration (v1.0)](../specifications/surface-97-loft-surface-patch-orchestration-v1_0.md)

## Manual Smoke Check

- Run one split-birth fixture and one merge-death fixture through the private
  surfaced loft executor.
- Confirm the result is a coordinated patch group rather than a direct
  triangulated mesh artifact.

## Automated Smoke Tests

- split-birth staging emits ruled sidewall patches plus loop closure-cap patches
- merge-death staging emits ruled sidewall patches plus region closure-cap patches
- staged split/merge surfaced bodies tessellate successfully through the
  standard surface boundary

## Automated Acceptance Tests

- staged split-birth bodies with end caps classify closed and tessellate watertight
- staged merge-death bodies with end caps classify closed and tessellate watertight
- staged split/merge surfaced bodies preserve deterministic branch metadata and
  closure ownership in emitted patch records

## Notes

- This spec is satisfied on the private surfaced loft executor before public
  loft promotion.
