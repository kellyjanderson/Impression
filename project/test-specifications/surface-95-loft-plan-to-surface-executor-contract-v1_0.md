# Surface Spec 95 Test: Loft Plan-to-Surface Executor Contract

## Overview

This test specification defines verification for the private loft surface
executor that consumes a resolved `LoftPlan` and produces a `SurfaceBody`.

## Backlink

- [Surface Spec 95: Loft Plan-to-Surface Executor Contract (v1.0)](../specifications/surface-95-loft-plan-to-surface-executor-contract-v1_0.md)

## Manual Smoke Check

- Run a representative simple loft plan through the private surface executor.
- Confirm the result is surface-native rather than direct mesh output.
- Confirm the emitted body can flow through standard surface tessellation.

## Automated Smoke Tests

- private loft surface execution returns `SurfaceBody`
- stable loop pairs map to ruled sidewall patches
- simple closure/cap paths emit planar surface patches rather than mesh-native closure triangles

## Automated Acceptance Tests

- adjacent loft intervals reuse station seams in the surfaced result
- loop-wrap seams are represented on surfaced loft sidewalls
- surfaced simple lofts with end caps tessellate closed and watertight through the standard tessellation boundary

## Notes

- This test spec is satisfied on the private migration path before public loft
  promotion.
