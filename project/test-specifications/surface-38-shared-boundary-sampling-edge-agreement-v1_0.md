# Surface Spec 38 Test: Shared-Boundary Sampling and Edge Agreement

## Overview

This test specification defines verification for seam-first shared-boundary
sampling.

## Backlink

- [Surface Spec 38: Shared-Boundary Sampling and Edge Agreement Rules (v1.0)](../specifications/surface-38-shared-boundary-sampling-edge-agreement-v1_0.md)

## Manual Smoke Check

- Tessellate a shell with two adjacent patches sharing one seam.
- Inspect the seam visually or through debug output and confirm no cracks or
  doubled edge samples appear.

## Automated Smoke Tests

- one canonical sample set is produced per seam/request pair
- adjacent boundary uses reuse the same seam sample sequence

## Automated Acceptance Tests

- shared-boundary vertex coordinates match exactly across adjacent patches
- edge agreement holds without post-weld repair
- changing tessellation request invalidates seam samples deterministically and
  recomputes one new canonical sample set

## Notes

- Use the simplest two-patch shell that still exercises an explicit shared seam.
