# Surface Spec 132 Test: Surface Boolean Overlap Shell and Seam Assembly for Initial Box Slice

## Overview

This test specification defines verification for assembling overlap fragments
into surfaced shells and seam records on the initial box slice.

## Backlink

- [Surface Spec 132: Surface Boolean Overlap Shell and Seam Assembly for Initial Box Slice (v1.0)](../specifications/surface-132-surface-boolean-overlap-shell-and-seam-assembly-for-initial-box-slice-v1_0.md)

## Automated Smoke Tests

- representative overlap results expose surfaced shell and seam truth
- shared cut boundaries become explicit seams when the bounded slice requires them

## Automated Acceptance Tests

- shell membership remains deterministic for representative overlap results
- seam-versus-open-boundary ownership is explicit on reconstructed cut boundaries
- overlap result assembly does not rely on mesh stitching as the primary result law
