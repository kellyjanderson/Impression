# Surface Spec 131 Test: Surface Boolean Overlap Cut-Boundary Trim-Loop Reconstruction for Initial Box Slice

## Overview

This test specification defines verification for rebuilding trim loops around
overlap cut boundaries on the initial surfaced box slice.

## Backlink

- [Surface Spec 131: Surface Boolean Overlap Cut-Boundary Trim-Loop Reconstruction for Initial Box Slice (v1.0)](../specifications/surface-131-surface-boolean-overlap-cut-boundary-trim-loop-reconstruction-for-initial-box-slice-v1_0.md)

## Automated Smoke Tests

- representative overlap cases rebuild surfaced trim loops rather than mesh-only cut boundaries
- unaffected source fragments remain reusable where the bounded slice allows it

## Automated Acceptance Tests

- reconstructed trim loops have deterministic orientation and categorization
- representative overlap cases rebuild the expected number of trimmed surviving fragments
- cut-boundary trim reconstruction stays explicit at patch-local parameter-space level
