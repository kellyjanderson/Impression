# Surface Spec 129 Test: Surface Boolean Initial Box Slice Fragment Classification and Split Records

## Overview

This test specification defines verification for fragment classification and
split-record generation on the initial surfaced box slice.

## Backlink

- [Surface Spec 129: Surface Boolean Initial Box Slice Fragment Classification and Split Records (v1.0)](../specifications/surface-129-surface-boolean-initial-box-slice-fragment-classification-and-split-records-v1_0.md)

## Automated Smoke Tests

- representative union, difference, and intersection inputs produce deterministic split records
- fragment labels are produced without demoting the operation to mesh truth

## Automated Acceptance Tests

- inside, outside, and on labels are correct for representative bounded box-slice cases
- operation-aware fragment selection differs correctly across union, difference, and intersection
- split records keep stable links back to cut-curve and trim-fragment inputs
